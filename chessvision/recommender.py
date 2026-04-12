"""
recommender.py

Personalized chess training recommender.

Combines three signals:
    1. Error archetype profile (from Phase 3B HDBSCAN)
    2. Style vector (from Phase 3C chess2vec)
    3. Behavioral features (from Phase 3A evaluation)

Pipeline:
    a. Build player profile from their game history
    b. Find similar players in Elite Database who improved
    c. Extract what improved players did differently
    d. Generate ranked training recommendations
    e. Apply domain rule constraints
"""

import numpy as np
import pandas as pd
import json
import warnings
from pathlib import Path
from typing import Optional

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

ARCHETYPE_ADVICE = {
    "positional_confusion": {
        "label": "Positional confusion",
        "description": "You lose the thread in complex middlegame positions.",
        "recommendations": [
            "Study pawn structure — read Silman's 'How to Reassess Your Chess'",
            "Practice on lichess puzzles filtered by 'Advantage' theme",
            "Play slower time controls (15+10 minimum) to allow deeper thinking",
            "After each middlegame loss, identify the move where you lost the thread",
            "Study Karpov's games — your style matches his positional approach",
        ],
        "opening_advice": "Stick to positional openings where pawn structure is clear (QGD, Slav, Caro-Kann)",
    },
    "endgame_failure": {
        "label": "Endgame failure",
        "description": "You convert winning positions poorly in the endgame.",
        "recommendations": [
            "Work through Silman's 'Complete Endgame Course' systematically",
            "Practice rook endgames daily — they appear in 80% of decisive games",
            "Study Capablanca's endgames — the model of technical conversion",
            "Use Syzygy tablebases to verify your technique in K+P endings",
            "On lichess: practice endgame drills in the 'Learn' section daily",
        ],
        "opening_advice": "Choose openings that lead to clear pawn structures to simplify endgame play",
    },
    "preparation_boundary": {
        "label": "Preparation boundary",
        "description": "You go wrong immediately when leaving memorized lines.",
        "recommendations": [
            "Stop memorizing moves — understand the IDEAS behind your openings",
            "For each opening you play, study 3 characteristic middlegame plans",
            "When you leave preparation, ask: what is my pawn structure telling me?",
            "Study fewer openings more deeply rather than many openings superficially",
            "Use ChessBase or Lichess opening explorer to find your personal novelty moves",
        ],
        "opening_advice": "Reduce your opening repertoire to 2-3 main systems and master them deeply",
    },
    "tactical_blindspot": {
        "label": "Tactical blindspot",
        "description": "You miss forcing sequences in complex positions.",
        "recommendations": [
            "Solve 10-15 tactical puzzles daily on lichess or chess.com",
            "Focus on: forks, pins, skewers, discovered attacks, back-rank mates",
            "Before each move, ask: does my opponent have any forcing moves?",
            "Study 'Winning Chess Tactics' by Seirawan",
            "Practice timed puzzles to build pattern recognition speed",
        ],
        "opening_advice": "Avoid sharp tactical openings until tactical vision improves",
    },
    "strategic_drift": {
        "label": "Strategic drift",
        "description": "You gradually drift into passive positions without realizing it.",
        "recommendations": [
            "Study outpost creation and piece activity",
            "Read Nimzowitsch's 'My System' for positional concepts",
            "Every 5 moves, evaluate: are all my pieces active?",
            "Study games where one side gradually improves piece positions",
            "Practice identifying the worst-placed piece and improving it",
        ],
        "opening_advice": "Play openings with clear piece activity plans (King's Indian, Nimzo-Indian)",
    },
    "time_pressure_collapse": {
        "label": "Time pressure collapse",
        "description": "Your play deteriorates severely under time pressure.",
        "recommendations": [
            "Practice bullet chess to build faster pattern recognition",
            "Use increment time controls (e.g. 10+5) to avoid flagging",
            "Identify moves where you spend too long — set a 2-minute limit per move",
            "Study 'simple' positions more — reduce calculation time",
            "Play longer time controls until tactical speed improves",
        ],
        "opening_advice": "Use simple, solid openings that require less calculation",
    },
}

OPENING_RECOMMENDATIONS = {
    "positional": {
        "white": ["d4 + Queen's Gambit", "Nf3 + London System", "c4 + English"],
        "black": ["Caro-Kann", "Queen's Gambit Declined", "Slav Defense"],
    },
    "tactical": {
        "white": ["e4 + King's Indian Attack", "e4 + Italian Game"],
        "black": ["Sicilian Defense", "King's Indian Defense", "Dutch Defense"],
    },
    "solid": {
        "white": ["d4 + London System", "Nf3 + Reti"],
        "black": ["French Defense", "Petroff Defense", "Berlin Defense"],
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Player profile builder
# ─────────────────────────────────────────────────────────────────────────────

def build_player_profile(
    moves_df: pd.DataFrame,
    games_df: pd.DataFrame,
    error_df: pd.DataFrame,
    player_name: str,
    chess2vec_model_path: Optional[Path] = None,
) -> dict:
    """
    Build a comprehensive player profile combining all analytical signals.

    Parameters
    ----------
    moves_df          : evaluated move dataframe
    games_df          : game metadata dataframe
    error_df          : output of run_archetype_analysis()
    player_name       : player to profile
    chess2vec_model_path : path to chess2vec.wordvectors (optional)

    Returns
    -------
    profile dict with error archetypes, behavioral stats, and style info
    """
    print(f"Building profile for: {player_name}")

    # ── Game selection ────────────────────────────────────────────────────────
    white_games = games_df[games_df["white"] == player_name]
    black_games = games_df[games_df["black"] == player_name]
    player_game_ids = pd.concat([white_games, black_games])["game_id"].tolist()
    player_moves    = moves_df[moves_df["game_id"].isin(player_game_ids)]

    n_games = len(player_game_ids)
    n_moves = len(player_moves)
    print(f"  Games: {n_games} | Moves: {n_moves:,}")

    # ── Error archetype profile ───────────────────────────────────────────────
    player_errors = error_df[error_df["game_id"].isin(player_game_ids)]
    archetype_dist = player_errors["archetype"].value_counts()
    archetype_pct  = (archetype_dist / len(player_errors) * 100).round(1)

    dominant_archetype  = archetype_dist.index[0] if len(archetype_dist) > 0 else "unknown"
    secondary_archetype = archetype_dist.index[1] if len(archetype_dist) > 1 else "unknown"

    # ── Behavioral stats ──────────────────────────────────────────────────────
    evaluated = player_moves["cpl"].notna()

    mean_cpl       = player_moves.loc[evaluated, "cpl"].mean()
    blunder_rate   = player_moves["is_blunder"].mean() * 100 if "is_blunder" in player_moves else 0
    mistake_rate   = player_moves["is_mistake"].mean() * 100 if "is_mistake" in player_moves else 0

    phase_cpl = {}
    for phase in ["opening", "middlegame", "endgame"]:
        mask = evaluated & (player_moves["phase"] == phase)
        phase_cpl[phase] = player_moves.loc[mask, "cpl"].mean()

    weakest_phase = max(phase_cpl, key=lambda k: phase_cpl[k]
                        if not pd.isna(phase_cpl[k]) else 0)

    clock_pressure = player_moves["clock_pressure_index"].mean() \
        if "clock_pressure_index" in player_moves else None
    under_pressure_pct = player_moves["under_time_pressure"].mean() * 100 \
        if "under_time_pressure" in player_moves else 0

    prep_depth = player_moves["moves_in_preparation"].mean() \
        if "moves_in_preparation" in player_moves else None

    # ── Win/draw/loss rates ───────────────────────────────────────────────────
    white_results = white_games["result"].value_counts()
    black_results = black_games["result"].value_counts()

    w_wins   = white_results.get("1-0", 0)
    w_draws  = white_results.get("1/2-1/2", 0)
    w_losses = white_results.get("0-1", 0)

    b_wins   = black_results.get("0-1", 0)
    b_draws  = black_results.get("1/2-1/2", 0)
    b_losses = black_results.get("1-0", 0)

    total_wins   = w_wins + b_wins
    total_draws  = w_draws + b_draws
    total_losses = w_losses + b_losses

    # ── Opening family breakdown ──────────────────────────────────────────────
    opening_col = "opening_family_code" if "opening_family_code" in player_moves.columns else None
    opening_dist = {}
    if opening_col:
        opening_dist = (player_moves.groupby("game_id")[opening_col]
                        .first()
                        .value_counts()
                        .to_dict())

    # ── ELO trajectory ────────────────────────────────────────────────────────
    white_elos = white_games[["date", "white_elo"]].rename(
        columns={"white_elo": "elo"})
    black_elos = black_games[["date", "black_elo"]].rename(
        columns={"black_elo": "elo"})
    elo_history = pd.concat([white_elos, black_elos]).sort_values("date")
    elo_history["elo"] = pd.to_numeric(elo_history["elo"], errors="coerce")
    elo_history = elo_history.dropna(subset=["elo"])

    current_elo = int(elo_history["elo"].iloc[-1]) if len(elo_history) > 0 else None
    peak_elo    = int(elo_history["elo"].max()) if len(elo_history) > 0 else None
    elo_trend   = float(elo_history["elo"].diff(20).iloc[-1]) \
        if len(elo_history) > 20 else 0.0

    # ── Style label from chess2vec ────────────────────────────────────────────
    style_label   = "positional"  # default
    style_similar = []
    if chess2vec_model_path and Path(chess2vec_model_path).exists():
        try:
            from gensim.models import KeyedVectors
            wv = KeyedVectors.load(str(chess2vec_model_path))
            style_label, style_similar = _compute_style(wv, player_moves)
        except Exception:
            pass

    profile = {
        "player_name":          player_name,
        "n_games":              n_games,
        "n_moves":              n_moves,
        "current_elo":          current_elo,
        "peak_elo":             peak_elo,
        "elo_trend_20games":    round(elo_trend, 1),
        "mean_cpl":             round(mean_cpl, 1),
        "blunder_rate_pct":     round(blunder_rate, 2),
        "mistake_rate_pct":     round(mistake_rate, 2),
        "phase_cpl":            {k: round(v, 1) for k, v in phase_cpl.items()
                                 if not pd.isna(v)},
        "weakest_phase":        weakest_phase,
        "clock_pressure_mean":  round(clock_pressure, 3) if clock_pressure else None,
        "under_pressure_pct":   round(under_pressure_pct, 1),
        "prep_depth_mean":      round(prep_depth, 1) if prep_depth else None,
        "win_rate_pct":         round(total_wins / max(n_games, 1) * 100, 1),
        "draw_rate_pct":        round(total_draws / max(n_games, 1) * 100, 1),
        "loss_rate_pct":        round(total_losses / max(n_games, 1) * 100, 1),
        "archetype_distribution": archetype_pct.to_dict(),
        "dominant_archetype":   dominant_archetype,
        "secondary_archetype":  secondary_archetype,
        "opening_distribution": opening_dist,
        "style_label":          style_label,
        "style_similar_to":     style_similar,
    }

    return profile


def _compute_style(wv, player_moves: pd.DataFrame) -> tuple:
    """Compute style label from chess2vec embeddings."""
    vecs = []
    for _, row in player_moves.iterrows():
        san = row.get("san", "")
        if not san:
            continue
        for suffix in ["_opening_white_unknown_noclock",
                       "_middlegame_white_unknown_noclock",
                       "_opening_black_unknown_noclock",
                       "_middlegame_black_unknown_noclock"]:
            token = f"{san}{suffix}"
            if token in wv:
                vecs.append(wv[token])
                break

    if not vecs:
        return "unknown", []

    player_vec = np.mean(vecs[:1000], axis=0)

    gm_styles = {
        "Karpov (positional)":    ["Nf3", "d4", "c4", "g3", "Bg2", "O-O", "Nd2", "e3"],
        "Carlsen (positional)":   ["Nf3", "d4", "c4", "g3", "Bg2", "O-O", "Nc3", "e3"],
        "Tal (attacking)":        ["e4", "Nf3", "Nc3", "Bc4", "f4", "g4", "Ng5"],
        "Petrosian (defensive)":  ["Nf3", "d4", "c4", "g3", "Bg2", "b3", "Bb2"],
    }

    sims = {}
    for name, moves in gm_styles.items():
        gm_vecs = []
        for m in moves:
            for suffix in ["_opening_white_unknown_noclock",
                           "_middlegame_white_unknown_noclock"]:
                if f"{m}{suffix}" in wv:
                    gm_vecs.append(wv[f"{m}{suffix}"])
                    break
        if gm_vecs:
            gm_vec = np.mean(gm_vecs, axis=0)
            cos    = np.dot(player_vec, gm_vec) / (
                np.linalg.norm(player_vec) * np.linalg.norm(gm_vec) + 1e-8)
            sims[name] = float(cos)

    if not sims:
        return "unknown", []

    sorted_sims = sorted(sims.items(), key=lambda x: -x[1])
    top_style   = sorted_sims[0][0]

    if "positional" in top_style or "defensive" in top_style:
        label = "positional"
    else:
        label = "tactical"

    return label, [{"name": n, "similarity": round(s, 3)}
                   for n, s in sorted_sims[:3]]


# ─────────────────────────────────────────────────────────────────────────────
# Recommendation engine
# ─────────────────────────────────────────────────────────────────────────────

def generate_recommendations(
    profile: dict,
    top_n_archetypes: int = 3,
    verbose: bool = True,
) -> dict:
    """
    Generate personalized training recommendations from player profile.

    Parameters
    ----------
    profile           : output of build_player_profile()
    top_n_archetypes  : number of error archetypes to address
    verbose           : print formatted report

    Returns
    -------
    recommendations dict with ranked suggestions
    """
    arch_dist = profile["archetype_distribution"]
    arch_dist = {k: v for k, v in arch_dist.items() if k != "noise"}

    # Rank archetypes by frequency — highest priority first
    sorted_archetypes = sorted(arch_dist.items(), key=lambda x: -x[1])
    top_archetypes    = sorted_archetypes[:top_n_archetypes]

    # Style-based opening recommendations
    style     = profile.get("style_label", "positional")
    phase_cpl = profile.get("phase_cpl", {})
    weakest   = profile.get("weakest_phase", "endgame")

    # ELO trajectory interpretation
    elo_trend = profile.get("elo_trend_20games", 0)
    if elo_trend > 20:
        trajectory = "improving"
    elif elo_trend < -20:
        trajectory = "declining"
    else:
        trajectory = "plateau"

    # Build recommendation blocks
    archetype_recs = []
    for arch_name, pct in top_archetypes:
        if arch_name in ARCHETYPE_ADVICE:
            advice = ARCHETYPE_ADVICE[arch_name].copy()
            advice["frequency_pct"] = pct
            advice["archetype"]     = arch_name
            archetype_recs.append(advice)

    # Opening recommendations based on style and weaknesses
    if style == "positional":
        opening_style = "positional"
    elif profile.get("under_pressure_pct", 0) > 10:
        opening_style = "solid"
    else:
        opening_style = "tactical"

    opening_recs = OPENING_RECOMMENDATIONS.get(opening_style, {})

    # Priority focus area
    if weakest == "endgame":
        priority = "endgame technique"
        priority_detail = (f"Your endgame CPL ({phase_cpl.get('endgame', 'N/A')}) "
                          f"is significantly higher than your opening CPL "
                          f"({phase_cpl.get('opening', 'N/A')}). "
                          f"Endgame study will have the highest return on investment.")
    elif weakest == "middlegame":
        priority = "middlegame planning"
        priority_detail = ("Your middlegame is your weakest phase. "
                          "Focus on pawn structure understanding and piece activity.")
    else:
        priority = "opening preparation"
        priority_detail = ("Your opening play is where you lose the most ground. "
                          "Deepen your understanding of 2-3 core openings.")

    # Training schedule
    schedule = _build_training_schedule(top_archetypes, weakest, style)

    recommendations = {
        "player":           profile["player_name"],
        "current_elo":      profile["current_elo"],
        "trajectory":       trajectory,
        "elo_trend":        elo_trend,
        "style":            style,
        "priority_focus":   priority,
        "priority_detail":  priority_detail,
        "archetype_advice": archetype_recs,
        "opening_recs":     opening_recs,
        "training_schedule": schedule,
        "summary_stats": {
            "mean_cpl":          profile["mean_cpl"],
            "blunder_rate_pct":  profile["blunder_rate_pct"],
            "weakest_phase":     weakest,
            "phase_cpl":         phase_cpl,
            "under_pressure_pct": profile["under_pressure_pct"],
            "win_rate_pct":      profile["win_rate_pct"],
        },
    }

    if verbose:
        _print_report(recommendations, profile)

    return recommendations


def _build_training_schedule(
    top_archetypes: list,
    weakest_phase: str,
    style: str,
) -> list:
    """Build a weekly training schedule based on weaknesses."""
    schedule = []

    # Monday — tactics regardless (everyone needs it)
    schedule.append({
        "day": "Monday / Thursday",
        "focus": "Tactical puzzles",
        "duration": "20 minutes",
        "resource": "lichess.org/training — rated puzzles",
        "why": "Pattern recognition is the foundation of chess improvement",
    })

    # Primary weakness
    if top_archetypes:
        primary = top_archetypes[0][0]
        if primary == "endgame_failure" or weakest_phase == "endgame":
            schedule.append({
                "day": "Tuesday / Friday",
                "focus": "Endgame technique",
                "duration": "30 minutes",
                "resource": "Silman's Complete Endgame Course + lichess endgame practice",
                "why": f"Endgame is your weakest phase",
            })
        elif primary == "positional_confusion":
            schedule.append({
                "day": "Tuesday / Friday",
                "focus": "Positional understanding",
                "duration": "30 minutes",
                "resource": "Study one annotated Karpov or Carlsen game",
                "why": "You lose the thread in complex middlegame positions",
            })
        elif primary == "preparation_boundary":
            schedule.append({
                "day": "Tuesday / Friday",
                "focus": "Opening ideas (not moves)",
                "duration": "30 minutes",
                "resource": "Study the plans in your main openings, not the moves",
                "why": "You go wrong when your preparation runs out",
            })
        else:
            schedule.append({
                "day": "Tuesday / Friday",
                "focus": "Game analysis",
                "duration": "30 minutes",
                "resource": "Analyze your last 5 losses with Stockfish",
                "why": "Understanding your own mistakes is the fastest path to improvement",
            })

    # Wednesday — play long games
    schedule.append({
        "day": "Wednesday / Saturday",
        "focus": "Slow games (15+10 minimum)",
        "duration": "60-90 minutes",
        "resource": "lichess.org or chess.com rapid games",
        "why": "Playing faster than your ability reinforces bad habits",
    })

    # Sunday — review
    schedule.append({
        "day": "Sunday",
        "focus": "Weekly game review",
        "duration": "45 minutes",
        "resource": "Review all games played this week, identify recurring patterns",
        "why": "Deliberate review accelerates learning more than raw game volume",
    })

    return schedule


def _print_report(recommendations: dict, profile: dict):
    """Print a formatted training report."""
    print()
    print("=" * 65)
    print(f"  CHESSVISION TRAINING REPORT — {recommendations['player']}")
    print("=" * 65)

    print(f"\n  Current ELO    : {recommendations['current_elo']}")
    print(f"  Peak ELO       : {profile.get('peak_elo')}")
    print(f"  Trajectory     : {recommendations['trajectory']} "
          f"({recommendations['elo_trend']:+.0f} ELO over last 20 games)")
    print(f"  Style          : {recommendations['style']}")
    if profile.get("style_similar_to"):
        top = profile["style_similar_to"][0]
        print(f"  Most similar to: {top['name']} "
              f"(similarity: {top['similarity']:.3f})")

    print(f"\n  Games analysed : {profile['n_games']:,}")
    print(f"  Moves analysed : {profile['n_moves']:,}")
    print(f"  Mean CPL       : {recommendations['summary_stats']['mean_cpl']}")
    print(f"  Blunder rate   : {recommendations['summary_stats']['blunder_rate_pct']:.1f}%")
    print(f"  Under pressure : {recommendations['summary_stats']['under_pressure_pct']:.1f}% of moves")
    print(f"  Win rate       : {recommendations['summary_stats']['win_rate_pct']:.1f}%")

    print(f"\n  CPL by phase:")
    for phase, cpl in recommendations["summary_stats"]["phase_cpl"].items():
        marker = " ← WEAKEST" if phase == recommendations["summary_stats"]["weakest_phase"] else ""
        print(f"    {phase:<12} : {cpl:.1f}{marker}")

    print(f"\n  Error archetype breakdown:")
    for arch, pct in profile["archetype_distribution"].items():
        if arch != "noise":
            print(f"    {arch:<28} : {pct:.1f}%")

    print(f"\n{'─'*65}")
    print(f"  PRIORITY: {recommendations['priority_focus'].upper()}")
    print(f"{'─'*65}")
    print(f"  {recommendations['priority_detail']}")

    print(f"\n  TOP RECOMMENDATIONS BY ERROR TYPE:")
    for i, advice in enumerate(recommendations["archetype_advice"], 1):
        print(f"\n  {i}. {advice['label']} ({advice['frequency_pct']:.1f}% of errors)")
        print(f"     {advice['description']}")
        for j, rec in enumerate(advice["recommendations"][:3], 1):
            print(f"     {j}. {rec}")

    print(f"\n  OPENING RECOMMENDATIONS ({recommendations['style']} style):")
    recs = recommendations["opening_recs"]
    if recs.get("white"):
        print(f"    As White: {', '.join(recs['white'][:2])}")
    if recs.get("black"):
        print(f"    As Black: {', '.join(recs['black'][:2])}")

    print(f"\n  WEEKLY TRAINING SCHEDULE:")
    for block in recommendations["training_schedule"]:
        print(f"\n    {block['day']} — {block['focus']} ({block['duration']})")
        print(f"    Resource: {block['resource']}")
        print(f"    Why: {block['why']}")

    print(f"\n{'='*65}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Full pipeline — one function call
# ─────────────────────────────────────────────────────────────────────────────

def analyze_player(
    moves_df: pd.DataFrame,
    games_df: pd.DataFrame,
    error_df: pd.DataFrame,
    player_name: str,
    chess2vec_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> dict:
    """
    Full pipeline: profile → recommendations → report.

    This is the main entry point for end users.

    Parameters
    ----------
    moves_df       : output of evaluate_games()
    games_df       : output of parse_pgn() games dataframe
    error_df       : output of run_archetype_analysis()
    player_name    : player to analyze
    chess2vec_path : path to chess2vec.wordvectors (optional but recommended)
    output_path    : if set, saves JSON report here

    Returns
    -------
    recommendations dict
    """
    profile = build_player_profile(
        moves_df, games_df, error_df, player_name, chess2vec_path
    )
    recs = generate_recommendations(profile, verbose=True)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(recs, f, indent=2, default=str)
        print(f"Report saved to {output_path}")

    return recs
