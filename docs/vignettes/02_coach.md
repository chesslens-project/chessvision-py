# Vignette 2 — Comparing multiple players (coach)

Analyze and compare multiple students' game archives.

## Setup

````python
import chessvision as cv
import pandas as pd

students = {
    "Alice": "alice_games.pgn",
    "Bob":   "bob_games.pgn",
    "Carol": "carol_games.pgn",
}
````

## Analyze each student

````python
profiles = {}
for name, pgn in students.items():
    games, moves = cv.parse_pgn(pgn)
    moves        = cv.evaluate_games(moves)
    moves        = cv.engineer_features(games, moves)
    errors       = cv.run_archetype_analysis(moves)
    profile      = cv.build_player_profile(moves, games, errors, name)
    profiles[name] = profile
````

## Compare profiles

````python
comparison = pd.DataFrame({
    name: {
        "ELO":              p["current_elo"],
        "Mean CPL":         p["mean_cpl"],
        "Blunder rate %":   p["blunder_rate_pct"],
        "Weakest phase":    p["weakest_phase"],
        "Primary error":    p["dominant_archetype"],
    }
    for name, p in profiles.items()
}).T

print(comparison)
````

## Generate individual reports

````python
for name, profile in profiles.items():
    recs = cv.generate_recommendations(profile, verbose=False)
    print(f"\n{name}: focus on {recs['priority_focus']}")
