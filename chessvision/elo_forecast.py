"""
elo_forecast.py

LSTM-based ELO trajectory forecasting.

Takes a player's game history and predicts ELO change
over the next N games based on behavioral features.

Architecture:
    Input  : sequence of game-level feature vectors
    LSTM   : 1 layer, hidden_size=128, dropout=0.2
    Output : predicted ELO change over next 10 games
"""

import numpy as np
import pandas as pd
import json
import joblib
from pathlib import Path
from typing import Optional, Tuple

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# ─────────────────────────────────────────────────────────────────────────────
# Feature engineering — game level
# ─────────────────────────────────────────────────────────────────────────────

GAME_FEATURES = [
    "accuracy_score",
    "mean_cpl",
    "blunder_rate",
    "mistake_rate",
    "inaccuracy_rate",
    "opening_accuracy",
    "middlegame_accuracy",
    "endgame_accuracy",
    "mean_clock_pressure",
    "under_pressure_rate",
    "mean_complexity",
    "mean_mobility_ratio",
    "mean_moves_in_prep",
    "game_length",
    "opponent_elo_delta",
]


def compute_accuracy_score(cpl_series: pd.Series) -> float:
    """
    Convert mean CPL to accuracy score (0-100).
    Uses lichess formula: accuracy = 103.1668 * exp(-0.04354 * cpl) - 3.1668
    """
    mean_cpl = cpl_series.mean()
    if pd.isna(mean_cpl):
        return np.nan
    score = 103.1668 * np.exp(-0.04354 * mean_cpl) - 3.1668
    return float(np.clip(score, 0, 100))


def build_game_features(
    moves_df: pd.DataFrame,
    games_df: pd.DataFrame,
    player_name: str,
) -> pd.DataFrame:
    """
    Aggregate move-level features to game level for a specific player.

    Returns one row per game with behavioral features,
    sorted chronologically by date.
    """
    # Identify player's games and their color
    white_games = games_df[games_df["white"] == player_name].copy()
    black_games = games_df[games_df["black"] == player_name].copy()

    white_games["player_color"] = "white"
    black_games["player_color"] = "black"
    white_games["player_elo"]   = white_games["white_elo"]
    black_games["player_elo"]   = black_games["black_elo"]
    white_games["opp_elo"]      = white_games["black_elo"]
    black_games["opp_elo"]      = black_games["white_elo"]

    player_games = pd.concat([white_games, black_games])
    player_games = player_games.sort_values("date").reset_index(drop=True)

    if len(player_games) < 10:
        raise ValueError(
            f"Player '{player_name}' has only {len(player_games)} games. "
            f"Need at least 10."
        )

    rows = []
    for _, game_row in player_games.iterrows():
        gid         = game_row["game_id"]
        color       = game_row["player_color"]
        game_moves  = moves_df[
            (moves_df["game_id"] == gid) &
            (moves_df["color"]   == color)
        ]

        if len(game_moves) == 0:
            continue

        evaluated = game_moves["cpl"].notna()

        row = {
            "game_id":            gid,
            "date":               game_row["date"],
            "player_elo":         game_row["player_elo"],
            "opp_elo":            game_row.get("opp_elo", np.nan),
            "result":             game_row["result"],
            "color":              color,
            "accuracy_score":     compute_accuracy_score(
                                      game_moves.loc[evaluated, "cpl"]),
            "mean_cpl":           game_moves["cpl"].mean(),
            "blunder_rate":       game_moves["is_blunder"].mean()
                                  if "is_blunder" in game_moves else np.nan,
            "mistake_rate":       game_moves["is_mistake"].mean()
                                  if "is_mistake" in game_moves else np.nan,
            "inaccuracy_rate":    game_moves["is_inaccuracy"].mean()
                                  if "is_inaccuracy" in game_moves else np.nan,
            "opening_accuracy":   compute_accuracy_score(
                                      game_moves.loc[
                                          evaluated & (game_moves["phase"] == "opening"),
                                          "cpl"
                                      ]),
            "middlegame_accuracy": compute_accuracy_score(
                                      game_moves.loc[
                                          evaluated & (game_moves["phase"] == "middlegame"),
                                          "cpl"
                                      ]),
            "endgame_accuracy":   compute_accuracy_score(
                                      game_moves.loc[
                                          evaluated & (game_moves["phase"] == "endgame"),
                                          "cpl"
                                      ]),
            "mean_clock_pressure": game_moves["clock_pressure_index"].mean()
                                   if "clock_pressure_index" in game_moves
                                   else np.nan,
            "under_pressure_rate": game_moves["under_time_pressure"].mean()
                                   if "under_time_pressure" in game_moves
                                   else np.nan,
            "mean_complexity":    game_moves["position_complexity_score"].mean()
                                  if "position_complexity_score" in game_moves
                                  else np.nan,
            "mean_mobility_ratio": game_moves["mobility_ratio"].mean()
                                   if "mobility_ratio" in game_moves
                                   else np.nan,
            "mean_moves_in_prep": game_moves["moves_in_preparation"].mean()
                                  if "moves_in_preparation" in game_moves
                                  else np.nan,
            "game_length":        len(game_moves),
            "opponent_elo_delta": (game_row.get("opp_elo", np.nan) -
                                   game_row["player_elo"]),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df["player_elo"] = pd.to_numeric(df["player_elo"], errors="coerce")

    # Compute ELO change (target variable)
    df["elo_change_next10"] = (
        df["player_elo"].shift(-10) - df["player_elo"]
    )

    return df.fillna(df.median(numeric_only=True))


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class ELODataset(Dataset):
    """
    Sliding window dataset for LSTM training.
    Each sample: sequence of SEQ_LEN games → ELO change in next 10 games.
    """
    def __init__(
        self,
        game_df: pd.DataFrame,
        seq_len: int = 20,
        feature_cols: list = GAME_FEATURES,
    ):
        self.seq_len  = seq_len
        self.features = feature_cols
        self.X        = []
        self.y        = []

        data = game_df.dropna(
            subset=feature_cols + ["elo_change_next10"]
        ).reset_index(drop=True)

        for i in range(len(data) - seq_len):
            seq    = data.iloc[i:i+seq_len][feature_cols].values.astype(np.float32)
            target = data.iloc[i+seq_len]["elo_change_next10"]
            self.X.append(seq)
            self.y.append(float(target))

        self.X = np.array(self.X)
        self.y = np.array(self.y, dtype=np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])


# ─────────────────────────────────────────────────────────────────────────────
# LSTM model
# ─────────────────────────────────────────────────────────────────────────────

class ELOForecaster(nn.Module):
    """
    LSTM model for ELO trajectory forecasting.

    Input  : (batch, seq_len, n_features)
    Output : (batch, 1) — predicted ELO change over next 10 games
    """
    def __init__(
        self,
        input_size:  int,
        hidden_size: int = 128,
        num_layers:  int = 1,
        dropout:     float = 0.2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            dropout     = dropout if num_layers > 1 else 0,
            batch_first = True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1     = nn.Linear(hidden_size, 64)
        self.relu    = nn.ReLU()
        self.fc2     = nn.Linear(64, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out     = out[:, -1, :]  # take last timestep
        out     = self.dropout(out)
        out     = self.relu(self.fc1(out))
        return self.fc2(out).squeeze(-1)


# ─────────────────────────────────────────────────────────────────────────────
# Training pipeline
# ─────────────────────────────────────────────────────────────────────────────

def train_elo_model(
    game_df: pd.DataFrame,
    seq_len:     int   = 20,
    hidden_size: int   = 128,
    epochs:      int   = 50,
    batch_size:  int   = 32,
    lr:          float = 0.001,
    patience:    int   = 10,
    output_dir:  Optional[Path] = None,
) -> Tuple[ELOForecaster, StandardScaler, dict]:
    """
    Train the LSTM ELO forecasting model.

    Parameters
    ----------
    game_df     : game-level feature dataframe from build_game_features()
    seq_len     : number of past games to use as input sequence
    hidden_size : LSTM hidden dimension
    epochs      : maximum training epochs
    batch_size  : training batch size
    lr          : learning rate
    patience    : early stopping patience
    output_dir  : if set, saves model here

    Returns
    -------
    model, scaler, training_history
    """
    print("=" * 60)
    print("Phase 3D — ELO Trajectory Forecasting (LSTM)")
    print("=" * 60)

    # Scale features
    scaler   = StandardScaler()
    data     = game_df.copy()
    feat_cols = [c for c in GAME_FEATURES if c in data.columns]
    data[feat_cols] = scaler.fit_transform(data[feat_cols])

    # Train/val/test split (chronological — never shuffle time series)
    n        = len(data)
    train_df = data.iloc[:int(n * 0.7)]
    val_df   = data.iloc[int(n * 0.7):int(n * 0.85)]
    test_df  = data.iloc[int(n * 0.85):]

    print(f"\nDataset split:")
    print(f"  Train : {len(train_df)} games")
    print(f"  Val   : {len(val_df)} games")
    print(f"  Test  : {len(test_df)} games")
    print(f"  Features: {len(feat_cols)}")
    print(f"  Seq len : {seq_len}")

    train_ds = ELODataset(train_df, seq_len, feat_cols)
    val_ds   = ELODataset(val_df,   seq_len, feat_cols)
    test_ds  = ELODataset(test_df,  seq_len, feat_cols)

    if len(train_ds) < 10:
        raise ValueError(
            f"Not enough training samples ({len(train_ds)}). "
            f"Need more games. Try reducing seq_len."
        )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    # Model
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cpu")
    print(f"  Device  : {device}")
    print()

    model = ELOForecaster(
        input_size  = len(feat_cols),
        hidden_size = hidden_size,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5, verbose=False
    )
    criterion = nn.MSELoss()

    history        = {"train_loss": [], "val_loss": []}
    best_val_loss  = float("inf")
    best_weights   = None
    patience_count = 0

    print(f"Training for up to {epochs} epochs "
          f"(early stopping patience={patience})...")

    for epoch in range(epochs):
        # Train
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        # Validate
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                pred    = model(X_batch)
                loss    = criterion(pred, y_batch)
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss   = np.mean(val_losses)
        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        scheduler.step(val_loss)

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs} | "
                  f"train loss: {train_loss:.2f} | "
                  f"val loss: {val_loss:.2f}")

        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            best_weights   = {k: v.clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"\n  Early stopping at epoch {epoch+1}")
                break

    # Restore best weights
    model.load_state_dict(best_weights)

    # Test evaluation
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            pred = model(X_batch.to(device)).cpu().numpy()
            all_preds.extend(pred)
            all_targets.extend(y_batch.numpy())

    all_preds   = np.array(all_preds)
    all_targets = np.array(all_targets)
    mae         = np.mean(np.abs(all_preds - all_targets))
    rmse        = np.sqrt(np.mean((all_preds - all_targets)**2))
    direction   = np.mean(
        np.sign(all_preds) == np.sign(all_targets)
    ) * 100

    print(f"\nTest results:")
    print(f"  MAE              : {mae:.1f} ELO points")
    print(f"  RMSE             : {rmse:.1f} ELO points")
    print(f"  Direction acc.   : {direction:.1f}%")
    print(f"  Best val loss    : {best_val_loss:.2f}")

    history["mae"]       = float(mae)
    history["rmse"]      = float(rmse)
    history["direction"] = float(direction)

    # Save
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        torch.save(model.state_dict(),
                   output_dir / "elo_forecaster.pt")
        joblib.dump(scaler,
                    output_dir / "elo_scaler.joblib")
        with open(output_dir / "elo_history.json", "w") as f:
            json.dump(history, f, indent=2)
        with open(output_dir / "elo_config.json", "w") as f:
            json.dump({
                "input_size":  len(feat_cols),
                "hidden_size": hidden_size,
                "seq_len":     seq_len,
                "features":    feat_cols,
            }, f, indent=2)

        print(f"\nModel saved to {output_dir}")

    return model, scaler, history


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

def predict_elo_trajectory(
    game_df: pd.DataFrame,
    model: ELOForecaster,
    scaler: StandardScaler,
    seq_len: int = 20,
    horizon: int = 10,
) -> pd.DataFrame:
    """
    Predict ELO change for the next `horizon` games.

    Parameters
    ----------
    game_df : game-level feature dataframe (most recent games last)
    model   : trained ELOForecaster
    scaler  : fitted StandardScaler
    seq_len : sequence length used during training
    horizon : not used in prediction, kept for documentation

    Returns
    -------
    DataFrame with predictions and confidence intervals
    """
    feat_cols = [c for c in GAME_FEATURES if c in game_df.columns]
    data      = game_df.copy()
    data[feat_cols] = scaler.transform(data[feat_cols])
    data      = data.fillna(0)

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cpu")
    model  = model.to(device)
    model.eval()

    results = []
    for i in range(max(0, len(data) - seq_len), len(data)):
        start = max(0, i - seq_len + 1)
        seq   = data.iloc[start:i+1][feat_cols].values

        if len(seq) < seq_len:
            pad = np.zeros((seq_len - len(seq), len(feat_cols)))
            seq = np.vstack([pad, seq])

        x    = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(x).item()

        results.append({
            "game_index":      i,
            "current_elo":     game_df.iloc[i]["player_elo"]
                               if "player_elo" in game_df.columns else np.nan,
            "predicted_change": round(pred, 1),
            "predicted_elo":   (game_df.iloc[i]["player_elo"] + pred)
                               if "player_elo" in game_df.columns else np.nan,
            "direction":       "up" if pred > 0 else "down",
        })

    return pd.DataFrame(results)


def load_model(model_dir: Path) -> Tuple[ELOForecaster, StandardScaler, dict]:
    """Load a saved ELO forecasting model."""
    model_dir = Path(model_dir)

    with open(model_dir / "elo_config.json") as f:
        config = json.load(f)

    model = ELOForecaster(
        input_size  = config["input_size"],
        hidden_size = config["hidden_size"],
    )
    model.load_state_dict(
        torch.load(model_dir / "elo_forecaster.pt",
                   map_location="cpu")
    )
    model.eval()

    scaler  = joblib.load(model_dir / "elo_scaler.joblib")

    with open(model_dir / "elo_history.json") as f:
        history = json.load(f)

    return model, scaler, history


# ─────────────────────────────────────────────────────────────────────────────
# Option B — Win probability classifier
# ─────────────────────────────────────────────────────────────────────────────

class WinProbabilityClassifier(nn.Module):
    """
    LSTM classifier predicting win probability for the next game.
    Works well with smaller personal datasets (1,000+ games).

    Input  : (batch, seq_len, n_features)
    Output : (batch, 1) — probability of winning next game [0, 1]
    """
    def __init__(
        self,
        input_size:  int,
        hidden_size: int = 64,
        dropout:     float = 0.3,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = 1,
            batch_first = True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1     = nn.Linear(hidden_size, 32)
        self.relu    = nn.ReLU()
        self.fc2     = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        out     = out[:, -1, :]
        out     = self.dropout(out)
        out     = self.relu(self.fc1(out))
        return self.sigmoid(self.fc2(out)).squeeze(-1)


class WinDataset(Dataset):
    """
    Sliding window dataset for win probability training.
    Target: 1 if player won next game, 0 otherwise.
    """
    def __init__(
        self,
        game_df: pd.DataFrame,
        seq_len: int = 15,
        feature_cols: list = GAME_FEATURES,
    ):
        self.X = []
        self.y = []

        # Encode result as win for the player
        def encode_result(row):
            if row["result"] == "1-0" and row["color"] == "white":
                return 1.0
            elif row["result"] == "0-1" and row["color"] == "black":
                return 1.0
            elif row["result"] in ["1/2-1/2", "*"]:
                return 0.5
            return 0.0

        data = game_df.copy()
        data["win"] = data.apply(encode_result, axis=1)

        feat_cols = [c for c in feature_cols if c in data.columns]
        data = data.dropna(subset=feat_cols).reset_index(drop=True)

        for i in range(len(data) - seq_len):
            seq    = data.iloc[i:i+seq_len][feat_cols].values.astype(np.float32)
            target = data.iloc[i+seq_len]["win"]
            self.X.append(seq)
            self.y.append(float(target))

        self.X = np.array(self.X)
        self.y = np.array(self.y, dtype=np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])


def train_win_classifier(
    game_df: pd.DataFrame,
    seq_len:     int   = 15,
    hidden_size: int   = 64,
    epochs:      int   = 50,
    batch_size:  int   = 32,
    lr:          float = 0.001,
    patience:    int   = 10,
    output_dir:  Optional[Path] = None,
) -> Tuple[WinProbabilityClassifier, StandardScaler, dict]:
    """
    Train the win probability classifier on personal game history.

    Much more data-efficient than ELO forecasting.
    Works well with 1,000+ games.
    """
    print("=" * 60)
    print("Win Probability Classifier (LSTM)")
    print("=" * 60)

    scaler    = StandardScaler()
    data      = game_df.copy()
    feat_cols = [c for c in GAME_FEATURES if c in data.columns]
    data[feat_cols] = scaler.fit_transform(data[feat_cols])
    data      = data.fillna(0)

    n        = len(data)
    train_df = data.iloc[:int(n * 0.7)]
    val_df   = data.iloc[int(n * 0.7):int(n * 0.85)]
    test_df  = data.iloc[int(n * 0.85):]

    print(f"\nDataset split:")
    print(f"  Train : {len(train_df)} games")
    print(f"  Val   : {len(val_df)} games")
    print(f"  Test  : {len(test_df)} games")

    train_ds = WinDataset(train_df, seq_len, feat_cols)
    val_ds   = WinDataset(val_df,   seq_len, feat_cols)
    test_ds  = WinDataset(test_df,  seq_len, feat_cols)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cpu")
    print(f"  Device : {device}\n")

    model = WinProbabilityClassifier(
        input_size  = len(feat_cols),
        hidden_size = hidden_size,
    ).to(device)

    optimizer  = torch.optim.Adam(model.parameters(), lr=lr)
    criterion  = nn.BCELoss()
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )

    history        = {"train_loss": [], "val_loss": []}
    best_val_loss  = float("inf")
    best_weights   = None
    patience_count = 0

    print(f"Training for up to {epochs} epochs...")

    for epoch in range(epochs):
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                pred = model(X_batch.to(device))
                loss = criterion(pred, y_batch.to(device))
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss   = np.mean(val_losses)
        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        scheduler.step(val_loss)

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs} | "
                  f"train: {train_loss:.4f} | "
                  f"val: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            best_weights   = {k: v.clone()
                              for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"\n  Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(best_weights)

    # Test evaluation
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            pred = model(X_batch.to(device)).cpu().numpy()
            all_preds.extend(pred)
            all_targets.extend(y_batch.numpy())

    all_preds   = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Binary accuracy (threshold at 0.5)
    binary_preds   = (all_preds > 0.5).astype(float)
    binary_targets = (all_targets > 0.5).astype(float)
    accuracy       = np.mean(binary_preds == binary_targets) * 100

    # Brier score (lower is better, 0.25 = random)
    brier = np.mean((all_preds - all_targets)**2)

    print(f"\nTest results:")
    print(f"  Accuracy    : {accuracy:.1f}%")
    print(f"  Brier score : {brier:.4f} (random=0.25, perfect=0.0)")
    print(f"  Best val loss: {best_val_loss:.4f}")

    history["accuracy"]   = float(accuracy)
    history["brier"]      = float(brier)

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(),
                   output_dir / "win_classifier.pt")
        joblib.dump(scaler,
                    output_dir / "win_scaler.joblib")
        with open(output_dir / "win_history.json", "w") as f:
            json.dump(history, f, indent=2)
        with open(output_dir / "win_config.json", "w") as f:
            json.dump({
                "input_size":  len(feat_cols),
                "hidden_size": hidden_size,
                "seq_len":     seq_len,
                "features":    feat_cols,
            }, f, indent=2)
        print(f"  Saved to {output_dir}")

    return model, scaler, history


# ─────────────────────────────────────────────────────────────────────────────
# Option A — Population LSTM
# ─────────────────────────────────────────────────────────────────────────────

POPULATION_FEATURES = [
    "score",
    "elo_delta",
    "move_count",
    "eco_family",
    "color_encoded",
    "rolling_score_5",
    "rolling_elo_5",
    "elo_trend_10",
]


def build_population_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["color_encoded"] = (data["color"] == "white").astype(float)
    data["rolling_score_5"] = (
        data.groupby("player")["score"]
        .transform(lambda x: x.rolling(5, min_periods=1).mean())
    )
    data["rolling_elo_5"] = (
        data.groupby("player")["player_elo"]
        .transform(lambda x: x.rolling(5, min_periods=1).mean())
    )
    data["elo_trend_10"] = (
        data.groupby("player")["player_elo"]
        .transform(lambda x: x.diff(10).fillna(0))
    ).astype(float)
    return data.fillna(0)



class PopulationDataset(Dataset):
    """
    Multi-player sliding window dataset.
    Each sample: SEQ_LEN games from one player → ELO change in next 10 games.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        seq_len: int = 20,
        feature_cols: list = POPULATION_FEATURES,
        target_col: str = "elo_change_next10",
    ):
        self.X = []
        self.y = []

        feat_cols = [c for c in feature_cols if c in df.columns]

        for player, group in df.groupby("player"):
            group = group.sort_values("game_num").reset_index(drop=True)
            group = group.dropna(subset=[target_col])

            for i in range(len(group) - seq_len):
                seq    = group.iloc[i:i+seq_len][feat_cols].values.astype(
                    np.float32)
                target = group.iloc[i+seq_len][target_col]
                if not np.isnan(target):
                    self.X.append(seq)
                    self.y.append(float(target))

        self.X = np.array(self.X)
        self.y = np.array(self.y, dtype=np.float32)
        print(f"  Population dataset: {len(self.X):,} samples")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])


def train_population_lstm(
    population_parquet: Path,
    seq_len:     int   = 20,
    hidden_size: int   = 128,
    epochs:      int   = 30,
    batch_size:  int   = 256,
    lr:          float = 0.001,
    patience:    int   = 5,
    output_dir:  Optional[Path] = None,
) -> Tuple[ELOForecaster, StandardScaler, dict]:
    """
    Train LSTM on population data from thousands of players.
    This is the Option A model — trained on Elite Database players.

    Parameters
    ----------
    population_parquet : path to parquet file from build_population_data.py
    """
    print("=" * 60)
    print("Phase 3D Option A — Population LSTM")
    print("=" * 60)

    df = pd.read_parquet(population_parquet)
    print(f"\nLoaded {len(df):,} game records")
    print(f"Players: {df['player'].nunique():,}")

    df = build_population_features(df)

    feat_cols = [c for c in POPULATION_FEATURES if c in df.columns]
    print(f"Features: {feat_cols}")

    # Scale features globally
    scaler = StandardScaler()
    df[feat_cols] = scaler.fit_transform(df[feat_cols])
    df = df.fillna(0)

    # Split players into train/val/test (not games — avoid data leakage)
    players    = df["player"].unique()
    np.random.seed(42)
    np.random.shuffle(players)

    n          = len(players)
    train_p    = players[:int(n * 0.7)]
    val_p      = players[int(n * 0.7):int(n * 0.85)]
    test_p     = players[int(n * 0.85):]

    train_df   = df[df["player"].isin(train_p)]
    val_df     = df[df["player"].isin(val_p)]
    test_df    = df[df["player"].isin(test_p)]

    print(f"\nPlayer split:")
    print(f"  Train : {len(train_p):,} players, {len(train_df):,} games")
    print(f"  Val   : {len(val_p):,} players, {len(val_df):,} games")
    print(f"  Test  : {len(test_p):,} players, {len(test_df):,} games")

    print("\nBuilding datasets...")
    train_ds = PopulationDataset(train_df, seq_len, feat_cols)
    val_ds   = PopulationDataset(val_df,   seq_len, feat_cols)
    test_ds  = PopulationDataset(test_df,  seq_len, feat_cols)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cpu")
    print(f"Device: {device}\n")

    model = ELOForecaster(
        input_size  = len(feat_cols),
        hidden_size = hidden_size,
    ).to(device)

    optimizer  = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5
    )
    criterion  = nn.MSELoss()

    history        = {"train_loss": [], "val_loss": []}
    best_val_loss  = float("inf")
    best_weights   = None
    patience_count = 0

    print(f"Training for up to {epochs} epochs...")

    for epoch in range(epochs):
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                pred = model(X_batch.to(device))
                loss = criterion(pred, y_batch.to(device))
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss   = np.mean(val_losses)
        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        scheduler.step(val_loss)

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs} | "
                  f"train: {train_loss:.2f} | val: {val_loss:.2f}")

        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            best_weights   = {k: v.clone()
                              for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"\n  Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(best_weights)

    # Test evaluation
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            pred = model(X_batch.to(device)).cpu().numpy()
            all_preds.extend(pred)
            all_targets.extend(y_batch.numpy())

    all_preds   = np.array(all_preds)
    all_targets = np.array(all_targets)
    mae         = np.mean(np.abs(all_preds - all_targets))
    rmse        = np.sqrt(np.mean((all_preds - all_targets)**2))
    direction   = np.mean(
        np.sign(all_preds) == np.sign(all_targets)
    ) * 100

    print(f"\nPopulation LSTM test results:")
    print(f"  MAE            : {mae:.1f} ELO points")
    print(f"  RMSE           : {rmse:.1f} ELO points")
    print(f"  Direction acc. : {direction:.1f}%")

    history["mae"]       = float(mae)
    history["rmse"]      = float(rmse)
    history["direction"] = float(direction)

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(),
                   output_dir / "population_lstm.pt")
        joblib.dump(scaler,
                    output_dir / "population_scaler.joblib")
        with open(output_dir / "population_history.json", "w") as f:
            json.dump(history, f, indent=2)
        with open(output_dir / "population_config.json", "w") as f:
            json.dump({
                "input_size":  len(feat_cols),
                "hidden_size": hidden_size,
                "seq_len":     seq_len,
                "features":    feat_cols,
            }, f, indent=2)
        print(f"  Saved to {output_dir}")

    return model, scaler, history


# ─────────────────────────────────────────────────────────────────────────────
# Combined model — transfer learning from population to personal
# ─────────────────────────────────────────────────────────────────────────────

def fine_tune_on_personal(
    population_model_dir: Path,
    personal_game_df: pd.DataFrame,
    seq_len:     int   = 15,
    epochs:      int   = 20,
    batch_size:  int   = 16,
    lr:          float = 0.0001,
    patience:    int   = 5,
    output_dir:  Optional[Path] = None,
) -> Tuple[WinProbabilityClassifier, StandardScaler, dict]:
    """
    Fine-tune the population model on personal game data.

    This is the combined Option A + Option B model:
    1. Load population LSTM weights (Option A)
    2. Replace the output layer for win probability prediction
    3. Fine-tune on personal game history (Option B)
    4. Result: population knowledge + personal calibration

    Parameters
    ----------
    population_model_dir : directory containing population_lstm.pt
    personal_game_df     : output of build_game_features() for your games
    """
    print("=" * 60)
    print("Combined Model — Transfer Learning (Option A → B)")
    print("=" * 60)

    # Load population config
    with open(population_model_dir / "population_config.json") as f:
        pop_config = json.load(f)

    print(f"\nPopulation model: {pop_config['input_size']} features, "
          f"hidden={pop_config['hidden_size']}")

    # Build personal features using personal feature set
    personal_scaler = StandardScaler()
    data            = personal_game_df.copy()
    feat_cols       = [c for c in GAME_FEATURES if c in data.columns]
    data[feat_cols] = personal_scaler.fit_transform(data[feat_cols])
    data            = data.fillna(0)

    n        = len(data)
    train_df = data.iloc[:int(n * 0.7)]
    val_df   = data.iloc[int(n * 0.7):int(n * 0.85)]
    test_df  = data.iloc[int(n * 0.85):]

    train_ds = WinDataset(train_df, seq_len, feat_cols)
    val_ds   = WinDataset(val_df,   seq_len, feat_cols)
    test_ds  = WinDataset(test_df,  seq_len, feat_cols)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cpu")

    # Build classifier with same hidden size as population model
    model = WinProbabilityClassifier(
        input_size  = len(feat_cols),
        hidden_size = pop_config["hidden_size"],
    ).to(device)

    # Transfer LSTM weights from population model
    # (only if input sizes match — otherwise train from scratch)
    if pop_config["input_size"] == len(feat_cols):
        pop_state = torch.load(
            population_model_dir / "population_lstm.pt",
            map_location=device
        )
        # Transfer only LSTM weights, not output layers
        model_state = model.state_dict()
        transferred = {k: v for k, v in pop_state.items()
                       if k.startswith("lstm")}
        model_state.update(transferred)
        model.load_state_dict(model_state)
        print("  Transferred LSTM weights from population model")
    else:
        print(f"  Feature size mismatch "
              f"(pop={pop_config['input_size']}, "
              f"personal={len(feat_cols)}) — training from scratch")

    # Fine-tune with lower learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    history        = {"train_loss": [], "val_loss": []}
    best_val_loss  = float("inf")
    best_weights   = None
    patience_count = 0

    print(f"\nFine-tuning for up to {epochs} epochs "
          f"(lr={lr}, patience={patience})...")
    print(f"  Train: {len(train_ds)} samples | "
          f"Val: {len(val_ds)} | Test: {len(test_ds)}")

    for epoch in range(epochs):
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                pred = model(X_batch.to(device))
                loss = criterion(pred, y_batch.to(device))
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss   = np.mean(val_losses)
        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs} | "
                  f"train: {train_loss:.4f} | val: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            best_weights   = {k: v.clone()
                              for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"\n  Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(best_weights)

    # Test
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            pred = model(X_batch.to(device)).cpu().numpy()
            all_preds.extend(pred)
            all_targets.extend(y_batch.numpy())

    all_preds      = np.array(all_preds)
    all_targets    = np.array(all_targets)
    binary_preds   = (all_preds > 0.5).astype(float)
    binary_targets = (all_targets > 0.5).astype(float)
    accuracy       = np.mean(binary_preds == binary_targets) * 100
    brier          = np.mean((all_preds - all_targets)**2)

    print(f"\nCombined model test results:")
    print(f"  Accuracy    : {accuracy:.1f}%")
    print(f"  Brier score : {brier:.4f} (random=0.25)")
    print(f"  Best val    : {best_val_loss:.4f}")

    history["accuracy"] = float(accuracy)
    history["brier"]    = float(brier)

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(),
                   output_dir / "combined_model.pt")
        joblib.dump(personal_scaler,
                    output_dir / "combined_scaler.joblib")
        with open(output_dir / "combined_history.json", "w") as f:
            json.dump(history, f, indent=2)
        with open(output_dir / "combined_config.json", "w") as f:
            json.dump({
                "input_size":  len(feat_cols),
                "hidden_size": pop_config["hidden_size"],
                "seq_len":     seq_len,
                "features":    feat_cols,
            }, f, indent=2)
        print(f"  Saved to {output_dir}")

    return model, personal_scaler, history
