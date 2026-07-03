"""
End-to-end data preparation for the World Cup win predictor.

Pipeline: raw results -> clean -> filter to World Cup -> reshape to
one row per team per match -> rolling form/goals features (leakage-safe,
shifted by 1) -> merge FIFA ranking history (leakage-safe, merge_asof
backward) -> rank_diff feature -> final (X, y) matrix.
"""

import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Tuple

FEATURE_COLS: list[str] = ["form_5", "goals_avg_5", "conceded_avg_5", "is_host", "rank_diff"]

# FIFA ranking dataset uses different country names in places. Maps the
# international-results dataset's team names onto the ranking dataset's names.
NAME_MAPPING: dict[str, str] = {
    "South Korea": "Korea Republic",
    "North Korea": "Korea DPR",
    "United States": "USA",
    "Iran": "IR Iran",
    "China": "China PR",
    "Ivory Coast": "Côte d'Ivoire",
    "Czech Republic": "Czechia",
    "Curaçao": "Curacao",
    "Cape Verde": "Cabo Verde",
    "DR Congo": "Congo DR",
}



def load_results(path: str) -> DataFrame:
    """Load and clean the international results CSV.
    
    Args:
        path: Path to the results CSV file.
        
    Returns:
        Cleaned DataFrame with datetime and no missing scores.
        
    Raises:
        FileNotFoundError: If the file does not exist.
    """
    try:
        df: DataFrame = pd.read_csv(path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Results file not found: {path}")
    except Exception as e:
        raise ValueError(f"Failed to read results CSV: {e}")
    
    if "date" not in df.columns or "home_score" not in df.columns:
        raise ValueError("Results CSV missing required columns")
    
    df["date"] = pd.to_datetime(df["date"])
    df: DataFrame = df.dropna(subset=["home_score", "away_score"])
    return df


def filter_world_cup(df: DataFrame) -> DataFrame:
    """Filter to only World Cup matches."""
    return df[df["tournament"] == "FIFA World Cup"].copy()


def reshape_team_matches(wc_df: DataFrame) -> DataFrame:
    """One row per match becomes two rows: one per team's perspective."""
    home_df = wc_df.copy()
    home_df["team"] = home_df["home_team"]
    home_df["opponent"] = home_df["away_team"]
    home_df["goals_for"] = home_df["home_score"]
    home_df["goals_against"] = home_df["away_score"]
    home_df["is_home_label"] = ~home_df["neutral"]

    away_df = wc_df.copy()
    away_df["team"] = away_df["away_team"]
    away_df["opponent"] = away_df["home_team"]
    away_df["goals_for"] = away_df["away_score"]
    away_df["goals_against"] = away_df["home_score"]
    away_df["is_home_label"] = False

    team_matches: DataFrame = pd.concat([home_df, away_df], ignore_index=True)
    team_matches["win"] = (team_matches["goals_for"] > team_matches["goals_against"]).astype(int)
    team_matches["is_host"] = team_matches["is_home_label"].astype(int)
    return team_matches.sort_values(["team", "date"]).reset_index(drop=True)


def add_rolling_features(team_matches: DataFrame, window: int = 5) -> DataFrame:
    """Rolling form/goals features, shifted by 1 to prevent leakage.
    
    Args:
        team_matches: DataFrame with team-level match data.
        window: Rolling window size for computing averages.
        
    Returns:
        DataFrame with rolling features (form_5, goals_avg_5, conceded_avg_5).
    """
    tm = team_matches.copy()
    tm["form_5"] = (
        tm.groupby("team")["win"]
        .transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).mean())
    )
    tm["goals_avg_5"] = (
        tm.groupby("team")["goals_for"]
        .transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).mean())
    )
    tm["conceded_avg_5"] = (
        tm.groupby("team")["goals_against"]
        .transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).mean())
    )
    return tm.dropna(subset=["form_5", "goals_avg_5", "conceded_avg_5"])


def load_rankings(path: str) -> DataFrame:
    """Load FIFA rankings CSV and sort chronologically by country and date."""
    rankings: DataFrame = pd.read_csv(path)
    rankings["rank_date"] = pd.to_datetime(rankings["rank_date"])
    return rankings.sort_values(["country_full", "rank_date"]).reset_index(drop=True)


def merge_rankings(team_matches: DataFrame, rankings: DataFrame) -> DataFrame:
    """
    Leakage-safe merge: attaches each team's most recent FIFA rank as of
    the match date (never a rank published after the match), for both
    the team itself and its opponent, then derives rank_diff.
    
    Args:
        team_matches: DataFrame with team-level match data.
        rankings: DataFrame with FIFA ranking history.
        
    Returns:
        DataFrame with rank and opponent_rank columns merged in.
    """
    tm = team_matches.sort_values("date").reset_index(drop=True)
    rk = rankings.sort_values("rank_date").reset_index(drop=True)

    tm["team_mapped"] = tm["team"].replace(NAME_MAPPING)
    tm: DataFrame = pd.merge_asof(
        tm, rk[["rank_date", "country_full", "rank", "total_points"]],
        left_on="date", right_on="rank_date",
        left_by="team_mapped", right_by="country_full",
        direction="backward",
    )

    tm["opponent_mapped"] = tm["opponent"].replace(NAME_MAPPING)
    tm: DataFrame = pd.merge_asof(
        tm.sort_values("date"),
        rk[["rank_date", "country_full", "rank"]].rename(
            columns={"rank": "opponent_rank", "country_full": "country_full_opp"}
        ),
        left_on="date", right_on="rank_date",
        left_by="opponent_mapped", right_by="country_full_opp",
        direction="backward",
    )

    tm: DataFrame = tm.dropna(subset=["rank", "opponent_rank"]).copy()
    tm["rank_diff"] = tm["opponent_rank"] - tm["rank"]
    return tm


def build_feature_matrix(team_matches_final: DataFrame, 
                         feature_cols: list[str] = FEATURE_COLS) -> Tuple[np.ndarray, np.ndarray, DataFrame]:
    """Build feature matrix and target vector from processed team matches.
    
    Args:
        team_matches_final: Processed team-level match data with features.
        feature_cols: List of feature column names to extract.
        
    Returns:
        Tuple of (X, y, model_df) where X is feature matrix, y is target vector.
    """
    model_df = team_matches_final.dropna(subset=feature_cols + ["win"]).copy()
    X = model_df[feature_cols].to_numpy(dtype=float)
    y = model_df["win"].to_numpy(dtype=float)
    return X, y, model_df


def build_dataset(results_path: str, rankings_path: str, 
                  feature_cols: list[str] = FEATURE_COLS) -> Tuple[np.ndarray, np.ndarray, DataFrame]:
    """Run the full pipeline end to end and return (X, y, model_df).
    
    This function orchestrates the entire data preparation pipeline:
    1. Load and clean results data
    2. Filter to World Cup matches
    3. Reshape to team-level perspective
    4. Compute rolling form features (with leakage prevention)
    5. Merge FIFA ranking data (backward merge for leakage safety)
    6. Build feature matrix with rank differential
    
    Args:
        results_path: Path to international results CSV.
        rankings_path: Path to FIFA rankings CSV.
        feature_cols: List of feature columns to extract.
        
    Returns:
        Tuple of (X, y, model_df) where:
        - X is feature matrix (n_samples, n_features)
        - y is target vector (n_samples,)
        - model_df is full DataFrame with all features and metadata
    """
    df: DataFrame = load_results(results_path)
    wc_df: DataFrame = filter_world_cup(df)
    team_matches: NoReturn = reshape_team_matches(wc_df)
    team_matches: NoReturn = add_rolling_features(team_matches)
    rankings: DataFrame = load_rankings(rankings_path)
    team_matches_final: DataFrame = merge_rankings(team_matches, rankings)
    return build_feature_matrix(team_matches_final, feature_cols)
