import json
import re
from pathlib import Path
import pandas as pd

RAW_JSON = Path("data/raw/fortnite_tracker/fortnite_seasons_raw.json")
OUTPUT_CSV = Path("data/processed/fortnite_seasons_clean.csv")

PRIME_SEASONS = {
    "Ch.2 Season 2", "Ch.2 Season 3", "Ch.2 Season 4", "Ch.2 Season 5",
    "Ch.2 Season 6", "Ch.2 Season 7", "Ch.2 Season 8"
}
DECAY_SEASONS = {
    "Ch.3 Season 1", "Ch.3 Season 2", "Ch.3 Season 3", "Ch.3 Season 4",
    "Ch.4 Season 1", "Ch.4 Season 2", "Ch.4 Season 3", "Ch.4 Season 4"
}   

def parse_hours(time_str: str) -> float:
    m = re.match(r"(?:(\d+)h)?\s*(?:(\d+)m)?", time_str.strip())
    if not m:
        return 0.0
    hours = int(m.group(1) or 0)
    minutes = int(m.group(2) or 0)
    return round(hours + minutes / 60.0, 2)

def assign_era(season: str) -> str:
    if season in PRIME_SEASONS:
        return "Prime_Creator_Era"
    if season in DECAY_SEASONS:
        return "Channel_Decay_Era"
    return "Post_Creator_Era"

def main():
    with open(RAW_JSON, "r", encoding="utf-8") as f:
        raw = json.load(f)

    table = raw["tables"][0]
    df = pd.DataFrame(table["rows"], columns=table["headers"])

    # Sort chronologically from Ch.2 Season 2 -> Ch.7 Season 5
    df = df.iloc[::-1].reset_index(drop=True)
    df["season_index"] = range(1, len(df) + 1)

    df["best_tracker_rating"] = df["Best Tracker Rating"].str.replace(",", "").astype(int)
    df["matches"] = df["Matches"].str.replace(",", "").astype(int)
    df["win_pct"] = df["Win %"].str.replace("%", "").astype(float)
    df["wins"] = df["Wins"].str.replace(",", "").astype(int)
    df["top_3_5_10"] = df["Top 3/5/10"].str.replace(",", "").astype(int)
    df["top_6_12_25"] = df["Top 6/12/25"].str.replace(",", "").astype(int)
    df["hours_played"] = df["Time Played"].apply(parse_hours)

    df["top_tier_rate_pct"] = round(df["top_3_5_10"] / df["matches"] * 100, 2)
    df["top_mid_rate_pct"] = round(df["top_6_12_25"] / df["matches"] * 100, 2)
    df["avg_match_min"] = round((df["hours_played"] * 60) / df["matches"], 2)
    df["era"] = df["Season"].apply(assign_era)

    clean_cols = [
        "season_index", "Season", "era", "best_tracker_rating", "matches",
        "wins", "win_pct", "top_3_5_10", "top_tier_rate_pct",
        "top_6_12_25", "top_mid_rate_pct", "hours_played", "avg_match_min"
    ]
    df_clean = df[clean_cols].rename(columns={"Season": "season"})

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {len(df_clean)} seasons ({df_clean['matches'].sum()} matches) to {OUTPUT_CSV}")
    print(df_clean.groupby("era")[["matches", "wins", "win_pct", "hours_played"]].mean().round(2))

if __name__ == "__main__":
    main()