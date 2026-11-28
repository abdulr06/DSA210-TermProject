from pathlib import Path
import numpy as np
import pandas as pd

OUT_DIR = Path("data/processed")


def classify_video(title: str) -> str:
    t = str(title).lower()
    if "highlight" in t or "montage" in t or "synced" in t:
        return "Edited_Highlights"
    if "revert" in t or "day " in t or "live" in t:
        return "Long_Form_League"
    return "Casual_Other"


def build_clip_session_dataset() -> pd.DataFrame:
    clips = pd.read_csv("data/processed/clips_metadata.csv")
    df = clips[clips["is_playable"] == True].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601")
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Normalize file size column regardless of exact name in clips_metadata.csv
    if "file_size_mb" not in df.columns:
        if "size_mb" in df.columns:
            df["file_size_mb"] = df["size_mb"]
        elif "size_bytes" in df.columns:
            df["file_size_mb"] = df["size_bytes"] / (1024 ** 2)

    # Gap between consecutive clips in minutes; >45 min starts a new gaming session
    df["minutes_since_prev_clip"] = df["timestamp"].diff().dt.total_seconds().div(60.0).fillna(999.0)
    df["session_id"] = (df["minutes_since_prev_clip"] > 45.0).cumsum()
    df["session_clip_count"] = df.groupby("session_id")["filename"].transform("count")

    # Cyclical hour encoding (keeps 23:00 and 00:00 close in Euclidean space for k-NN)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["day_of_week"] = df["timestamp"].dt.day_of_week
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # Cap extreme inter-session gaps so distance-based models aren't skewed by multi-day breaks
    df["capped_gap_min"] = df["minutes_since_prev_clip"].clip(upper=180.0)

    # Target: Post-YouTube Recreational Era (after Nov 2, 2023)
    df["is_post_creator"] = (df["timestamp"] >= pd.Timestamp("2023-11-02")).astype(int)

    candidate_cols = [
        "filename", "timestamp", "hour", "hour_sin", "hour_cos",
        "day_of_week", "is_weekend", "duration_sec", "file_size_mb",
        "session_clip_count", "capped_gap_min", "is_post_creator"
    ]
    cols = [c for c in candidate_cols if c in df.columns]
    return df[cols]


def build_youtube_ml_dataset() -> pd.DataFrame:
    yt_table = pd.read_csv("data/raw/youtube/Table data.csv")
    df = yt_table[yt_table["Content"] != "Total"].copy()
    df["publish_dt"] = pd.to_datetime(df["Video publish time"], errors="coerce")
    df = df.dropna(subset=["publish_dt"]).sort_values("publish_dt").reset_index(drop=True)

    df["days_since_prev_upload"] = df["publish_dt"].diff().dt.days.fillna(14.0).clip(upper=90.0)
    df["upload_hour"] = df["publish_dt"].dt.hour
    df["is_weekend"] = (df["publish_dt"].dt.day_of_week >= 5).astype(int)

    df["rolling_30d_uploads"] = [
        ((df["publish_dt"] < dt) & (df["publish_dt"] >= dt - pd.Timedelta(days=30))).sum()
        for dt in df["publish_dt"]
    ]

    df["title_length"] = df["Video title"].astype(str).str.len()
    df["has_exclamation_or_caps"] = (
        df["Video title"].astype(str).str.contains(r"!|[A-Z]{4,}", regex=True).astype(int)
    )
    df["format"] = df["Video title"].apply(classify_video)
    df = pd.get_dummies(df, columns=["format"], dtype=int)

    # Binary target: Breakout video (>= 100 views)
    df["is_breakout_video"] = (df["Views"] >= 100).astype(int)

    cols = [
        "Video title", "Views", "Thumbnail click-through rate (%)",
        "days_since_prev_upload", "rolling_30d_uploads", "upload_hour",
        "is_weekend", "title_length", "has_exclamation_or_caps",
        "format_Casual_Other", "format_Edited_Highlights", "format_Long_Form_League",
        "is_breakout_video"
    ]
    return df[cols]


def build_monthly_counterfactual_dataset() -> pd.DataFrame:
    yt_daily = pd.read_csv("data/raw/youtube/Totals.csv")
    yt_daily["Date"] = pd.to_datetime(yt_daily["Date"])
    monthly_views = yt_daily.set_index("Date")["Views"].resample("ME").sum()

    yt_table = pd.read_csv("data/raw/youtube/Table data.csv")
    vids = yt_table[yt_table["Content"] != "Total"].copy()
    vids["publish_dt"] = pd.to_datetime(vids["Video publish time"], errors="coerce")
    vids = vids.dropna(subset=["publish_dt"])
    vids["is_highlight"] = vids["Video title"].apply(
        lambda t: int(classify_video(t) == "Edited_Highlights")
    )

    monthly_uploads = vids.set_index("publish_dt")["Content"].resample("ME").count().reindex(monthly_views.index, fill_value=0)
    monthly_highlights = vids.set_index("publish_dt")["is_highlight"].resample("ME").sum().reindex(monthly_views.index, fill_value=0)

    df = pd.DataFrame({
        "month": monthly_views.index.strftime("%Y-%m"),
        "views": monthly_views.values,
        "uploads": monthly_uploads.values,
        "highlights": monthly_highlights.values,
    })
    df["highlight_share"] = np.where(df["uploads"] > 0, df["highlights"] / df["uploads"], 0.0)
    df["lag1_views"] = df["views"].shift(1).fillna(df["views"].iloc[0])
    df["lag1_uploads"] = df["uploads"].shift(1).fillna(df["uploads"].iloc[0])
    df["is_pre_break"] = (df["month"] <= "2020-12").astype(int)
    return df


if __name__ == "__main__":
    clip_df = build_clip_session_dataset()
    yt_df = build_youtube_ml_dataset()
    cf_df = build_monthly_counterfactual_dataset()

    clip_df.to_csv(OUT_DIR / "ml_clip_session_features.csv", index=False)
    yt_df.to_csv(OUT_DIR / "ml_youtube_features.csv", index=False)
    cf_df.to_csv(OUT_DIR / "ml_monthly_burnout.csv", index=False)

    print(f"1. Clip Session ML table:   {clip_df.shape} | Post-creator share: {clip_df['is_post_creator'].mean():.2%}")
    print(f"2. YouTube Video ML table:  {yt_df.shape} | Breakout share:     {yt_df['is_breakout_video'].mean():.2%}")
    print(f"3. Monthly Burnout table:   {cf_df.shape} | Pre-break months:   {cf_df['is_pre_break'].sum()}")