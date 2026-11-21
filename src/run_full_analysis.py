from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)
sns.set_theme(style="whitegrid", context="notebook")

def detect_single_changepoint(series: np.ndarray) -> tuple[int, float]:
    """Finds the split index that maximizes the two-sample Mann-Whitney U / mean shift."""
    n = len(series)
    best_idx, best_score = 0, -1.0
    for k in range(4, n - 4):
        left, right = series[:k], series[k:]
        # Sum of squared deviations reduction (classic single change-point SSE criterion)
        sse_total = np.sum((series - series.mean()) ** 2)
        sse_split = np.sum((left - left.mean()) ** 2) + np.sum((right - right.mean()) ** 2)
        reduction = sse_total - sse_split
        if reduction > best_score:
            best_score = reduction
            best_idx = k
    _, p_val = stats.mannwhitneyu(series[:best_idx], series[best_idx:], alternative="two-sided")
    return best_idx, p_val

# ---------------------------------------------------------
# 1. LOAD ALL THREE DATASETS
# ---------------------------------------------------------
yt_table = pd.read_csv("data/raw/youtube/Table data.csv")
yt_videos = yt_table[yt_table["Content"] != "Total"].copy()
yt_videos["publish_dt"] = pd.to_datetime(yt_videos["Video publish time"], errors="coerce")
yt_videos = yt_videos.dropna(subset=["publish_dt"]).sort_values("publish_dt").reset_index(drop=True)

yt_daily = pd.read_csv("data/raw/youtube/Totals.csv")
yt_daily["Date"] = pd.to_datetime(yt_daily["Date"])

fn = pd.read_csv("data/processed/fortnite_seasons_clean.csv")
clips = pd.read_csv("data/processed/clips_metadata.csv")
clips["timestamp"] = pd.to_datetime(clips["timestamp"], format="ISO8601")
playable_clips = clips[clips["is_playable"] == True].copy()

# ---------------------------------------------------------
# 2. HYPOTHESIS 1: CREATOR BURDEN (FORTNITE PERFORMANCE BY ERA)
# ---------------------------------------------------------
prime_win = fn[fn["era"] == "Prime_Creator_Era"]["win_pct"]
decay_win = fn[fn["era"] == "Channel_Decay_Era"]["win_pct"]
post_win = fn[fn["era"] == "Post_Creator_Era"]["win_pct"]

u_prime_post, p_prime_post = stats.mannwhitneyu(prime_win, post_win, alternative="two-sided")
kw_stat, kw_p = stats.kruskal(prime_win, decay_win, post_win)
r_vol_win, p_vol_win = stats.spearmanr(fn["matches"], fn["win_pct"])

print("=== HYPOTHESIS 1: CREATOR BURDEN ===")
print(f"Kruskal-Wallis across 3 Eras (Win %): H={kw_stat:.3f}, p={kw_p:.4f}")
print(f"Mann-Whitney U (Prime vs Post-Creator Win %): U={u_prime_post:.1f}, p={p_prime_post:.4f}")
print(f"Spearman Corr (Matches Played vs Win %): r={r_vol_win:.3f}, p={p_vol_win:.4f}\n")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
palette = {"Prime_Creator_Era": "#2563eb", "Channel_Decay_Era": "#f59e0b", "Post_Creator_Era": "#10b981"}
colors = fn["era"].map(palette)

ax1.bar(fn["season"], fn["matches"], color=colors, alpha=0.85)
ax1.set_ylabel("Matches Played", fontweight="bold")
ax1.set_title("32-Season Gameplay Volume & Win Efficiency Across Creator Lifecycle Regimes", fontsize=13, fontweight="bold")

ax2.plot(fn["season"], fn["win_pct"], color="#1e293b", marker="o", linewidth=2, label="Seasonal Win %")
ax2.scatter(fn["season"], fn["win_pct"], color=colors, s=70, zorder=5)
ax2.axhline(prime_win.mean(), color="#2563eb", linestyle="--", alpha=0.7, label=f"Prime Mean ({prime_win.mean():.1f}%)")
ax2.axhline(post_win.mean(), color="#10b981", linestyle="--", alpha=0.7, label=f"Post-Creator Mean ({post_win.mean():.1f}%)")
ax2.set_ylabel("Win Rate (%)", fontweight="bold")
ax2.legend(loc="upper left")
plt.xticks(rotation=60, ha="right")
fig.tight_layout()
fig.savefig(FIG_DIR / "01_creator_burden_seasons.png", dpi=180)
plt.close(fig)

# ---------------------------------------------------------
# 3. HYPOTHESIS 2: ALGORITHMIC BURNOUT & CHANGE-POINT DETECTION
# ---------------------------------------------------------
monthly_views = yt_daily.set_index("Date")["Views"].resample("ME").sum()
monthly_uploads = yt_videos.set_index("publish_dt")["Content"].resample("ME").count().reindex(monthly_views.index, fill_value=0)
yt_monthly = pd.DataFrame({"views": monthly_views, "uploads": monthly_uploads})

r_same, p_same = stats.spearmanr(yt_monthly["uploads"], yt_monthly["views"])
r_lag1, p_lag1 = stats.spearmanr(yt_monthly["uploads"].shift(1).dropna(), yt_monthly["views"].iloc[1:])

bkp_idx, bkp_p = detect_single_changepoint(yt_monthly["uploads"].values)
bkp_date = yt_monthly.index[bkp_idx]

print("=== HYPOTHESIS 2: ALGORITHMIC BURNOUT ===")
print(f"Monthly Uploads vs Views (Same Month): r={r_same:.3f}, p={p_same:.4f}")
print(f"Monthly Uploads (t-1) vs Views (t): r={r_lag1:.3f}, p={p_lag1:.4f}")
print(f"Detected Upload Cadence Structural Break: {bkp_date.strftime('%Y-%m')} (Mann-Whitney p={bkp_p:.4f})\n")

fig, ax1 = plt.subplots(figsize=(12, 5))
ax1.fill_between(yt_monthly.index, yt_monthly["views"], color="#3b82f6", alpha=0.25)
ax1.plot(yt_monthly.index, yt_monthly["views"], color="#2563eb", linewidth=2, label="Monthly Channel Views")
ax1.set_ylabel("Monthly Views", color="#1d4ed8", fontweight="bold")

ax2 = ax1.twinx()
ax2.bar(yt_monthly.index, yt_monthly["uploads"], width=20, color="#ef4444", alpha=0.65, label="Videos Uploaded")
ax2.set_ylabel("Videos Uploaded / Month", color="#b91c1c", fontweight="bold")
ax1.axvline(bkp_date, color="#0f172a", linestyle="--", linewidth=1.8, label=f"Detected Cadence Break ({bkp_date.strftime('%Y-%m')})")
ax1.legend(loc="upper left")
ax1.set_title("YouTube Channel Trajectory: Monthly Uploads vs. Views (Nov 2019 – Nov 2023)", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(FIG_DIR / "02_youtube_burnout_timeline.png", dpi=180)
plt.close(fig)

# ---------------------------------------------------------
# 4. HYPOTHESIS 3: FORMAT & CLIP CIRCADIAN SHIFT
# ---------------------------------------------------------
def classify_video(title: str) -> str:
    t = str(title).lower()
    if "highlight" in t or "montage" in t or "synced" in t:
        return "Edited Highlights"
    if "revert" in t or "day " in t or "live" in t:
        return "Long-Form / League"
    return "Casual / Other"

yt_videos["format"] = yt_videos["Video title"].apply(classify_video)
print("=== HYPOTHESIS 3: FORMAT & CLIP SESSION PATTERNS ===")
print(yt_videos.groupby("format")[["Views", "Watch time (hours)", "Thumbnail click-through rate (%)"]].agg(["count", "median", "mean"]).round(2))

playable_clips["post_yt"] = playable_clips["timestamp"] >= pd.Timestamp("2023-11-02")
hourly = playable_clips.groupby(["post_yt", "hour"]).size().reset_index(name="count")
hourly["pct"] = hourly.groupby("post_yt")["count"].transform(lambda x: x / x.sum() * 100)
hourly["Period"] = hourly["post_yt"].map({False: "Pre-Nov 2023 (Creator Era)", True: "Post-Nov 2023 (Recreational)"})

fig, ax = plt.subplots(figsize=(11, 4.5))
sns.lineplot(data=hourly, x="hour", y="pct", hue="Period", marker="o", linewidth=2.2, ax=ax)
ax.set_xticks(range(0, 24))
ax.set_xlabel("Hour of Day (0–23)", fontweight="bold")
ax.set_ylabel("Share of Saved Clips (%)", fontweight="bold")
ax.set_title("Gaming Session Circadian Shift: Hourly Clip Distribution Before vs. After Quitting YouTube", fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(FIG_DIR / "03_clip_circadian_shift.png", dpi=180)
plt.close(fig)

print("\nAll figures saved successfully to figures/")