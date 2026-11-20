# Why Do Gaming Creators Quit? A Longitudinal Analysis of Gameplay Performance, Highlight Generation, and Audience Feedback (2018–2026)
 
**Author:** Abdul Rehman Muhammad  - 36077

---

## Overview & Motivation
Many independent gaming creators start uploading consistently, hit a creative peak, and eventually abandon their channels—even while continuing to play the game for years afterward. 

Initial exploratory analysis of my YouTube Studio archive (`130` videos, `16,761` total views) reveals that my channel did not die abruptly in late 2023; rather, it peaked between **late 2020 and mid-2021** (`2020 Q4` peak output of 36 videos; `2021 Q1` peak efficiency of 304.1 views/video) before entering a prolonged upload and click-through decay through November 2023.

This project investigates the empirical drivers behind that decline by merging three personal longitudinal datasets across three distinct behavioral regimes:
1. **Prime Creator Era (`Jul 2020 – Jun 2021`):** Peak upload cadence, watch time, and subscriber growth.
2. **Channel Decay Era (`Jul 2021 – Nov 1, 2023`):** Sharp drop in upload frequency (`-75%+`) and median CTR (`~9.0%` → `~6.4%`).
3. **Post-Creator Control Era (`Nov 2, 2023 – Nov 2026`):** Pure recreational/competitive play after abandoning YouTube uploads.

---

## Research Questions & Hypotheses

1. **The "Creator Burden" Hypothesis (Prime Era vs. Decay Era vs. Post-Creator Era)**
   * *Question:* Did playing for content during my YouTube Prime (2020–2021) help or hurt actual in-game performance and match volume compared to my Decay (2021–2023) and Post-Creator (2023–2026) eras?
   * *Hypothesis:* Competitive performance metrics (seasonal K/D, win rate, kills/match, and total matches played) differ significantly across the three channel lifecycle regimes.

2. **The "Algorithmic Burnout" Hypothesis (Nov 2019 – Nov 2023)**
   * *Question:* Did upload frequency drop after viewer engagement (CTR and views) declined, or did irregular upload gaps precede the collapse in channel impressions?
   * *Hypothesis:* Lagged cross-correlation and change-point detection (`ruptures`) on daily YouTube time-series data will identify a structural break in upload cadence immediately following the 2020 Q4 – 2021 Q1 peak.

3. **The "Highlight Intensity & Play-Pattern" Hypothesis (Clip Telemetry + HUD OCR)**
   * *Question:* Do gameplay session patterns (time of day, clip duration distributions, and OCR-extracted HUD elimination counts) reflect a shift from content-focused highlight hunting to casual late-night play?
   * *Hypothesis:* Clip duration, session clustering, and elimination density vary systematically by time of day and competitive season.

---

## Data Sources & Bias Controls

All data in this repository is strictly personal, spanning three overlapping timelines:

| Dataset | Time Span | Source & Format | Key Variables Extracted |
| :--- | :--- | :--- | :--- |
| **YouTube Channel Analytics** | Nov 1, 2019 – Nov 1, 2023 | YouTube Studio Export (`Table data.csv`, `Chart data.csv`, `Totals.csv`) | Publish timestamps, views, watch time, average view duration, impressions, CTR, daily channel trends |
| **Raw Gameplay Clips** | 2018 – Nov 2026 | Local video archive metadata & sampled HUD frames (`clips_metadata.csv`, `2,769` unique clips) | Clip creation timestamps, session clustering, clip duration, resolution/FPS, playability flags, OCR HUD kills |
| **Fortnite Tracker Stats** | 2020 – Nov 2026 | Personal profile seasonal & match telemetry (`fortnite_stats.csv`) | Seasonal matches played, K/D, win rate, kills per match, TRN rating across Chapters |

**Methodological Control for Archival/Recency Bias:**  
Initial inspection of the `2,769` local clips showed a large volume surge starting in 2023 (`2,295` clips in 2023 vs. `172` across 2020–2022) because active clip-saving habits were not adopted consistently until 2023. Therefore, raw clip counts are **not** used as a proxy for historical playtime; instead, **Fortnite Tracker `Matches Played`** serves as the unbiased denominator for gameplay volume across 2020–2026, while the dense 2023–2026 clip archive is analyzed for session timing, duration distributions, and computer-vision HUD telemetry.

*(Note: Raw `.mp4`/`.mkv` video files are excluded via `.gitignore`; only extracted tabular metadata and analysis scripts are committed.)*

---

## Repository Structure

```text
DSA210-TermProject/
├── data/
│   ├── raw/
│   │   ├── youtube/            # Raw YouTube Studio CSV exports (2019–2023)
│   │   └── fortnite_tracker/   # Extracted Fortnite Tracker stats (2020–2026)
│   └── processed/
│       └── clips_metadata.csv  # Deduplicated metadata for 2,769 local clips (2018–2026)
├── notebooks/
│   └── 01_youtube_eda.ipynb    # Exploratory analysis of YouTube lifecycle & regimes
├── src/
│   └── extract_clip_metadata.py # Multi-directory video metadata & playability scanner
├── .gitignore
└── README.md
