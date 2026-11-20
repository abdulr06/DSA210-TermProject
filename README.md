# Why Do Gaming Creators Quit? A Longitudinal Analysis of Gameplay Performance, Highlight Generation, and Audience Feedback (2018–2026)

## **Author:** Abdul Rahman Muhammad Ehsan Ul Haque

## Overview & Motivation

Many independent gaming creators start uploading consistently, plateau, and eventually abandon their channels—even while continuing to play the game for years afterward.

This project investigates the empirical drivers behind quitting content creation using an 8-year personal multimodal dataset spanning **pre-channel baseline play (2018–2019)**, an **active YouTube publishing era (Nov 2019 – Nov 2023)**, and a **post-channel control period (Nov 2023 – Nov 2026)**. By merging competitive match telemetry, local gameplay clip metadata/computer vision features, and YouTube Studio analytics, this study tests whether channel abandonment was triggered by declining in-game performance, algorithmic stagnation, or the "creator burden" of filming and editing.

---

## Research Questions & Hypotheses

1. **The "Creator Burden" Hypothesis (2020–2023 vs. 2024–2026)**
   - _Question:_ Did playing for content hurt or improve actual in-game performance compared to playing purely recreationally after quitting YouTube?
   - _Hypothesis:_ Competitive performance metrics (K/D ratio, win rate, kills/match) and raw clip-saving frequency differ significantly between the active creator era and the post-creator era.

2. **The "Algorithmic Burnout" Hypothesis (Nov 2019 – Nov 2023)**
   - _Question:_ Did upload cessation follow a collapse in viewer engagement, or did irregular upload gaps precede the decay in views and impressions?
   - _Hypothesis:_ Lagged cross-correlation and change-point detection will reveal a structural break in upload cadence prior to the channel's final inactivity in November 2023.

3. **The "Highlight Quality vs. Retention" Hypothesis (2019–2023 Overlap)**
   - _Question:_ Do higher-intensity gameplay moments (measured via clip saving bursts, clip duration, and HUD elimination counts) translate into higher YouTube audience retention and click-through rates?
   - _Hypothesis:_ Videos backed by denser clip bursts and higher-elimination gameplay achieved higher average view duration than routine uploads.

---

## Data Sources

All data in this repository is strictly personal, spanning three overlapping timelines:

| Dataset                       | Time Span                 | Source & Format                                                          | Key Variables Extracted                                                                                           |
| :---------------------------- | :------------------------ | :----------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------- |
| **YouTube Channel Analytics** | Nov 1, 2019 – Nov 1, 2023 | YouTube Studio Export (`Table data.csv`, `Chart data.csv`, `Totals.csv`) | Publish timestamps, views, watch time, average view duration, impressions, CTR, daily channel trends              |
| **Raw Gameplay Clips**        | 2018 – Nov 2026           | Local video archive metadata & sampled HUD frames (`clips_metadata.csv`) | Clip creation timestamps, session clustering, clip duration, resolution/FPS, OCR-extracted HUD elimination counts |
| **Fortnite Tracker Stats**    | 2020 – Nov 2026           | Personal profile match & seasonal telemetry (`fortnite_stats.csv`)       | Seasonal K/D, win rate, matches played, kills per match, TRN rating across Chapters                               |

_(Note: Raw `.mp4`/`.mkv` video files are excluded via `.gitignore`; only extracted tabular metadata and telemetry scripts are stored in this repository.)_

---

## Methodology & Planned Pipeline

1. **Data Collection & Preprocessing (`src/`)**
   - Parse and clean YouTube Studio daily and video-level CSV exports.
   - Extract file creation timestamps, durations, and video properties across the 2018–2026 clip archive (`ffprobe` / `OpenCV`), plus OCR HUD telemetry on sampled highlight frames.
   - Collect and standardize 2020–2026 seasonal and match-level stats from Fortnite Tracker.
2. **Exploratory Data Analysis (`notebooks/01_youtube_eda.ipynb`)**
   - Visualize multi-year timelines aligning upload cadence, view decay, clip generation density, and seasonal K/D curves.
3. **Statistical Inference & Modeling (`notebooks/`)**
   - **Hypothesis Testing:** Mann-Whitney U and two-sample t-tests comparing Creator Era vs. Post-Creator Era performance.
   - **Time-Series Analysis:** Change-point detection (`ruptures`) and lagged cross-correlation to pinpoint the exact inflection point of channel decline.
   - **Predictive Modeling:** Regression/classification models linking gameplay intensity features to video retention metrics.

---

## Repository Structure

```text
DSA210-TermProject/
├── data/
│   ├── raw/
│   │   ├── youtube/            # Raw YouTube Studio CSV exports (2019–2023)
│   │   └── fortnite_tracker/   # Extracted Fortnite Tracker stats (2020–2026)
│   └── processed/              # Cleaned, merged datasets & clip metadata CSVs
├── notebooks/
│   └── 01_youtube_eda.ipynb    # Initial EDA on YouTube Studio analytics
├── src/                        # Data extraction, clip metadata, and OCR scripts
├── .gitignore
└── README.md
```
