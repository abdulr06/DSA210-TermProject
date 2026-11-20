import os
import re
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd

# Silence noisy OpenCV/FFmpeg C++ stderr warnings before importing cv2
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"
import cv2

# ADD ALL YOUR CLIP FOLDER PATHS HERE
CLIPS_DIRS = [
    Path(r"D:\Abdul Rahman\Old Fortnte Clips 2020"),
    Path(r"D:\Fortnite"),
    Path(r"C:\Users\Abdul Rehman\Videos\NVIDIA\Fortnite"),
    Path(r"D:\Captures\OG"),
    Path(r"D:\Captures\Some Clips"),
    Path(r"D:\Captures\Chapter 2")
]

OUTPUT_CSV = Path("data/processed/clips_metadata.csv")
VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi"}
FILENAME_DATE_RE = re.compile(r"(20\d{2})[.\-_](\d{2})[.\-_](\d{2})")

class SuppressStderr:
    """Redirects low-level C/FFmpeg stderr noise during corrupted file checks."""
    def __enter__(self):
        self.null_fd = os.open(os.devnull, os.O_RDWR)
        self.save_fd = os.dup(2)
        os.dup2(self.null_fd, 2)

    def __exit__(self, *_):
        os.dup2(self.save_fd, 2)
        os.close(self.null_fd)
        os.close(self.save_fd)

def extract_best_timestamp(file_path: Path) -> datetime:
    stat_dt = datetime.fromtimestamp(file_path.stat().st_mtime)
    match = FILENAME_DATE_RE.search(file_path.name)
    if match:
        try:
            year, month, day = map(int, match.groups())
            if (stat_dt.year, stat_dt.month, stat_dt.day) == (year, month, day):
                return stat_dt
            return datetime(year, month, day, stat_dt.hour, stat_dt.minute, stat_dt.second)
        except ValueError:
            pass
    return stat_dt

def scan_all_clips(directories: list[Path]) -> pd.DataFrame:
    records = []
    seen_files = set()
    scanned_count = 0

    for root_dir in directories:
        if not root_dir.exists():
            print(f"Skipping missing path: {root_dir}")
            continue

        print(f"Scanning: {root_dir} ...")
        for file_path in root_dir.rglob("*"):
            if file_path.suffix.lower() not in VIDEO_EXTS:
                continue

            stat = file_path.stat()
            dedup_key = (file_path.name.lower(), stat.st_size)
            if dedup_key in seen_files:
                continue
            seen_files.add(dedup_key)

            scanned_count += 1
            if scanned_count % 100 == 0:
                print(f"  Processed {scanned_count} unique clips...")

            created_dt = extract_best_timestamp(file_path)
            size_mb = round(stat.st_size / (1024 * 1024), 2)

            with SuppressStderr():
                cap = cv2.VideoCapture(str(file_path))
                is_valid = cap.isOpened()
                fps = cap.get(cv2.CAP_PROP_FPS) or 0 if is_valid else 0
                frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0 if is_valid else 0
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0) if is_valid else 0
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0) if is_valid else 0
                cap.release()

            duration_sec = round(frame_count / fps, 2) if fps > 0 else 0.0
            is_playable = is_valid and duration_sec > 0 and width > 0

            records.append({
                "filename": file_path.name,
                "source_root": str(root_dir),
                "subfolder": file_path.parent.name,
                "timestamp": created_dt.isoformat(),
                "date": created_dt.date().isoformat(),
                "year": created_dt.year,
                "month": created_dt.month,
                "hour": created_dt.hour,
                "day_of_week": created_dt.strftime("%A"),
                "duration_sec": duration_sec,
                "fps": round(fps, 1),
                "resolution": f"{width}x{height}",
                "size_mb": size_mb,
                "is_playable": is_playable,
                "era": "Creator_Era" if created_dt < datetime(2023, 11, 2) else "Post_Creator_Era",
            })

    df = pd.DataFrame(records)
    if not df.empty:
        df = df.sort_values("timestamp").reset_index(drop=True)
    return df

if __name__ == "__main__":
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_clips = scan_all_clips(CLIPS_DIRS)
    df_clips.to_csv(OUTPUT_CSV, index=False)
    print(f"\nDone! Saved {len(df_clips)} unique clips to {OUTPUT_CSV}")
    if not df_clips.empty:
        print(f"Playable clips: {df_clips['is_playable'].sum()} | Corrupted/Incomplete: {(~df_clips['is_playable']).sum()}")
        print("\nClips by Year:")
        print(df_clips["year"].value_counts().sort_index())
        print("\nClips by Era:")
        print(df_clips["era"].value_counts())

        