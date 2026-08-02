"""
extract_frames.py — Extract frames from the real UR5 MP4 and prepare training data.

Uses ffmpeg to decode the video, converts to grayscale 64×64, and produces:
  - real_frames_full.npy  (N, 64, 64) float32 [0,1] — all frames for the HTML viewer
  - real_clips.npy        (N-T+1, T, 64, 64) float32 — sliding windows for V-JEPA training
"""
import subprocess, struct, sys
import numpy as np
from PIL import Image
import io

VIDEO = "YTDown.com_Shorts_UR5-demonstration_Media_RPhDaaa79vg_001_1080p.mp4"
OUT_SIZE = 64
T = 8  # V-JEPA temporal window

def extract_all_frames(video_path, size=64):
    """Use ffmpeg to decode video → raw grayscale frames at target resolution."""
    # First probe the video to get frame count and dimensions
    probe = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "flat",
         "-show_entries", "stream=width,height,nb_frames,r_frame_rate,duration",
         video_path],
        capture_output=True, text=True
    )
    print("ffprobe output:", probe.stdout[:500])

    # Extract raw frames as grayscale at target resolution
    cmd = [
        "ffmpeg", "-i", video_path,
        "-vf", f"scale={size}:{size}:force_original_aspect_ratio=increase,crop={size}:{size},format=gray",
        "-f", "rawvideo", "-pix_fmt", "gray",
        "-v", "quiet", "-"
    ]
    result = subprocess.run(cmd, capture_output=True)
    raw = result.stdout

    n_frames = len(raw) // (size * size)
    print(f"Extracted {n_frames} frames at {size}×{size} grayscale")

    if n_frames == 0:
        print("ERROR: No frames extracted! ffmpeg stderr:", result.stderr.decode()[:500])
        sys.exit(1)

    frames = np.frombuffer(raw, dtype=np.uint8).reshape(n_frames, size, size)
    frames = frames.astype(np.float32) / 255.0
    return frames


if __name__ == "__main__":
    print(f"Extracting frames from {VIDEO}...")
    frames = extract_all_frames(VIDEO, OUT_SIZE)
    print(f"frames shape: {frames.shape}, range [{frames.min():.3f}, {frames.max():.3f}]")

    # Save all frames for the HTML viewer
    np.save("real_frames_full.npy", frames)
    print(f"Saved real_frames_full.npy ({frames.shape})")

    # Create sliding windows of T frames for V-JEPA training
    N = len(frames)
    n_clips = N - T + 1
    clips = np.zeros((n_clips, T, OUT_SIZE, OUT_SIZE), dtype=np.float32)
    for i in range(n_clips):
        clips[i] = frames[i:i+T]

    np.save("real_clips.npy", clips)
    print(f"Saved real_clips.npy ({clips.shape})")
    print(f"Video: {N} frames → {n_clips} training clips of {T} frames each")
