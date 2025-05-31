import torch
import numpy as np
import cv2
import os
import sys
from tqdm import tqdm
sys.path.append("/content/drive/MyDrive/Improving-Temporal-Consistency-of-VDMs-in-the-Domain-of-Sport-Exercise")
from calculate_fvd import calculate_fvd

# Check command-line arguments
if len(sys.argv) != 3:
    print("Usage: python process_fvd.py <parent_dir> <ref_dir>")
    sys.exit(1)

# Set paths and constants
parent_dir = sys.argv[1]
ref_dir = sys.argv[2]
NUM_FRAMES = 16
RESOLUTION = 224
MIN_FRAMES = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load frames from MP4
def load_frames_from_mp4(video_path):
    frames = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open {video_path}")
        return []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (RESOLUTION, RESOLUTION))
        frames.append(frame)
    cap.release()
    return frames

# Sample frames uniformly
def sample_frames(frames, num_frames):
    T = len(frames)
    if T < MIN_FRAMES:
        print(f"Warning: Video has only {T} frames, needs at least {MIN_FRAMES}")
        return None
    if T > num_frames:
        indices = np.linspace(0, T-1, num_frames, dtype=int)
        frames = [frames[i] for i in indices]
    elif T < num_frames:
        frames.extend([frames[-1]] * (num_frames - T))
    frames = np.array(frames, dtype=np.float32) / 255.0
    print(f"Frame shape: {frames.shape}, min: {frames.min():.4f}, max: {frames.max():.4f}")
    return frames

# Process all videos in each subfolder as a single batch
fvd_scores = {}
for condition in os.listdir(parent_dir):
    condition_path = os.path.join(parent_dir, condition)
    if os.path.isdir(condition_path) and condition != 'reference':
        ref_video_path = os.path.join(ref_dir, f"{condition}.mp4")
        if not os.path.exists(ref_video_path):
            print(f"Warning: Reference video {ref_video_path} not found, skipping {condition}")
            continue
        ref_frames = load_frames_from_mp4(ref_video_path)
        if not ref_frames:
            print(f"Warning: Failed to load reference video {ref_video_path}, skipping {condition}")
            continue
        ref_frames = sample_frames(ref_frames, NUM_FRAMES)
        if ref_frames is None:
            print(f"Warning: Insufficient frames in reference video {ref_video_path}, skipping {condition}")
            continue
        ref_tensor = torch.from_numpy(ref_frames).permute(0, 3, 1, 2)  # [T, C, H, W]

        # Collect all generated videos in the subfolder
        gen_video_paths = [os.path.join(condition_path, v) for v in os.listdir(condition_path) if v.endswith('.mp4')]
        if not gen_video_paths:
            print(f"Warning: No videos in {condition}, skipping")
            continue

        print(f"\nProcessing condition: {condition} ({len(gen_video_paths)} videos)")
        gen_tensors = []
        for video_path in tqdm(gen_video_paths, desc="Loading videos"):
            frames = load_frames_from_mp4(video_path)
            if not frames:
                print(f"Warning: Failed to load {video_path}")
                continue
            gen_frames = sample_frames(frames, NUM_FRAMES)
            if gen_frames is None:
                print(f"Warning: Insufficient frames in {video_path}")
                continue
            gen_tensor = torch.from_numpy(gen_frames).permute(0, 3, 1, 2)  # [T, C, H, W]
            gen_tensors.append(gen_tensor)

        if not gen_tensors:
            print(f"Warning: No valid videos in {condition}, skipping")
            continue

        # Stack all generated videos
        gen_videos = torch.stack(gen_tensors, dim=0)  # [N, T, C, H, W]
        # Create N copies of reference video
        ref_videos = ref_tensor.repeat(len(gen_tensors), 1, 1, 1, 1)  # [N, T, C, H, W]

        print(f"Gen videos shape: {gen_videos.shape}, min: {gen_videos.min():.4f}, max: {gen_videos.max():.4f}")
        print(f"Ref videos shape: {ref_videos.shape}, min: {ref_videos.min():.4f}, max: {ref_videos.max():.4f}")

        try:
            result = calculate_fvd(gen_videos, ref_videos, device, method='styleganv', only_final=True)
            fvd_score = result["value"][0]
            print(f"  {condition} FVD Score: {fvd_score:.4f}")
            if fvd_score > 1000:
                print(f"  Warning: High FVD score ({fvd_score:.4f}), check frame consistency")
            fvd_scores[condition] = fvd_score
        except Exception as e:
            print(f"Error computing FVD for {condition}: {str(e)}")

# Print results
print("\nFVD Scores per Condition:")
for condition, score in fvd_scores.items():
    print(f"{condition}: {score:.4f}")

if fvd_scores:
    overall_avg = np.mean(list(fvd_scores.values()))
    print(f"\nOverall Average FVD Score: {overall_avg:.4f}")
else:
    print("\nNo valid FVD scores computed.")