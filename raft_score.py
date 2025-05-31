import torch
import numpy as np
import cv2
import os
import sys
from torchvision.models.optical_flow import raft_large
from torchvision.transforms import ToTensor

# Check command-line arguments
if len(sys.argv) != 2:
    print("Usage: python raft_score.py <parent_dir>")
    sys.exit(1)

# Set parent directory and GPU device
parent_dir = sys.argv[1]
target_frames = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load frames from MP4
def load_frames_from_mp4(video_path):
    frames = []
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (512, 512))
        frames.append(frame)
    cap.release()
    return frames

# Sample frames
def trim_uniform(frames, target_frames):
    T = len(frames)
    if T > target_frames:
        interval = T / target_frames
        indices = np.floor(np.arange(0, T, interval)).astype(int)[:target_frames]
        frames = [frames[i] for i in indices]
    elif T < target_frames:
        frames.extend([frames[-1]] * (target_frames - T))
    return frames

# Compute RAFT score: Mean optical flow magnitude
def compute_raft_score(frames):
    model = raft_large(pretrained=True, progress=False).to(device)
    model.eval()
    transform = ToTensor()
    flows = []
    for i in range(len(frames) - 1):
        img1 = transform(frames[i]).unsqueeze(0).to(device)
        img2 = transform(frames[i + 1]).unsqueeze(0).to(device)
        with torch.no_grad():
            flow = model(img1, img2)[-1]
        flow_magnitude = torch.sqrt(flow[:, 0, :, :] ** 2 + flow[:, 1, :, :] ** 2)
        flows.append(flow_magnitude.mean().item())
    return np.mean(flows) if flows else 0.0

# Process videos
raft_scores = {}
all_scores = []
for condition in os.listdir(parent_dir):
    condition_path = os.path.join(parent_dir, condition)
    if os.path.isdir(condition_path) and condition != 'reference':
        scores = []
        print(f"\nProcessing condition: {condition}")
        for video in os.listdir(condition_path):
            if video.endswith('.mp4'):
                video_path = os.path.join(condition_path, video)
                frames = load_frames_from_mp4(video_path)
                if frames:
                    frames_raft = trim_uniform(frames, target_frames)
                    score = compute_raft_score(frames_raft)
                    print(f"  Video: {video}, RAFT Score: {score:.4f}")
                    scores.append(score)
                    all_scores.append(score)
        if scores:
            raft_scores[condition] = np.mean(scores)

# Print averages
print("\nAverage RAFT Scores per Condition:")
for condition, score in raft_scores.items():
    print(f"{condition}: {score:.4f}")

# Print overall average
if all_scores:
    overall_avg = np.mean(all_scores)
    print(f"\nOverall Average RAFT Score: {overall_avg:.4f}")