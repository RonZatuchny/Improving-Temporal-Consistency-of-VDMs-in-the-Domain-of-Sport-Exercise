import torch
import numpy as np
from transformers import ViTImageProcessor, ViTModel
from PIL import Image
import cv2
import os
import sys

# Check command-line arguments
if len(sys.argv) != 2:
    print("Usage: python dino_score.py <parent_dir>")
    sys.exit(1)

# Set parent directory and GPU device
parent_dir = sys.argv[1]
target_frames = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load all frames from an MP4 video, resizing to 512x512
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

# Compute DINO score: Cosine similarity of frame features
def compute_dino_score(frames):
    processor = ViTImageProcessor.from_pretrained('facebook/dino-vits16')
    model = ViTModel.from_pretrained('facebook/dino-vits16').to(device)
    model.eval()
    features = []
    for frame in frames:
        img = Image.fromarray(frame)
        inputs = processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            cls_feature = outputs.last_hidden_state[:, 0, :]  # Shape: (1, 384)
        features.append(cls_feature.squeeze().cpu().numpy())
    features = np.array(features)
    ref_feature = features[0]
    similarities = np.dot(features, ref_feature) / (np.linalg.norm(features, axis=1) * np.linalg.norm(ref_feature))
    return np.mean(similarities[1:]) if len(similarities) > 1 else 1.0

# Initialize dictionary for average scores
dino_scores = {}
all_scores = []
# Iterate through condition subdirectories
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
                    #frames_dino = trim_uniform(frames, target_frames)
                    score = compute_dino_score(frames)
                    print(f"  Video: {video}, DINO Score: {score:.4f}")
                    scores.append(score)
                    all_scores.append(score)
        if scores:
            dino_scores[condition] = np.mean(scores)

# Print average DINO scores per condition
print("\nAverage DINO Scores per Condition:")
for condition, score in dino_scores.items():
    print(f"{condition}: {score:.4f}")

# Print overall average
if all_scores:
    overall_avg = np.mean(all_scores)
    print(f"\nOverall Average DINO Score: {overall_avg:.4f}")