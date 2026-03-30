import os
import cv2
import sys
import torch
import argparse
import numpy as np
from torch.nn import functional as F
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# Parse arguments
parser = argparse.ArgumentParser(description='LC-Mamba Video Interpolation (2x)')
parser.add_argument('--video', type=str, required=True, help='Path to input video')
parser.add_argument('--output', type=str, required=True, help='Path to output video (.mp4)')
parser.add_argument('--model', type=str, default='ckpt')
parser.add_argument('--TTA', action='store_true')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── 1. Setup Model (Using our fixed logic!) ───────────────────────
if os.path.isdir(args.model):
    pkl_files = [f for f in os.listdir(args.model) if f.endswith('.pkl')]
    ckpt_name = 'OURS-B.pkl' if 'OURS-B.pkl' in pkl_files else pkl_files[0]
    ckpt_path = os.path.join(args.model, ckpt_name)
else:
    ckpt_path = args.model
    ckpt_name = os.path.basename(args.model)

model_core_name = ckpt_name.replace('.pkl', '')

try:
    from Trainer import Model
    from config import Model_create
except ImportError:
    print("❌ Run this from inside the LC-Mamba folder!")
    sys.exit(1)

parts = model_core_name.split('-')
model_string = f"{parts[0].capitalize()}-{parts[1].upper()}" if len(parts) >= 2 else 'Ours-B'
MODEL_CONFIG = Model_create(model_string) or Model_create('Ours-B')
model = Model(-1, MODEL_CONFIG)

checkpoint = torch.load(ckpt_path, map_location='cpu')
state_dict = checkpoint.get('model_state_dict', checkpoint.get('model', checkpoint.get('state_dict', checkpoint)))

clean_state_dict = {}
for k, v in state_dict.items():
    clean_key = k.replace('module.', '').replace('net.', '')
    if clean_key.startswith('uup'): clean_key = clean_key.replace('uup', 'unet.up', 1)
    clean_state_dict[clean_key] = v

if hasattr(model, 'net'): model.net.load_state_dict(clean_state_dict, strict=False)
else: model.load_state_dict(clean_state_dict, strict=False)

model.eval()
model.device()
print("✅ LC-Mamba Model loaded and ready!")

# ── 2. Process Video ──────────────────────────────────────────────
cap = cv2.VideoCapture(args.video)
if not cap.isOpened():
    print(f"❌ Cannot open video: {args.video}")
    sys.exit(1)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# We are doubling the framerate!
out_fps = fps * 2
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(args.output, fourcc, out_fps, (width, height))

print(f"🎥 Processing Video: {width}x{height} | {fps} FPS -> {out_fps} FPS")

# Padding logic for VFI
ph = ((height - 1) // 32 + 1) * 32
pw = ((width - 1) // 32 + 1) * 32
padding = (0, pw - width, 0, ph - height)
timestep_tensor = torch.tensor([0.5], dtype=torch.float32, device=device)

ret, last_frame = cap.read()
if not ret:
    print("❌ Video is empty.")
    sys.exit(1)

# Write the very first frame
out.write(last_frame)

pbar = tqdm(total=total_frames - 1, desc="Interpolating Frames")
while True:
    ret, frame = cap.read()
    if not ret: break

    # Prep frames for neural net
    img0 = (torch.tensor(last_frame.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)
    img1 = (torch.tensor(frame.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)
    
    img0 = F.pad(img0, padding)
    img1 = F.pad(img1, padding)

    # Predict the middle frame
    with torch.no_grad():
        if hasattr(model, 'inference'):
            mid = model.inference(img0, img1, TTA=args.TTA, fast_TTA=args.TTA, timestep=timestep_tensor)
        else:
            mid = model.update(img0, img1, timestep=timestep_tensor)
            
    if isinstance(mid, (list, tuple)): mid = mid[0]
    
    # Format and unpad
    mid = mid[:, :, :height, :width]
    mid = (mid[0].cpu().numpy().transpose(1, 2, 0) * 255.).clip(0, 255).astype(np.uint8)

    # Write the predicted middle frame, THEN the actual next frame
    out.write(mid)
    out.write(frame)
    
    last_frame = frame
    pbar.update(1)

cap.release()
out.release()
print(f"\n🎉 Success! 2x Smoothed video saved to: {args.output}")
