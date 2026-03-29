import os
import cv2
import sys
import torch
import argparse
import numpy as np
from torch.nn import functional as F
import warnings
warnings.filterwarnings("ignore")

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--img0', type=str, required=True)
parser.add_argument('--img1', type=str, required=True)
parser.add_argument('--output', type=str, required=True)
parser.add_argument('--model', type=str, default='ckpt')
parser.add_argument('--scale', type=float, default=1.0)
parser.add_argument('--TTA', action='store_true')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── 1. Handle Checkpoint Path ─────────────────────────────────────
if os.path.isdir(args.model):
    pkl_files = [f for f in os.listdir(args.model) if f.endswith('.pkl')]
    if not pkl_files:
        print(f"❌ No .pkl files found in directory: {args.model}")
        sys.exit(1)
    ckpt_name = 'OURS-B.pkl' if 'OURS-B.pkl' in pkl_files else pkl_files[0]
    ckpt_path = os.path.join(args.model, ckpt_name)
else:
    ckpt_path = args.model
    ckpt_name = os.path.basename(args.model)

model_core_name = ckpt_name.replace('.pkl', '')

# ── 2. Load Config using the Authors' Model_create function ───────
try:
    from Trainer import Model
    from config import Model_create
except ImportError as e:
    print(f"❌ Failed to import from repository: {e}")
    sys.exit(1)

parts = model_core_name.split('-')
if len(parts) >= 2:
    model_string = f"{parts[0].capitalize()}-{parts[1].upper()}"
else:
    model_string = 'Ours-B' 

MODEL_CONFIG = Model_create(model_string)
if MODEL_CONFIG is None:
    MODEL_CONFIG = Model_create('Ours-B')

# Initialize model
model = Model(-1, MODEL_CONFIG)

# ── 3. ROBUST CUSTOM WEIGHT LOADER ────────────────────────────────
checkpoint = torch.load(ckpt_path, map_location='cpu')

if isinstance(checkpoint, dict) and 'model' in checkpoint:
    state_dict = checkpoint['model']
elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
    state_dict = checkpoint['state_dict']
else:
    state_dict = checkpoint

clean_state_dict = {}
for k, v in state_dict.items():
    clean_key = k.replace('module.', '').replace('net.', '')
    clean_state_dict[clean_key] = v

if hasattr(model, 'net'):
    missing, unexpected = model.net.load_state_dict(clean_state_dict, strict=False)
else:
    missing, unexpected = model.load_state_dict(clean_state_dict, strict=False)

print(f"✅ Weights forcefully loaded. (Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)})")

model.eval()
model.device()

# ── 4. Image Preprocessing ────────────────────────────────────────
img0 = cv2.imread(args.img0)
img1 = cv2.imread(args.img1)

if img0 is None or img1 is None:
    print("❌ Error reading input images.")
    sys.exit(1)

img0 = (torch.tensor(img0.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)
img1 = (torch.tensor(img1.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)

n, c, h, w = img0.shape
ph = ((h - 1) // 32 + 1) * 32
pw = ((w - 1) // 32 + 1) * 32
padding = (0, pw - w, 0, ph - h)
img0 = F.pad(img0, padding)
img1 = F.pad(img1, padding)

# ── 5. Run Inference with Tensor Timestep Fix ─────────────────────
print("Interpolating...")
timestep_tensor = torch.tensor([0.5], dtype=torch.float32, device=device)

with torch.no_grad():
    if hasattr(model, 'inference'):
        # Pass the tensor explicitly to avoid the float attribute error
        mid = model.inference(img0, img1, TTA=args.TTA, fast_TTA=args.TTA, timestep=timestep_tensor)
    else:
        mid = model.update(img0, img1, timestep=timestep_tensor)
        
if isinstance(mid, (list, tuple)):
    mid = mid[0]

# Unpad and save
mid = mid[:, :, :h, :w]
mid = (mid[0].cpu().numpy().transpose(1, 2, 0) * 255.).clip(0, 255).astype(np.uint8)

cv2.imwrite(args.output, mid)
print(f"✅ Success! Interpolated frame saved to {args.output}")
