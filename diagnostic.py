
import torch
import sys
sys.path.append('/teamspace/studios/this_studio/LC-Mamba')

from Trainer import Model
from config import Model_create

ckpt_path = '/teamspace/studios/this_studio/LC-Mamba/ckpt/OURS-B.pkl'
checkpoint = torch.load(ckpt_path, map_location='cpu')
ckpt_dict = checkpoint['model_state_dict']

clean_ckpt_dict = {}
for k, v in ckpt_dict.items():
    clean_key = k.replace('module.', '').replace('net.', '')
    clean_ckpt_dict[clean_key] = v

model = Model(-1, Model_create('Ours-B'))
model_dict = model.net.state_dict() if hasattr(model, 'net') else model.state_dict()

missing = [k for k in model_dict.keys() if k not in clean_ckpt_dict]
unexpected = [k for k in clean_ckpt_dict.keys() if k not in model_dict]

print("--- EXPECTED BY CODE (MISSING FROM CKPT) ---")
for m in missing[:5]: print(m)
print("...")

print("\n--- ACTUAL IN CHECKPOINT (UNEXPECTED BY CODE) ---")
for u in unexpected[:5]: print(u)
print("...")
