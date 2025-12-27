import os
import math
import random
import numpy as np
import torch
from PIL import Image

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

def make_dir(path: str):
    os.makedirs(path, exist_ok=True)

@torch.no_grad()
def save_image_grid(x: torch.Tensor, path: str, nrow: int = 8):
    """
    x: (B,3,H,W) in [-1,1]
    """
    x = (x.clamp(-1, 1) + 1) * 0.5
    x = (x * 255).byte().cpu()
    b, c, h, w = x.shape
    ncol = nrow
    nrow_grid = math.ceil(b / ncol)
    grid = torch.zeros((c, nrow_grid * h, ncol * w), dtype=torch.uint8)

    for idx in range(b):
        r = idx // ncol
        col = idx % ncol
        grid[:, r*h:(r+1)*h, col*w:(col+1)*w] = x[idx]

    grid = grid.permute(1, 2, 0).numpy()
    Image.fromarray(grid).save(path)

@torch.no_grad()
def save_images_to_dir(x: torch.Tensor, out_dir: str, start_idx: int = 0):
    """
    x: (B,3,H,W) in [-1,1]
    保存为 out_dir/000000.png ...
    """
    make_dir(out_dir)
    x = (x.clamp(-1, 1) + 1) * 0.5
    x = (x * 255).byte().cpu().permute(0, 2, 3, 1).numpy()
    for i in range(x.shape[0]):
        Image.fromarray(x[i]).save(os.path.join(out_dir, f"{start_idx + i:06d}.png"))
