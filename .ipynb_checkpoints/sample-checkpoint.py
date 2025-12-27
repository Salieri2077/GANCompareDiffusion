import os
import argparse
import torch
from tqdm import trange

from src.unet import UNet
from src.diffusion import DDPM
from src.utils import make_dir, save_images_to_dir

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--out_dir", type=str, default="./generated/ddpm_cifar10")
    p.add_argument("--num_images", type=int, default=5000)
    p.add_argument("--batch_size", type=int, default=100)
    p.add_argument("--T", type=int, default=1000)
    p.add_argument("--base_ch", type=int, default=128)
    p.add_argument("--use_ema", action="store_true")

    p.add_argument("--sampler", type=str, choices=["ddpm", "ddim"], default="ddim")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--eta", type=float, default=0.0)

    return p.parse_args()

def main():
    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    make_dir(args.out_dir)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    model = UNet(base_ch=args.base_ch).to(device)
    model.load_state_dict(ckpt["model"], strict=True)

    if args.use_ema and "ema" in ckpt:
        # 把 EMA 权重 copy 到模型
        from src.ema import EMA
        ema = EMA(model, decay=0.0)
        ema.load_state_dict(ckpt["ema"])
        ema.copy_to(model)

    model.eval()
    ddpm = DDPM(T=args.T, device=device)

    saved = 0
    for _ in trange((args.num_images + args.batch_size - 1) // args.batch_size, desc="Sampling"):
        bs = min(args.batch_size, args.num_images - saved)
        if bs <= 0:
            break
            
        if args.sampler == "ddim":
            x_gen = ddpm.sample_ddim(model, n=bs, steps=args.steps, eta=args.eta, image_size=32, channels=3)
        else:
            x_gen = ddpm.sample(model, n=bs, image_size=32, channels=3)
            
        save_images_to_dir(x_gen, args.out_dir, start_idx=saved)
        saved += bs

    print(f"Done. Saved {saved} images to {args.out_dir}")

if __name__ == "__main__":
    main()
