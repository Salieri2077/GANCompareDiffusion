import os
import re
import csv
import json
import argparse
import shutil
import torch
import matplotlib.pyplot as plt
import torch_fidelity

from src.unet import UNet
from src.diffusion import DDPM
from src.ema import EMA
from src.utils import make_dir, save_images_to_dir

# python tools/evaluate_checkpoints.py \
#   --ckpt_dir ./runs/ddpm_cifar10/checkpoints \
#   --real_dir ./real_images/cifar10_test \
#   --out_dir ./runs/ddpm_cifar10/eval_test \
#   --num_images 5000 --batch_size 100 \
#   --use_ema --sampler ddim --steps 50 --eta 0.0 --cuda

# python export_real_images.py --data_dir ./CIFARdata --out_dir ./real_images/cifar10_train --split train --num_images 50000

# python tools/evaluate_checkpoints.py \
#   --ckpt_dir ./runs/ddpm_cifar10/checkpoints \
#   --real_dir ./real_images/cifar10_train \
#   --out_dir ./runs/ddpm_cifar10/eval_train \
#   --num_images 5000 --batch_size 100 \
#   --use_ema --sampler ddim --steps 50 --eta 0.0 --cuda

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", type=str, required=True, help="runs/.../checkpoints")
    p.add_argument("--real_dir", type=str, required=True, help="真实图像目录（已导出png）")
    p.add_argument("--out_dir", type=str, default="./eval_out")
    p.add_argument("--num_images", type=int, default=5000)
    p.add_argument("--batch_size", type=int, default=100)

    p.add_argument("--T", type=int, default=1000)
    p.add_argument("--base_ch", type=int, default=128)

    p.add_argument("--use_ema", action="store_true")
    p.add_argument("--sampler", type=str, choices=["ddpm", "ddim"], default="ddim")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--eta", type=float, default=0.0)

    p.add_argument("--cuda", action="store_true")
    p.add_argument("--keep_gen", action="store_true", help="是否保留每个ckpt生成的图片目录")
    return p.parse_args()

def find_ckpts(ckpt_dir):
    files = [f for f in os.listdir(ckpt_dir) if f.endswith(".pt")]
    # 优先挑 epoch ckpt
    epoch_ckpts = []
    for f in files:
        m = re.search(r"ckpt_epoch_(\d+)\.pt", f)
        if m:
            epoch_ckpts.append((int(m.group(1)), os.path.join(ckpt_dir, f)))
    epoch_ckpts.sort(key=lambda x: x[0])
    if epoch_ckpts:
        return epoch_ckpts
    # fallback: last
    last = os.path.join(ckpt_dir, "ckpt_last.pt")
    if os.path.isfile(last):
        return [(999999, last)]
    raise RuntimeError("找不到 ckpt_epoch_*.pt 或 ckpt_last.pt")

@torch.no_grad()
def generate_images(ddpm, model, out_dir, num_images, batch_size, sampler, steps, eta):
    make_dir(out_dir)
    saved = 0
    while saved < num_images:
        bs = min(batch_size, num_images - saved)
        if sampler == "ddim":
            x = ddpm.sample_ddim(model, n=bs, steps=steps, eta=eta, image_size=32, channels=3)
        else:
            x = ddpm.sample(model, n=bs, image_size=32, channels=3)
        save_images_to_dir(x, out_dir, start_idx=saved)
        saved += bs

def plot_curve(xs, ys, title, xlabel, ylabel, out_png):
    plt.figure()
    plt.plot(xs, ys, marker="o")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()

def main():
    args = get_args()
    device = "cuda" if (args.cuda and torch.cuda.is_available()) else "cpu"
    make_dir(args.out_dir)

    ddpm = DDPM(T=args.T, device=device)
    ckpts = find_ckpts(args.ckpt_dir)

    rows = []
    for epoch, ckpt_path in ckpts:
        tag = f"epoch_{epoch:04d}" if epoch != 999999 else "last"
        gen_dir = os.path.join(args.out_dir, f"gen_{tag}")

        ckpt = torch.load(ckpt_path, map_location="cpu")
        model = UNet(base_ch=args.base_ch).to(device)
        model.load_state_dict(ckpt["model"], strict=True)

        if args.use_ema and "ema" in ckpt:
            ema = EMA(model, decay=0.0)
            ema.load_state_dict(ckpt["ema"])
            ema.copy_to(model)

        model.eval()
        generate_images(ddpm, model, gen_dir, args.num_images, args.batch_size, args.sampler, args.steps, args.eta)

        metrics = torch_fidelity.calculate_metrics(
            input1=gen_dir,
            input2=args.real_dir,
            cuda=(device == "cuda"),
            isc=True,
            fid=True,
            kid=True,
            verbose=False,
        )

        row = {"epoch": epoch, "ckpt": os.path.basename(ckpt_path)}
        row.update({k: float(v) for k, v in metrics.items()})
        rows.append(row)

        with open(os.path.join(args.out_dir, f"metrics_{tag}.json"), "w", encoding="utf-8") as f:
            json.dump(row, f, ensure_ascii=False, indent=2)

        print(f"[{tag}] {row}")

        if not args.keep_gen:
            shutil.rmtree(gen_dir, ignore_errors=True)

    # 写CSV
    csv_path = os.path.join(args.out_dir, "metrics.csv")
    keys = sorted({k for r in rows for k in r.keys()})
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Saved metrics CSV: {csv_path}")

    # 画曲线（如果有多个epoch）
    if len(rows) >= 2:
        xs = [r["epoch"] for r in rows]
        if "frechet_inception_distance" in rows[0]:
            ys = [r["frechet_inception_distance"] for r in rows]
            plot_curve(xs, ys, "FID over epochs", "epoch", "FID", os.path.join(args.out_dir, "fid_curve.png"))
        if "inception_score_mean" in rows[0]:
            ys = [r["inception_score_mean"] for r in rows]
            plot_curve(xs, ys, "Inception Score over epochs", "epoch", "IS (mean)", os.path.join(args.out_dir, "is_curve.png"))
        if "kernel_inception_distance_mean" in rows[0]:
            ys = [r["kernel_inception_distance_mean"] for r in rows]
            plot_curve(xs, ys, "KID over epochs", "epoch", "KID (mean)", os.path.join(args.out_dir, "kid_curve.png"))

if __name__ == "__main__":
    main()
