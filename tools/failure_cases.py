import os
import csv
import argparse
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

# python tools/failure_cases.py --gen_dir ./generated/ddpm_cifar10 --out_dir ./runs/ddpm_cifar10/failures --topk 64


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gen_dir", type=str, required=True)
    p.add_argument("--out_dir", type=str, default="./failure_out")
    p.add_argument("--topk", type=int, default=64, help="每类失败案例展示多少张")
    p.add_argument("--thumb", type=int, default=64)
    return p.parse_args()

def list_images(d):
    exts = (".png", ".jpg", ".jpeg", ".webp")
    return [os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith(exts)]

def laplacian_var(gray: np.ndarray) -> float:
    # gray: HxW float32
    # Laplacian kernel: [[0,1,0],[1,-4,1],[0,1,0]]
    g = gray
    # pad
    p = np.pad(g, ((1,1),(1,1)), mode="edge")
    lap = (
        p[0:-2,1:-1] + p[2:,1:-1] + p[1:-1,0:-2] + p[1:-1,2:] - 4.0 * p[1:-1,1:-1]
    )
    return float(lap.var())

def ahash(img: Image.Image, size=8) -> str:
    g = ImageOps.grayscale(img)
    g = g.resize((size, size), Image.BICUBIC)
    arr = np.asarray(g).astype(np.float32)
    m = arr.mean()
    bits = (arr > m).astype(np.uint8).flatten()
    # 转成16进制字符串
    h = 0
    out = []
    for i, b in enumerate(bits):
        h = (h << 1) | int(b)
        if (i + 1) % 4 == 0:
            out.append(format(h, "x"))
            h = 0
    return "".join(out)

def save_grid(paths, out_png, thumb=64, ncol=8):
    if len(paths) == 0:
        return
    n = len(paths)
    ncol = min(ncol, n)
    nrow = int(np.ceil(n / ncol))
    canvas = Image.new("RGB", (ncol * thumb, nrow * thumb), (255, 255, 255))
    for i, p in enumerate(paths):
        img = Image.open(p).convert("RGB")
        img = ImageOps.fit(img, (thumb, thumb), method=Image.BICUBIC)
        r = i // ncol
        c = i % ncol
        canvas.paste(img, (c * thumb, r * thumb))
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    canvas.save(out_png)

def plot_hist(values, title, out_png):
    plt.figure()
    plt.hist(values, bins=50)
    plt.title(title)
    plt.xlabel(title)
    plt.ylabel("count")
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()

def main():
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)

    imgs = list_images(args.gen_dir)
    if not imgs:
        raise RuntimeError("gen_dir 没有图片")

    records = []
    hash_map = {}

    for p in imgs:
        im = Image.open(p).convert("RGB")
        arr = np.asarray(im).astype(np.float32) / 255.0  # HWC
        gray = (0.299 * arr[...,0] + 0.587 * arr[...,1] + 0.114 * arr[...,2]).astype(np.float32)

        blur = laplacian_var(gray)
        brightness = float(arr.mean())
        contrast = float(arr.std())
        saturation = float((arr.max(axis=2) - arr.min(axis=2)).mean())  # 简化饱和度 proxy

        h = ahash(im, size=8)
        hash_map.setdefault(h, []).append(p)

        records.append({
            "path": os.path.basename(p),
            "blur_lap_var": blur,
            "brightness_mean": brightness,
            "contrast_std": contrast,
            "saturation_proxy": saturation,
            "ahash": h,
        })

    # 输出CSV
    csv_path = os.path.join(args.out_dir, "image_scores.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        w.writeheader()
        w.writerows(records)
    print(f"Saved: {csv_path}")

    # 直方图
    plot_hist([r["blur_lap_var"] for r in records], "blur_lap_var", os.path.join(args.out_dir, "hist_blur.png"))
    plot_hist([r["brightness_mean"] for r in records], "brightness_mean", os.path.join(args.out_dir, "hist_brightness.png"))
    plot_hist([r["contrast_std"] for r in records], "contrast_std", os.path.join(args.out_dir, "hist_contrast.png"))
    plot_hist([r["saturation_proxy"] for r in records], "saturation_proxy", os.path.join(args.out_dir, "hist_saturation.png"))

    # 失败案例挑选
    rec_sorted_blur = sorted(records, key=lambda r: r["blur_lap_var"])  # 越小越模糊
    blur_fail = [os.path.join(args.gen_dir, r["path"]) for r in rec_sorted_blur[:args.topk]]
    save_grid(blur_fail, os.path.join(args.out_dir, "fail_blurry.png"), thumb=args.thumb, ncol=8)

    rec_sorted_bright = sorted(records, key=lambda r: r["brightness_mean"])
    dark_fail = [os.path.join(args.gen_dir, r["path"]) for r in rec_sorted_bright[:args.topk]]
    bright_fail = [os.path.join(args.gen_dir, r["path"]) for r in rec_sorted_bright[-args.topk:]]
    save_grid(dark_fail, os.path.join(args.out_dir, "fail_too_dark.png"), thumb=args.thumb, ncol=8)
    save_grid(bright_fail, os.path.join(args.out_dir, "fail_too_bright.png"), thumb=args.thumb, ncol=8)

    rec_sorted_sat = sorted(records, key=lambda r: r["saturation_proxy"])
    low_sat = [os.path.join(args.gen_dir, r["path"]) for r in rec_sorted_sat[:args.topk]]
    high_sat = [os.path.join(args.gen_dir, r["path"]) for r in rec_sorted_sat[-args.topk:]]
    save_grid(low_sat, os.path.join(args.out_dir, "fail_low_saturation.png"), thumb=args.thumb, ncol=8)
    save_grid(high_sat, os.path.join(args.out_dir, "fail_high_saturation.png"), thumb=args.thumb, ncol=8)

    # 疑似重复（mode collapse）：找出 ahash 出现次数最多的组
    dup_groups = sorted([(h, ps) for h, ps in hash_map.items() if len(ps) >= 3], key=lambda x: len(x[1]), reverse=True)
    if dup_groups:
        top = dup_groups[0][1][:args.topk]
        save_grid(top, os.path.join(args.out_dir, "fail_duplicates.png"), thumb=args.thumb, ncol=8)
        print(f"Found duplicate-like group size={len(dup_groups[0][1])}, saved fail_duplicates.png")
    else:
        print("No obvious duplicate-like groups found (ahash>=3).")

    print(f"All outputs in: {args.out_dir}")

if __name__ == "__main__":
    main()
