import os
import random
import argparse
from PIL import Image, ImageOps
# python tools/compare_real_vs_gen.py --real_dir ./real_images/cifar10_test --gen_dir ./generated/ddpm_cifar10 --out_png ./runs/ddpm_cifar10/real_vs_gen.png --pairs 40

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--real_dir", type=str, required=True)
    p.add_argument("--gen_dir", type=str, required=True)
    p.add_argument("--out_png", type=str, default="./real_vs_gen.png")
    p.add_argument("--pairs", type=int, default=32, help="展示多少对（真实,生成）")
    p.add_argument("--thumb", type=int, default=64, help="缩略图边长")
    return p.parse_args()

def list_images(d):
    exts = (".png", ".jpg", ".jpeg", ".webp")
    return [os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith(exts)]

def main():
    args = get_args()
    real_list = list_images(args.real_dir)
    gen_list = list_images(args.gen_dir)
    if len(real_list) == 0 or len(gen_list) == 0:
        raise RuntimeError("real_dir 或 gen_dir 没有图片")

    pairs = min(args.pairs, len(real_list), len(gen_list))
    real_sel = random.sample(real_list, pairs)
    gen_sel = random.sample(gen_list, pairs)

    # 2 列：左真实 右生成；共 pairs 行
    w = args.thumb * 2
    h = args.thumb * pairs
    canvas = Image.new("RGB", (w, h), (255, 255, 255))

    for i in range(pairs):
        r = Image.open(real_sel[i]).convert("RGB")
        g = Image.open(gen_sel[i]).convert("RGB")
        r = ImageOps.fit(r, (args.thumb, args.thumb), method=Image.BICUBIC)
        g = ImageOps.fit(g, (args.thumb, args.thumb), method=Image.BICUBIC)
        canvas.paste(r, (0, i * args.thumb))
        canvas.paste(g, (args.thumb, i * args.thumb))

    os.makedirs(os.path.dirname(args.out_png) or ".", exist_ok=True)
    canvas.save(args.out_png)
    print(f"Saved to: {args.out_png}")

if __name__ == "__main__":
    main()
