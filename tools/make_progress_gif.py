import os
import re
import argparse
from PIL import Image

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--samples_dir", type=str, required=True, help="runs/.../samples 目录")
    p.add_argument("--out_gif", type=str, default="./progress.gif")
    p.add_argument("--every", type=int, default=1, help="每隔多少张取一帧")
    p.add_argument("--duration_ms", type=int, default=200)
    return p.parse_args()

def main():
    args = get_args()
    files = [f for f in os.listdir(args.samples_dir) if f.lower().endswith(".png")]

    def epoch_key(name):
        m = re.search(r"epoch_(\d+)", name)
        return int(m.group(1)) if m else 10**9

    files.sort(key=epoch_key)
    files = files[::max(1, args.every)]
    if not files:
        raise RuntimeError("samples_dir 下找不到 epoch_*.png")

    frames = []
    for f in files:
        img = Image.open(os.path.join(args.samples_dir, f)).convert("RGB")
        frames.append(img)

    frames[0].save(
        args.out_gif,
        save_all=True,
        append_images=frames[1:],
        duration=args.duration_ms,
        loop=0,
    )
    print(f"Saved GIF to: {args.out_gif} (frames={len(frames)})")

if __name__ == "__main__":
    main()
