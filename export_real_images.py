import os
import argparse
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils import make_dir, save_images_to_dir

def get_args():
    p = argparse.ArgumentParser()
    # p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--data_dir", type=str, default="./CIFARdata")

    p.add_argument("--out_dir", type=str, default="./real_images/cifar10_test")
    p.add_argument("--split", type=str, choices=["train", "test"], default="test")
    p.add_argument("--num_images", type=int, default=10000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=4)
    return p.parse_args()

def main():
    args = get_args()
    make_dir(args.out_dir)

    # 注意：这里不做 Normalize，直接保存 [0,1] -> 转 [-1,1] 再走统一保存逻辑
    tfm = transforms.ToTensor()
    is_train = (args.split == "train")
    # ds = datasets.CIFAR10(root=args.data_dir, train=is_train, download=True, transform=tfm)
    ds = datasets.CIFAR10(root=args.data_dir, train=is_train, download=False, transform=tfm)

    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, pin_memory=True)

    saved = 0
    for x, _ in tqdm(dl, desc=f"Export CIFAR10 {args.split}"):
        # x: [0,1] -> [-1,1]
        x = x * 2 - 1
        bs = x.shape[0]
        remain = args.num_images - saved
        if remain <= 0:
            break
        if bs > remain:
            x = x[:remain]
            bs = remain
        save_images_to_dir(x, args.out_dir, start_idx=saved)
        saved += bs

    print(f"Done. Exported {saved} images to {args.out_dir}")

if __name__ == "__main__":
    main()
