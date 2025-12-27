import os
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from src.utils import seed_everything, make_dir, save_image_grid
from src.unet import UNet
from src.diffusion import DDPM, ddpm_loss
from src.ema import EMA

def get_args():
    p = argparse.ArgumentParser()
    # p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--data_dir", type=str, default="./CIFARdata")
    p.add_argument("--save_dir", type=str, default="./runs/ddpm_cifar10")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--epochs", type=int, default=200) # epochs200只是diffusion的一个基础迭代数
    # p.add_argument("--epochs", type=int, default=12)

    p.add_argument("--num_workers", type=int, default=4)

    p.add_argument("--T", type=int, default=1000)
    p.add_argument("--base_ch", type=int, default=128)
    p.add_argument("--ema_decay", type=float, default=0.9999)
    p.add_argument("--grad_clip", type=float, default=1.0)

    p.add_argument("--amp", action="store_true")
    p.add_argument("--sample_every", type=int, default=1)   # 每 N 个 epoch 生成一次网格图
    p.add_argument("--save_every", type=int, default=5)     # 每 N 个 epoch 存一次 ckpt
    return p.parse_args()

def main():
    args = get_args()
    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    make_dir(args.save_dir)
    make_dir(os.path.join(args.save_dir, "samples"))
    make_dir(os.path.join(args.save_dir, "checkpoints"))

    tfm = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # -> [-1,1]
    ])
    # trainset = datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=tfm)
    trainset = datasets.CIFAR10(root=args.data_dir, train=True, download=False, transform=tfm)

    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                             num_workers=args.num_workers, pin_memory=True, drop_last=True)

    model = UNet(base_ch=args.base_ch).to(device)
    ddpm = DDPM(T=args.T, device=device)
    ema = EMA(model, decay=args.ema_decay)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(trainloader, desc=f"Epoch {epoch}/{args.epochs}")
        for x, _ in pbar:
            x = x.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=args.amp):
                loss = ddpm_loss(ddpm, model, x)

            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            # scaler.step(opt)
            # scaler.update()
            # train.py 修改部分
            scaler.step(opt)
            # 只有当 scaler 没有检测到 Inf/NaN 并成功更新了权重后，才更新 EMA
            scale = scaler.get_scale()
            scaler.update()
            skip_lr_sched = (scale > scaler.get_scale()) # 判断是否发生了 skip
            
            if not skip_lr_sched:
                ema.update(model) # 仅在有效更新后进行 EMA
                
            ema.update(model)
            global_step += 1
            pbar.set_postfix(loss=float(loss.item()), step=global_step)

        # 采样主观图
        if epoch % args.sample_every == 0:
            model.eval()
            # 用 EMA 参数采样更稳
            tmp = UNet(base_ch=args.base_ch).to(device)
            tmp.load_state_dict(model.state_dict())
            ema.copy_to(tmp)

            with torch.no_grad():
                x_gen = ddpm.sample(tmp, n=64, image_size=32, channels=3)
            save_image_grid(x_gen, os.path.join(args.save_dir, "samples", f"epoch_{epoch:04d}.png"), nrow=8)
            del tmp

        # 保存 ckpt
        if epoch % args.save_every == 0:
            ckpt = {
                "model": model.state_dict(),
                "ema": ema.state_dict(),
                "opt": opt.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
                "args": vars(args),
            }
            torch.save(ckpt, os.path.join(args.save_dir, "checkpoints", f"ckpt_epoch_{epoch:04d}.pt"))

    # 最终保存
    ckpt = {
        "model": model.state_dict(),
        "ema": ema.state_dict(),
        "opt": opt.state_dict(),
        "epoch": args.epochs,
        "global_step": global_step,
        "args": vars(args),
    }
    torch.save(ckpt, os.path.join(args.save_dir, "checkpoints", "ckpt_last.pt"))

if __name__ == "__main__":
    main()
