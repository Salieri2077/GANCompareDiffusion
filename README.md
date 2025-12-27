# CIFAR-10 彩色图像生成（DDPM / Diffusion）

## 安装
pip install -r requirements.txt

## 训练
python train.py --data_dir ./CIFARdata --save_dir ./runs/ddpm_cifar10 --amp

训练过程中会在 runs/ddpm_cifar10/samples 里定期输出 epoch_xxxx.png 供主观观察。

## 生成图像（用于 IS/FID/KID）
python sample.py --ckpt ./runs/ddpm_cifar10/checkpoints/ckpt_last.pt \
  --out_dir ./generated/ddpm_cifar10 --num_images 5000 --batch_size 100 --use_ema

python sample.py --ckpt ./runs/ddpm_cifar10/checkpoints/ckpt_epoch_0200.pt \
  --out_dir ./generated/ddpm_cifar10 --num_images 5000 --batch_size 100

python sample.py \
  --ckpt ./runs/ddpm_cifar10/checkpoints/ckpt_last.pt \
  --out_dir ./generated/ddpm_cifar10 \
  --num_images 5000 \
  --batch_size 100 \
  --use_ema \
  --sampler ddpm
## 导出真实图像（测试集）
python export_real_images.py --data_dir ./CIFARdata --out_dir ./real_images/cifar10_test --split test --num_images 10000

## 评估（torch-fidelity）
python eval_fidelity.py --gen_dir ./generated/ddpm_cifar10 --real_dir ./real_images/cifar10_test --cuda


## 假设你已训练好并有 ckpt_last.pt：
## 导出真实 test 集（如果还没导出）
##python export_real_images.py --data_dir ./CIFARdata --out_dir ./real_images/cifar10_test --split test --num_images 10000


## 用 DDIM 快速生成一批图用于指标与失败案例
##python sample.py --ckpt ./runs/ddpm_cifar10/checkpoints/ckpt_last.pt \
  --out_dir ./generated/ddpm_cifar10 --num_images 5000 --batch_size 100 --use_ema \
  --sampler ddim --steps 50 --eta 0.0


## 客观指标（IS/FID/KID）
##ython eval_fidelity.py --gen_dir ./generated/ddpm_cifar10 --real_dir ./real_images/cifar10_test --cuda


## 失败案例与直方图分析
python tools/failure_cases.py --gen_dir ./generated/ddpm_cifar10 --out_dir ./runs/ddpm_cifar10/failures --topk 64


## 主观对比图
python tools/compare_real_vs_gen.py --real_dir ./real_images/cifar10_test --gen_dir ./generated/ddpm_cifar10 \
  --out_png ./runs/ddpm_cifar10/real_vs_gen.png --pairs 40


## 训练过程演化 GIF（如果你有 samples/epoch_*.png）
python tools/make_progress_gif.py --samples_dir ./runs/ddpm_cifar10/samples --out_gif ./runs/ddpm_cifar10/progress.gif --every 2


## 如果你保存了多轮 checkpoint：画 IS/FID/KID 随 epoch 曲线（高分点）
python -m tools.evaluate_checkpoints \
  --ckpt_dir ./runs/ddpm_cifar10/checkpoints \
  --real_dir ./real_images/cifar10_test \
  --out_dir ./runs/ddpm_cifar10/eval_test \
  --num_images 5000 --batch_size 100 \
  --use_ema --sampler ddim --steps 50 --eta 0.0 --cuda

python -m tools.evaluate_checkpoints \
  --ckpt_dir ./runs/ddpm_cifar10/checkpoints \
  --real_dir ./real_images/cifar10_test \
  --out_dir ./runs/ddpm_cifar10/eval_test \
  --num_images 5000 --batch_size 100 \
  --use_ema --sampler ddpm --steps 50 --eta 0.0 --cuda