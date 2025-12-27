import argparse
import torch_fidelity

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gen_dir", type=str, required=True, help="生成图像目录（png/jpg）")
    p.add_argument("--real_dir", type=str, required=True, help="真实图像目录（png/jpg）")
    p.add_argument("--cuda", action="store_true")
    return p.parse_args()

def main():
    args = get_args()
    metrics = torch_fidelity.calculate_metrics(
        input1=args.gen_dir,
        input2=args.real_dir,
        cuda=args.cuda,
        isc=True,
        fid=True,
        kid=True,
        verbose=False
    )
    print("torch-fidelity metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()
