import argparse
import os


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_size", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--total_step", type=int, default=30000)
    parser.add_argument("--save_step", type=int, default=5000)

    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--lr_g", type=float, default=1e-4)
    parser.add_argument("--lr_d", type=float, default=1e-5)
    parser.add_argument("--beta1", type=float, default=0)
    parser.add_argument("--beta2", type=float, default=0.9)

    # Model
    parser.add_argument("--num_channels", type=int, default=3)
    parser.add_argument("--num_features", type=int, default=64)

    # Data
    parser.add_argument("--src_root", type=str, default="./data/CelebA-HQ-img-masked")
    parser.add_argument("--dst_root", type=str, default="./data/cartoon")

    parser.add_argument("--exp_dir", type=str, default="experiments")
    parser.add_argument("--exp_name", type=str, default="name")
    args = parser.parse_args()

    os.makedirs(args.exp_dir, exist_ok=True)
    exp_folder = os.path.join(args.exp_dir, args.exp_name)
    os.makedirs(exp_folder, exist_ok=True)
    os.makedirs(os.path.join(exp_folder, "imgs"), exist_ok=True)
    os.makedirs(os.path.join(exp_folder, "save"), exist_ok=True)

    return args
