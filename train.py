import os

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.autograd import grad
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms
from torchvision.utils import save_image
from tqdm import trange

from configs import get_args
from dataset import Dataset
from models import Decoder, Discriminator, Encoder


def main():
    args = get_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset
    transform = transforms.Compose(
        [
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    src_dataset = Dataset(args.src_root, transform=transform)
    dst_dataset = Dataset(args.dst_root, transform=transform)

    src_loader = DataLoader(
        src_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )
    dst_loader = DataLoader(
        dst_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )

    # Model
    encoder = Encoder(args.num_channels, args.num_features).to(device)
    decoder = Decoder(args.num_channels, args.num_features).to(device)
    discriminator = Discriminator(args.num_channels, args.num_features).to(device)

    # criterion = nn.MSELoss()
    criterion = nn.L1Loss()

    update_params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = optim.Adam(update_params, args.lr_g, betas=(args.beta1, args.beta2))
    optimizer_D = optim.Adam(discriminator.parameters(), args.lr_d, betas=(args.beta1, args.beta2))

    exp_dir_path = os.path.join(args.exp_dir, args.exp_name)
    writer = SummaryWriter(log_dir=exp_dir_path)

    pbar = trange(1, args.total_step + 1)
    for step in pbar:
        encoder.train()
        decoder.train()
        if (step - 1) % len(src_loader) == 0:
            src_iter = iter(src_loader)
        if (step - 1) % len(dst_loader) == 0:
            dst_iter = iter(dst_loader)

        src_data = next(src_iter).to(device)
        dst_data = next(dst_iter).to(device)

        optimizer.zero_grad()
        # Src
        src_latent = encoder(src_data, mode="AA")
        pred_src = decoder(src_latent)
        loss_src = criterion(pred_src, src_data)

        dst_latent = encoder(dst_data, mode="AB")
        pred_dst = decoder(dst_latent)
        loss_dst = criterion(pred_dst, dst_data)

        latent = encoder(src_data, mode="AB")
        pred_img = decoder(latent)
        fake_logit = discriminator(pred_img)

        loss_G = F.softplus(-fake_logit).mean()

        loss = loss_src + loss_dst + 1e-3 * loss_G
        loss.backward()
        optimizer.step()

        optimizer_D.zero_grad()

        dst_data.requires_grad_(True)
        fake_logit = discriminator(pred_img.detach())
        real_logit = discriminator(dst_data)

        loss_D = F.softplus(fake_logit).mean() + F.softplus(-real_logit).mean()

        real_grads = grad(
            outputs=real_logit,
            inputs=dst_data,
            grad_outputs=torch.ones(real_logit.size()).to(dst_data.device),
            create_graph=True,
            retain_graph=True,
        )[0].view(dst_data.size(0), -1)
        r1_penalty = torch.mul(real_grads, real_grads).mean()

        loss_D = loss_D + 10 * r1_penalty
        loss_D.backward()

        optimizer_D.step()

        writer.add_scalar("loss/total", loss.item(), global_step=step)
        writer.add_scalar("loss/src", loss_src.item(), global_step=step)
        writer.add_scalar("loss/dst", loss_dst.item(), global_step=step)
        writer.add_scalar("loss/G", loss_G.item(), global_step=step)
        writer.add_scalar("loss/D", loss_D.item(), global_step=step)
        pbar.set_description(
            f"{step}/{args.total_step},"
            f"loss[total/src/dst/G/D]={loss.item():.4f}/{loss_src.item():.4f}/{loss_dst.item():.4f}/{loss_G.item():.4f}/{loss_D.item():.4f}"
            # f"loss[total/src/dst]={loss.item():.4f}/{loss_src.item():.4f}/{loss_dst.item():.4f}"
        )

        if step % args.save_step == 0:
            encoder.eval()
            decoder.eval()
            with torch.no_grad():
                latent = encoder(src_data, mode="AB")
                pred_img = decoder(latent)

                save_image(
                    torch.cat(
                        [(src_data + 1.0) * 0.5, (pred_img + 1.0) * 0.5, (pred_dst + 1.0) * 0.5],
                        dim=0,
                    ),
                    os.path.join(exp_dir_path, "imgs", f"{step}.png"),
                    nrow=args.batch_size,
                )

            ckpt = {
                "encoder": encoder.state_dict(),
                "decoder": decoder.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            torch.save(ckpt, os.path.join(exp_dir_path, "save", f"{step}.ckpt"))

    ckpt = {
        "encoder": encoder.state_dict(),
        "decoder": decoder.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(ckpt, os.path.join(exp_dir_path, "save", "final.ckpt"))


if __name__ == "__main__":
    main()
