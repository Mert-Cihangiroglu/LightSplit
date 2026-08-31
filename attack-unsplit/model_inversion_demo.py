import argparse
import csv
from pathlib import Path

import torch
from torchvision import transforms, datasets
from torchvision.utils import save_image

import unsplit.attacks as unsplit
from unsplit.models import MnistNet, CifarNet
from unsplit.util import TV, l2loss, get_test_score, get_examples_by_class, normalize
from unsplit.defended_models import DefendedSplitModel
from unsplit.defended_clone import DefendedClone, build_target_mask


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("dataset", choices=["mnist", "f_mnist", "cifar"])
    p.add_argument("split_layer", type=int)
    p.add_argument("num_images", type=int, nargs="?", default=10)
    p.add_argument("--defense", choices=["none", "raw", "learned_1x1", "topk_sparse",
                                         "fixed_random", "block_random"], default="none")
    p.add_argument("--k", type=int, default=None,
                   help="bottleneck width in scalars/sample. Defaults to activation_dim/cr.")
    p.add_argument("--cr", type=int, default=8)
    p.add_argument("--block-size", type=int, default=2,
                   help="block_random tile size (privacy/utility dial): 1=per-position "
                        "(max utility), spatial_hw=global dense R==fixed_random (max privacy)")
    p.add_argument("--clip-norm", type=float, default=0.0,
                   help="DP: per-sample L2 clip norm on the transmitted payload (0 = DP off). "
                        "unsplit payloads are O(1-3); ~2.0 is a sane default.")
    p.add_argument("--noise-multiplier", type=float, default=0.0,
                   help="DP: Gaussian noise multiplier; std = noise_multiplier * clip_norm "
                        "(requires --clip-norm > 0). Operating point: 0.02.")
    p.add_argument("--wcc-lambda", type=float, default=0.0)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--main-iters", type=int, default=1000)
    p.add_argument("--input-iters", type=int, default=100)
    p.add_argument("--model-iters", type=int, default=100)
    p.add_argument("--max-train-batches", type=int, default=0,
                   help="limit training to N batches per epoch (0 = full epoch). For smoke tests.")
    p.add_argument("--seed", type=int, default=None,
                   help="If set, calls torch.manual_seed(seed) at the very start "
                        "so the random projection R, model init, attack x_pred init, "
                        "and dataloader shuffling are reproducible.")
    p.add_argument("--device", default=None,
                   help="torch device (e.g. 'cuda', 'cuda:0', 'cpu'). "
                        "Default: 'cuda' if available, else 'cpu'.")
    return p.parse_args()


def detect_split_shape(model_class, split_layer, input_chw):
    m = model_class()
    m.eval()
    with torch.no_grad():
        out = m(torch.zeros(1, *input_chw), end=split_layer)
    return tuple(out.shape[1:])  # (C, H, W)


def wcc_loss(z, y):
    """Within-class compaction over a batch of bottleneck features. Matches
    utility/train_split.py."""
    loss = torch.zeros((), device=z.device, dtype=z.dtype)
    for c in y.unique():
        mask = (y == c)
        if mask.sum() > 1:
            z_c = z[mask]
            mu_c = z_c.mean(dim=0, keepdim=True)
            loss = loss + (z_c - mu_c).pow(2).sum(dim=1).mean()
    return loss


def model_inversion_masked(clone_model, split_layer, target, input_size, target_mask=None,
                            lambda_tv=0.1, lambda_l2=1.0,
                            main_iters=1000, input_iters=100, model_iters=100,
                            device="cpu"):
    """Adapted from `unsplit.attacks.model_inversion_stealing` with optional target_mask
    so zeroed-out columns don't poison the MSE."""
    x_pred = torch.empty(input_size, device=device).fill_(0.5).requires_grad_(True)
    input_opt = torch.optim.Adam([x_pred], lr=0.001, amsgrad=True)
    model_opt = torch.optim.Adam(clone_model.parameters(), lr=0.001, amsgrad=True)
    mse = torch.nn.MSELoss()

    def loss_term(pred):
        if target_mask is not None:
            return mse(pred * target_mask, target * target_mask)
        return mse(pred, target)

    for _ in range(main_iters):
        for _ in range(input_iters):
            input_opt.zero_grad()
            pred = clone_model(x_pred, end=split_layer)
            loss = loss_term(pred) + lambda_tv * TV(x_pred) + lambda_l2 * l2loss(x_pred)
            loss.backward(retain_graph=True)
            input_opt.step()
        for _ in range(model_iters):
            model_opt.zero_grad()
            pred = clone_model(x_pred, end=split_layer)
            loss = loss_term(pred)
            loss.backward(retain_graph=True)
            model_opt.step()
    return x_pred.detach()


def get_test_score_defended(model: DefendedSplitModel, dataset, count=2000, device="cpu"):
    score = 0
    # Use a larger batch on GPU for speed; the random shuffle still ensures
    # we sample uniformly from the test set and stop after `count` examples.
    bs = 256 if device != "cpu" else 1
    loader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=True)
    seen = 0
    for images, labels in loader:
        if seen >= count:
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.no_grad():
            logits, _ = model.full_forward(images)
        score += int((torch.argmax(logits, dim=-1) == labels).sum().item())
        seen += images.size(0)
    return 100 * score / max(seen, 1)


CANONICAL_FIELDS = ["dataset", "defense", "k", "split", "wcc_lambda", "block_size",
                    "clip_norm", "noise_multiplier", "seed",
                    "idx", "mse", "ssim", "psnr", "lpips",
                    "mse_fg", "ssim_fg", "psnr_fg", "lpips_fg", "acc"]


def append_csv_row(path: Path, row: dict):
    """Append a row using the canonical 16-field schema, even when the caller
    only supplies a subset (foreground metrics get filled in later by
    the summary augmentation pass). Missing fields are written as ''
    so DictReader parses subsequent runs without column-misalignment."""
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    full = {f: row.get(f, "") for f in CANONICAL_FIELDS}
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CANONICAL_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(full)


def main():
    args = parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")
    Path("results").mkdir(parents=True, exist_ok=True)

    if args.dataset == "mnist":
        trainset = datasets.MNIST("data/mnist", download=True, train=True, transform=transforms.ToTensor())
        testset = datasets.MNIST("data/mnist", download=True, train=False, transform=transforms.ToTensor())
        model_class, input_chw = MnistNet, (1, 28, 28)
    elif args.dataset == "f_mnist":
        trainset = datasets.FashionMNIST("data/f_mnist", download=True, train=True, transform=transforms.ToTensor())
        testset = datasets.FashionMNIST("data/f_mnist", download=True, train=False, transform=transforms.ToTensor())
        model_class, input_chw = MnistNet, (1, 28, 28)
    elif args.dataset == "cifar":
        trainset = datasets.CIFAR10("data/cifar", download=True, train=True, transform=transforms.ToTensor())
        testset = datasets.CIFAR10("data/cifar", download=True, train=False, transform=transforms.ToTensor())
        model_class, input_chw = CifarNet, (3, 32, 32)

    trainloader = torch.utils.data.DataLoader(trainset, shuffle=True, batch_size=64)

    # ---- LEGACY (no defense) PATH: byte-identical training to the original demo ----
    if args.defense == "none":
        client, server, clone = model_class(), model_class(), model_class()
        client_opt = torch.optim.Adam(client.parameters(), lr=0.001, amsgrad=True)
        server_opt = torch.optim.Adam(server.parameters(), lr=0.001, amsgrad=True)
        criterion = torch.nn.CrossEntropyLoss()

        print("Training models...")
        for epoch in range(args.epochs):
            running_loss = 0
            for batch_idx, (images, labels) in enumerate(trainloader):
                if args.max_train_batches and batch_idx >= args.max_train_batches:
                    break
                client_opt.zero_grad(); server_opt.zero_grad()
                pred = server(client(images, end=args.split_layer), start=args.split_layer + 1)
                loss = criterion(pred, labels)
                loss.backward()
                running_loss += loss
                server_opt.step(); client_opt.step()
            final_acc = get_test_score(client, server, testset, split=args.split_layer)
            print(f"Epoch: {epoch} Loss: {running_loss / len(trainloader)} "
                  f"Acc: {final_acc}")
        print("Done.")

        print("Starting model inversion & stealing attack...")
        inversion_targets = [get_examples_by_class(testset, c, count=1) for c in range(args.num_images)]
        targetloader = torch.utils.data.DataLoader(inversion_targets, shuffle=False)
        mse = torch.nn.MSELoss()
        losses = []
        for idx, target in enumerate(targetloader):
            client_out = client(target, end=args.split_layer)
            result = unsplit.model_inversion_stealing(
                clone, args.split_layer, client_out, target.size(),
                main_iters=args.main_iters, input_iters=args.input_iters, model_iters=args.model_iters)
            if args.dataset == "cifar":
                result = normalize(result)
            loss = mse(result, target)
            losses.append(loss)
            save_image(result, f"results/{args.dataset}_split{args.split_layer}_{idx}.png")
            print(f"\tImage {idx} loss: {loss}")
            append_csv_row(Path("results/summary.csv"), {
                "dataset": args.dataset, "defense": "none", "k": "",
                "split": args.split_layer, "wcc_lambda": 0.0,
                "seed": (args.seed if args.seed is not None else ""),
                "idx": idx, "mse": float(loss),
                "acc": f"{final_acc:.2f}",
            })
        print(f"Average MSE: {sum(losses) / len(losses)}")
        print(f"Clone test score: {get_test_score(client, clone, testset, split=args.split_layer)}%")
        print("Results saved to the results directory.")
        return

    # ---- DEFENDED PATH ----
    in_channels, H, W = detect_split_shape(model_class, args.split_layer, input_chw)
    activation_dim = in_channels * H * W
    k = args.k if args.k is not None else activation_dim // args.cr
    print(f"Activation shape at split={args.split_layer}: ({in_channels},{H},{W}) "
          f"= {activation_dim} dims; k={k} (CR={activation_dim/k:.2f})")
    if args.defense == "topk_sparse":
        # Honest accounting: a kept coord costs 4B value + 2B index; report the
        # byte-matched k_s and the effective CR in float32-equivalents.
        k_s = max(1, (4 * k) // 6)
        payload_floats = (k_s * 6 + 3) // 4
        print(f"  topk_sparse: keeping k_s={k_s} coords/sample "
              f"(byte-equiv {payload_floats} floats, effective CR={activation_dim/payload_floats:.2f})")
    if args.defense == "block_random":
        if H % args.block_size or W % args.block_size:
            print(f"ERROR: --block-size {args.block_size} does not divide the split "
                  f"activation map ({in_channels},{H},{W}). Pick a divisor of both {H} and {W}.")
            return
        # Honest accounting: out_per_block floor-divides the float budget across tiles, so
        # at block=1 / high CR the payload can exceed k (clamped to >=1 float/tile).
        n_blocks = (H // args.block_size) * (W // args.block_size)
        out_per_block = max(1, k // n_blocks)
        payload_floats = n_blocks * out_per_block
        print(f"  block_random: block={args.block_size} -> {n_blocks} tiles x "
              f"{out_per_block} floats = {payload_floats} floats "
              f"(effective CR={activation_dim/payload_floats:.2f})")

    if args.clip_norm > 0:
        print(f"  DP: per-sample L2 clip={args.clip_norm} + noise std={args.noise_multiplier * args.clip_norm:g} "
              f"(noise_multiplier={args.noise_multiplier}) on the transmitted payload")

    victim = DefendedSplitModel(model_class, split_layer=args.split_layer, defense=args.defense,
                                k=k, in_channels=in_channels, spatial_hw=(H, W),
                                block_size=args.block_size,
                                clip_norm=args.clip_norm,
                                noise_multiplier=args.noise_multiplier).to(device)
    opt = torch.optim.Adam(victim.parameters(), lr=0.001, amsgrad=True)
    criterion = torch.nn.CrossEntropyLoss()

    print(f"Training defended split model (defense={args.defense}, wcc_lambda={args.wcc_lambda})...")
    for epoch in range(args.epochs):
        running_loss = 0.0
        running_wcc = 0.0
        n_batches = 0
        for batch_idx, (images, labels) in enumerate(trainloader):
            if args.max_train_batches and batch_idx >= args.max_train_batches:
                break
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            opt.zero_grad()
            logits, info = victim.full_forward(images)
            loss = criterion(logits, labels)
            if args.wcc_lambda > 0 and "bottleneck_z" in info:
                wcc = wcc_loss(info["bottleneck_z"], labels)
                loss = loss + args.wcc_lambda * wcc
                running_wcc += float(wcc)
            running_loss += float(loss)
            loss.backward()
            opt.step()
            n_batches += 1
        final_acc = get_test_score_defended(victim, testset, device=device)
        msg = f"Epoch: {epoch} Loss: {running_loss / n_batches:.4f}"
        if args.wcc_lambda > 0:
            msg += f" WCC: {running_wcc / n_batches:.4f}"
        msg += f" Acc: {final_acc}"
        print(msg)
    print("Done.")

    # ---- ATTACK ----
    print("Starting model inversion attack on defended model...")
    clone = DefendedClone(model_class, split_layer=args.split_layer, defense=args.defense,
                          k=k, in_channels=in_channels, spatial_hw=(H, W),
                          block_size=args.block_size).to(device)
    mse_fn = torch.nn.MSELoss()

    targets = [get_examples_by_class(testset, c, count=1) for c in range(args.num_images)]
    losses = []
    for idx, target in enumerate(targets):
        target_t = target.unsqueeze(0).to(device)  # (1, C, H, W)
        with torch.no_grad():
            payload = victim.compress_payload(target_t)
        # topk_sparse zeros all but the top-k_s coords, so the inversion MSE must be
        # scored only on the observed (nonzero) columns.
        target_mask = build_target_mask(payload) if args.defense == "topk_sparse" else None
        if target_mask is not None:
            assert target_mask.sum() < target_mask.numel(), \
                f"target_mask should mask some columns but sums to {target_mask.sum()}/{target_mask.numel()}"

        result = model_inversion_masked(
            clone, args.split_layer, payload, target_t.size(),
            target_mask=target_mask, device=device,
            main_iters=args.main_iters, input_iters=args.input_iters, model_iters=args.model_iters)
        if args.dataset == "cifar":
            result = normalize(result)
        loss = mse_fn(result, target_t)
        losses.append(float(loss))
        # Encode split, k, block (for block_random), wcc (for fixed_random/block_random,
        # which sweep it) and seed in the filename so multiple split / CR / block / WCC /
        # seed values for the same defense don't overwrite each other.
        split_suffix = f"_split{args.split_layer}"
        k_suffix     = f"_k{k}" if args.defense != "raw" else ""   # raw has no compression budget
        block_suffix = f"_block{args.block_size}" if args.defense == "block_random" else ""
        wcc_suffix   = (f"_wcc{args.wcc_lambda:g}"
                        if args.defense in ("fixed_random", "block_random") else "")
        dp_suffix    = f"_dp{args.noise_multiplier:g}" if args.clip_norm > 0 else ""
        seed_suffix  = f"_seed{args.seed}" if args.seed is not None else ""
        stem = (f"results/{args.dataset}{split_suffix}_{args.defense}"
                f"{k_suffix}{block_suffix}{wcc_suffix}{dp_suffix}{seed_suffix}_{idx}")
        save_image(result, f"{stem}.png")
        save_image(target_t, f"{stem}_target.png")
        print(f"\tImage {idx} MSE: {loss}")
        append_csv_row(Path("results/summary.csv"), {
            "dataset": args.dataset, "defense": args.defense, "k": k,
            "split": args.split_layer, "wcc_lambda": args.wcc_lambda,
            "block_size": (args.block_size if args.defense == "block_random" else ""),
            "clip_norm": (args.clip_norm if args.clip_norm > 0 else ""),
            "noise_multiplier": (args.noise_multiplier if args.clip_norm > 0 else ""),
            "seed": (args.seed if args.seed is not None else ""), "idx": idx,
            "mse": float(loss),
            "acc": f"{final_acc:.2f}",
        })
    print(f"Average MSE: {sum(losses) / len(losses)}")
    print("Results saved to the results directory.")


if __name__ == "__main__":
    main()
