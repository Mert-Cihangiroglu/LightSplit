import argparse
import os
import pickle
import sys

import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw
from sklearn.model_selection import train_test_split

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.append("../src")
import data  # noqa: E402


def _read_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f, encoding="bytes")


def _load_cifar10_from_local_copy():
    candidates = [
        os.path.expanduser("~/.keras/datasets/cifar-10-batches-py"),
        "/tmp/fora_pristine/FORA/data/cifar-10-batches-py",
        "/tmp/unsplit_pristine_cpu_run/data/cifar/cifar-10-batches-py",
    ]
    data_dir = next((path for path in candidates if os.path.isdir(path)), None)
    if data_dir is None:
        raise FileNotFoundError("Could not find a local cifar-10-batches-py directory.")

    x_train_parts = []
    y_train_parts = []
    for i in range(1, 6):
        batch = _read_pickle(os.path.join(data_dir, f"data_batch_{i}"))
        x_train_parts.append(batch[b"data"])
        y_train_parts.extend(batch[b"labels"])

    test_batch = _read_pickle(os.path.join(data_dir, "test_batch"))
    x_train = np.concatenate(x_train_parts, axis=0)
    y_train = np.asarray(y_train_parts, dtype=np.int64).reshape(-1, 1)
    x_test = test_batch[b"data"]
    y_test = np.asarray(test_batch[b"labels"], dtype=np.int64).reshape(-1, 1)

    x = np.concatenate((x_train, x_test)).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    y = np.concatenate((y_train, y_test))
    return x.astype(np.float32), y


def _load_original_examples(dataset, batch_size, start, count):
    if dataset == "cifar10":
        try:
            eval_batch = data.load_single_batch(dataset, batch_size=batch_size)
        except Exception:
            x, y = _load_cifar10_from_local_copy()
            x_client, _, y_client, _ = train_test_split(x, y, train_size=0.5, random_state=42)
            client_ds = data.make_dataset(
                x_client,
                y_client,
                lambda image: tf.clip_by_value(image / 255.0, 0.0, 1.0),
                seed=420,
            )
            eval_batch = client_ds.batch(batch_size).take(1)
    else:
        eval_batch = data.load_single_batch(dataset, batch_size=batch_size)
    x, _ = [(x, y) for (x, y) in iter(eval_batch)][0]
    x = x.numpy()[start:start + count]
    return np.clip(x * 255.0, 0, 255).astype(np.uint8)


def _image_from_array(arr, size):
    return Image.fromarray(arr).resize((size, size), Image.Resampling.NEAREST)


def make_comparison(dataset, recon_fig, output, originals_output, batch_size, start, count):
    originals = _load_original_examples(dataset, batch_size, start, count)
    recon = Image.open(recon_fig).convert("RGB")

    cell = recon.height
    label_width = 110
    gap = 12
    canvas = Image.new("RGB", (label_width + count * cell, 2 * cell + gap), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((10, cell // 2 - 6), "Original", fill=(0, 0, 0))
    draw.text((10, cell + gap + cell // 2 - 6), "Recon", fill=(0, 0, 0))

    original_row = Image.new("RGB", (count * cell, cell), "white")
    for i, img in enumerate(originals):
        original_img = _image_from_array(img, cell)
        x0 = i * cell
        original_row.paste(original_img, (x0, 0))
        canvas.paste(original_img, (label_width + x0, 0))

        rx0 = round(i * recon.width / count)
        rx1 = round((i + 1) * recon.width / count)
        recon_crop = recon.crop((rx0, 0, rx1, recon.height)).resize(
            (cell, cell),
            Image.Resampling.BILINEAR,
        )
        canvas.paste(recon_crop, (label_width + x0, cell + gap))

    os.makedirs(os.path.dirname(output), exist_ok=True)
    canvas.save(output)
    if originals_output:
        original_row.save(originals_output)


def main():
    parser = argparse.ArgumentParser(description="Build original-vs-reconstruction visual comparison from an SDAR recon PNG.")
    parser.add_argument("--dataset", default="cifar10", choices=["cifar10", "cifar100", "tinyimagenet", "stl10"])
    parser.add_argument("--recon_fig", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--originals_output", default="")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--start", type=int, default=10)
    parser.add_argument("--count", type=int, default=20)
    args = parser.parse_args()

    make_comparison(
        dataset=args.dataset,
        recon_fig=args.recon_fig,
        output=args.output,
        originals_output=args.originals_output,
        batch_size=args.batch_size,
        start=args.start,
        count=args.count,
    )
    print(f"Wrote {args.output}")
    if args.originals_output:
        print(f"Wrote {args.originals_output}")


if __name__ == "__main__":
    main()
