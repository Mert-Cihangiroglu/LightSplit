#!/usr/bin/env python3
import argparse
import csv
import gzip
import json
import os
import struct
import sys
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

import FSHA


DEFAULT_HPARAMS = {
    "WGAN": True,
    "gradient_penalty": 500.0,
    "style_loss": None,
    "lr_f": 0.00001,
    "lr_tilde": 0.00001,
    "lr_D": 0.0001,
}


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def json_safe(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def dump_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=json_safe)
        handle.write("\n")


def normalize_digits(image, label):
    image = image[:, :, None]
    image = tf.tile(image, (1, 1, 3))
    image = tf.image.resize(image, (32, 32))
    image = tf.cast(image, tf.float32) / (255.0 / 2.0) - 1.0
    image = tf.clip_by_value(image, -1.0, 1.0)
    return image, label


def normalize_cifar(image, label):
    image = tf.cast(image, tf.float32) / (255.0 / 2.0) - 1.0
    image = tf.clip_by_value(image, -1.0, 1.0)
    return image, label


def make_dataset(x, y, parser, seed, shuffle_buffer):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.map(parser, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(shuffle_buffer, seed=seed, reshuffle_each_iteration=True)
    return dataset


def idx_handle(path):
    path = Path(path)
    if path.exists():
        return path.open("rb")
    gz_path = Path(str(path) + ".gz")
    if gz_path.exists():
        return gzip.open(gz_path, "rb")
    raise FileNotFoundError(path)


def read_idx_images(path):
    with idx_handle(path) as handle:
        magic, count, rows, cols = struct.unpack(">IIII", handle.read(16))
        if magic != 2051:
            raise ValueError("Unexpected IDX image magic %s in %s" % (magic, path))
        data = np.frombuffer(handle.read(rows * cols * count), dtype=np.uint8)
    return data.reshape(count, rows, cols)


def read_idx_labels(path):
    with idx_handle(path) as handle:
        magic, count = struct.unpack(">II", handle.read(8))
        if magic != 2049:
            raise ValueError("Unexpected IDX label magic %s in %s" % (magic, path))
        data = np.frombuffer(handle.read(count), dtype=np.uint8)
    return data


def load_fashion_mnist_arrays():
    candidates = [
        Path.home() / ".keras" / "datasets" / "fashion-mnist",
        REPO_DIR / "data" / "FashionMNIST" / "raw",
    ]
    for raw_dir in candidates:
        if (raw_dir / "train-images-idx3-ubyte").exists() or (raw_dir / "train-images-idx3-ubyte.gz").exists():
            x_train = read_idx_images(raw_dir / "train-images-idx3-ubyte")
            y_train = read_idx_labels(raw_dir / "train-labels-idx1-ubyte")
            x_test = read_idx_images(raw_dir / "t10k-images-idx3-ubyte")
            y_test = read_idx_labels(raw_dir / "t10k-labels-idx1-ubyte")
            return (x_train, y_train), (x_test, y_test)
    return tf.keras.datasets.fashion_mnist.load_data()


def load_arrays(dataset_name, examples, seed):
    if dataset_name == "mnist":
        local_mnist = Path.home() / ".keras" / "datasets" / "mnist.npz"
        if local_mnist.exists():
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path=str(local_mnist))
        else:
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        parser = normalize_digits
    elif dataset_name == "fashion_mnist":
        (x_train, y_train), (x_test, y_test) = load_fashion_mnist_arrays()
        parser = normalize_digits
    elif dataset_name == "cifar10":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        y_train = y_train.reshape(-1)
        y_test = y_test.reshape(-1)
        parser = normalize_cifar
    elif dataset_name == "synthetic":
        rng = np.random.default_rng(seed)
        x_train = rng.uniform(-1.0, 1.0, size=(examples, 32, 32, 3)).astype("float32")
        x_test = rng.uniform(-1.0, 1.0, size=(examples, 32, 32, 3)).astype("float32")
        y_train = rng.integers(0, 10, size=(examples,), dtype=np.int32)
        y_test = rng.integers(0, 10, size=(examples,), dtype=np.int32)
        parser = lambda image, label: (image, label)
    else:
        raise ValueError("Unknown dataset: %s" % dataset_name)

    if examples:
        x_train, y_train = x_train[:examples], y_train[:examples]
        x_test, y_test = x_test[:examples], y_test[:examples]
    return x_train.astype("float32"), y_train, x_test.astype("float32"), y_test, parser


def take_images(dataset, count):
    images = []
    for image, _ in dataset.take(count):
        images.append(image.numpy()[None, ...])
    if not images:
        return np.zeros((0, 32, 32, 3), dtype=np.float32)
    return np.concatenate(images, axis=0)


def write_log_csv(path, log):
    columns = ["f_loss", "tilde_f_loss", "D_loss", "validation"]
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns[: log.shape[1]])
        writer.writerows(log.tolist())


def save_reconstruction_png(path, original, recovered, control):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    count = len(original)
    rows = [original, recovered, control]
    row_names = ["private", "recovered", "control"]
    fig, axes = plt.subplots(3, count, figsize=(max(count, 1) * 2, 6))
    if count == 1:
        axes = np.asarray(axes).reshape(3, 1)
    for row_idx, row in enumerate(rows):
        for col_idx in range(count):
            axes[row_idx, col_idx].imshow(np.clip((row[col_idx] + 1.0) / 2.0, 0.0, 1.0))
            axes[row_idx, col_idx].set_xticks([])
            axes[row_idx, col_idx].set_yticks([])
            if col_idx == 0:
                axes[row_idx, col_idx].set_ylabel(row_names[row_idx])
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True


def build_accuracy_classifier(num_classes):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input((32, 32, 3)),
            tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same"),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same"),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(num_classes),
        ],
        name="reconstruction_accuracy_classifier",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return model


def make_batched_dataset(x, y, parser, seed, shuffle_buffer, batch_size):
    return make_dataset(x, y, parser, seed, shuffle_buffer).batch(batch_size).prefetch(tf.data.AUTOTUNE)


def train_accuracy_classifier(args, arrays, parser, root):
    if args.accuracy_examples <= 0:
        return None, {}

    x_train, y_train, x_test, y_test = arrays
    train_limit = args.accuracy_train_examples or len(x_train)
    train_limit = min(train_limit, len(x_train))
    test_limit = args.accuracy_test_examples or min(args.accuracy_examples, len(x_test))
    test_limit = min(test_limit, len(x_test))
    num_classes = int(max(np.max(y_train), np.max(y_test))) + 1

    classifier = build_accuracy_classifier(num_classes)
    train_ds = make_batched_dataset(
        x_train[:train_limit],
        y_train[:train_limit],
        parser,
        args.seed + 101,
        args.shuffle_buffer,
        args.accuracy_batch_size,
    )
    test_ds = make_batched_dataset(
        x_test[:test_limit],
        y_test[:test_limit],
        parser,
        args.seed + 102,
        args.shuffle_buffer,
        args.accuracy_batch_size,
    )

    print("[ACCURACY] training eval classifier examples=%d epochs=%d" % (
        train_limit,
        args.accuracy_classifier_epochs,
    ))
    history = classifier.fit(train_ds, epochs=args.accuracy_classifier_epochs, verbose=2)
    test_loss, test_accuracy = classifier.evaluate(test_ds, verbose=0)
    info = {
        "accuracy_classifier_train_examples": train_limit,
        "accuracy_classifier_test_examples": test_limit,
        "accuracy_classifier_epochs": args.accuracy_classifier_epochs,
        "accuracy_classifier_test_loss": float(test_loss),
        "accuracy_classifier_test_accuracy": float(test_accuracy),
        "accuracy_metric_description": "Label accuracy of original/recovered/control private images under a separately trained eval classifier.",
    }
    info["accuracy_classifier_history"] = {
        key: [float(v) for v in values]
        for key, values in history.history.items()
    }
    dump_json(root / "accuracy_classifier.json", info)
    return classifier, info


def classifier_accuracy(classifier, images, labels):
    logits = classifier(images, training=False)
    pred = tf.argmax(logits, axis=1, output_type=tf.int32)
    labels = tf.reshape(tf.cast(labels, tf.int32), [-1])
    return int(tf.reduce_sum(tf.cast(tf.equal(pred, labels), tf.int32)).numpy())


def evaluate_reconstruction_accuracy(attack, classifier, dataset, examples, batch_size):
    if classifier is None or examples <= 0:
        return {}

    total = 0
    original_correct = 0
    recovered_correct = 0
    control_correct = 0

    for images, labels in dataset.batch(batch_size):
        remaining = examples - total
        if remaining <= 0:
            break
        batch_count = int(tf.shape(images)[0].numpy())
        if batch_count > remaining:
            images = images[:remaining]
            labels = labels[:remaining]
            batch_count = remaining

        recovered, control = attack.attack(images)
        original_correct += classifier_accuracy(classifier, images, labels)
        recovered_correct += classifier_accuracy(classifier, recovered, labels)
        control_correct += classifier_accuracy(classifier, control, labels)
        total += batch_count

    if total == 0:
        return {}
    return {
        "accuracy_examples": total,
        "original_label_accuracy": original_correct / total,
        "recovered_label_accuracy": recovered_correct / total,
        "control_label_accuracy": control_correct / total,
    }


def format_cr(cr):
    if float(cr).is_integer():
        return "cr%02d" % int(cr)
    return "cr%s" % str(cr).replace(".", "p")


def format_wcc(wcc):
    return "wcc%s" % ("%g" % wcc).replace(".", "p")


def build_cases(compression_ratios, wcc_values, raw_per_cr, modes, include_raw=True):
    cases = []
    modes = set(modes)
    if include_raw and raw_per_cr:
        for cr in compression_ratios:
            cases.append({"mode": "none", "compression_ratio": cr, "wcc_lambda": 0.0})
    elif include_raw:
        cases.append({"mode": "none", "compression_ratio": 1.0, "wcc_lambda": 0.0})

    for cr in compression_ratios:
        if "fixed_random" in modes:
            for wcc in wcc_values:
                cases.append({"mode": "fixed_random", "compression_ratio": cr, "wcc_lambda": wcc})
        if "topk_sparse" in modes:
            cases.append({"mode": "topk_sparse", "compression_ratio": cr, "wcc_lambda": 0.0})
        if "learned_1x1" in modes:
            cases.append({"mode": "learned_1x1", "compression_ratio": cr, "wcc_lambda": 0.0})
    return cases


def _dp_suffix(case):
    return "_dp" if case.get("dp") else ""


def case_id(case):
    mode = case["mode"]
    dp = _dp_suffix(case)
    if mode == "none":
        if case["compression_ratio"] == 1.0:
            return "raw_no_defense" + dp
        return "%s_raw_no_defense%s" % (format_cr(case["compression_ratio"]), dp)
    if mode == "fixed_random":
        return "%s_fixed_random_%s%s" % (format_cr(case["compression_ratio"]), format_wcc(case["wcc_lambda"]), dp)
    if mode == "block_random":
        return "%s_block_random_b%d%s" % (format_cr(case["compression_ratio"]), int(case.get("block_size", 2)), dp)
    return "%s_%s%s" % (format_cr(case["compression_ratio"]), mode, dp)


def case_dir(root, case):
    mode = case["mode"]
    leaf_dp = "_dp" if case.get("dp") else ""
    if mode == "none":
        if case["compression_ratio"] == 1.0:
            return root / "raw" / ("no_defense" + leaf_dp)
        return root / format_cr(case["compression_ratio"]) / "raw" / ("no_defense" + leaf_dp)
    if mode == "block_random":
        return root / format_cr(case["compression_ratio"]) / "block_random" / ("b%d%s" % (int(case.get("block_size", 2)), leaf_dp))
    leaf = (format_wcc(case["wcc_lambda"]) if mode == "fixed_random" else "wcc0") + leaf_dp
    return root / format_cr(case["compression_ratio"]) / mode / leaf


def make_hparams(args, case):
    hparams = dict(DEFAULT_HPARAMS)
    hparams.update(
        {
            "WGAN": args.wgan,
            "gradient_penalty": args.gradient_penalty,
            "lr_f": args.lr_f,
            "lr_tilde": args.lr_tilde,
            "lr_D": args.lr_D,
            "bottleneck_mode": case["mode"],
            "compression_ratio": case["compression_ratio"],
            "bottleneck_k": None,
            "wcc_lambda": case["wcc_lambda"],
            "block_size": int(case.get("block_size", 2)),
            "dp_noise_multiplier": float(case.get("dp_noise_multiplier", 0.02)) if case.get("dp") else 0.0,
            "dp_clip": 0.0,
        }
    )
    if hparams["gradient_penalty"] <= 0:
        hparams.pop("gradient_penalty")
    return hparams


def run_one(args, arrays, parser, case, root, accuracy_classifier=None):
    x_train, y_train, x_test, y_test = arrays
    run_dir = case_dir(root, case)
    run_dir.mkdir(parents=True, exist_ok=True)

    hparams = make_hparams(args, case)
    config = {
        "case": case,
        "case_id": case_id(case),
        "dataset": args.dataset,
        "batch_size": args.batch_size,
        "id_setup_zero_based": args.id_setup,
        "id_setup_notebook": args.id_setup + 1,
        "iterations": args.iterations,
        "log_frequency": args.log_frequency,
        "hparams": hparams,
        "seed": args.seed,
    }
    dump_json(run_dir / "config.json", config)

    log_path = run_dir / "run.log"
    started = time.time()
    with log_path.open("w") as log_file:
        tee_out = Tee(sys.stdout, log_file)
        tee_err = Tee(sys.stderr, log_file)
        with redirect_stdout(tee_out), redirect_stderr(tee_err):
            print("[RUN] %s" % case_id(case))
            print("[RUN_DIR] %s" % run_dir)
            xpriv = make_dataset(x_train, y_train, parser, args.seed, args.shuffle_buffer)
            xpub = make_dataset(x_test, y_test, parser, args.seed + 1, args.shuffle_buffer)
            attack = FSHA.FSHA(xpriv, xpub, args.id_setup, args.batch_size, hparams)
            attack.calibrate_dp()  # no-op unless dp_noise_multiplier > 0
            log = attack(
                args.iterations,
                log_frequency=args.log_frequency,
                verbose=True,
                progress_bar=args.progress,
            )
            np.save(run_dir / "train_log.npy", log)
            write_log_csv(run_dir / "train_log.csv", log)

            sample_dataset = make_dataset(x_train, y_train, parser, args.seed + 2, args.shuffle_buffer)
            samples = take_images(sample_dataset, args.sample_count)
            recovered, control = attack.attack(samples)
            np.savez_compressed(
                run_dir / "reconstructions.npz",
                original=samples,
                recovered=recovered,
                control=control,
            )
            wrote_png = save_reconstruction_png(run_dir / "reconstructions.png", samples, recovered, control)

            score = None
            if args.score_examples > 0:
                score_dataset = make_dataset(x_train, y_train, parser, args.seed + 3, args.shuffle_buffer)
                score = float(attack.scoreAttack(score_dataset.take(args.score_examples)))

            accuracy_metrics = {}
            if accuracy_classifier is not None and args.accuracy_examples > 0:
                accuracy_dataset = make_dataset(x_train, y_train, parser, args.seed + 4, args.shuffle_buffer)
                accuracy_metrics = evaluate_reconstruction_accuracy(
                    attack,
                    accuracy_classifier,
                    accuracy_dataset,
                    args.accuracy_examples,
                    args.accuracy_batch_size,
                )

            duration = time.time() - started
            metrics = {
                "case_id": case_id(case),
                "status": "ok",
                "duration_sec": duration,
                "final_f_loss": float(log[-1, 0]) if len(log) else None,
                "final_tilde_f_loss": float(log[-1, 1]) if len(log) else None,
                "final_D_loss": float(log[-1, 2]) if len(log) else None,
                "final_validation": float(log[-1, 3]) if len(log) else None,
                "score_attack": score,
                "wrote_reconstruction_png": wrote_png,
            }
            metrics.update(accuracy_metrics)
            dump_json(run_dir / "metrics.json", metrics)
            print("[DONE] %s duration_sec=%0.2f final_validation=%s recovered_label_accuracy=%s" % (
                case_id(case),
                duration,
                metrics["final_validation"],
                metrics.get("recovered_label_accuracy"),
            ))
            return metrics


def write_summary(path, rows):
    fields = [
        "case_id",
        "status",
        "mode",
        "compression_ratio",
        "wcc_lambda",
        "run_dir",
        "duration_sec",
        "final_f_loss",
        "final_tilde_f_loss",
        "final_D_loss",
        "final_validation",
        "score_attack",
        "accuracy_examples",
        "original_label_accuracy",
        "recovered_label_accuracy",
        "control_label_accuracy",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def parse_args():
    parser = argparse.ArgumentParser(description="Run the default FSHA bottleneck defense grid.")
    parser.add_argument("--dataset", choices=["mnist", "fashion_mnist", "cifar10", "synthetic"], default="mnist")
    parser.add_argument("--output-root", default=str(REPO_DIR / "results" / "bottleneck_grid"))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--log-frequency", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--id-setup", type=int, default=3, help="Zero-based setup index. Notebook setup 4 is index 3.")
    parser.add_argument("--cr", type=float, nargs="+", default=[8.0, 16.0, 32.0])
    parser.add_argument("--wcc", type=float, nargs="+", default=[0.0, 0.01, 0.1])
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["fixed_random", "topk_sparse", "learned_1x1"],
        default=["fixed_random", "learned_1x1"],
    )
    parser.add_argument("--raw-per-cr", action="store_true")
    parser.add_argument("--skip-raw", action="store_true")
    parser.add_argument("--examples", type=int, default=0, help="Limit examples per split. 0 uses the full dataset.")
    parser.add_argument("--sample-count", type=int, default=20)
    parser.add_argument("--score-examples", type=int, default=0)
    parser.add_argument("--accuracy-examples", type=int, default=0)
    parser.add_argument("--accuracy-train-examples", type=int, default=10000)
    parser.add_argument("--accuracy-test-examples", type=int, default=2000)
    parser.add_argument("--accuracy-classifier-epochs", type=int, default=2)
    parser.add_argument("--accuracy-batch-size", type=int, default=128)
    parser.add_argument("--shuffle-buffer", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--wgan", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gradient-penalty", type=float, default=500.0)
    parser.add_argument("--lr-f", type=float, default=0.00001)
    parser.add_argument("--lr-tilde", type=float, default=0.00001)
    parser.add_argument("--lr-D", type=float, default=0.0001)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.log_frequency <= 0 or args.iterations <= 0:
        raise ValueError("iterations and log_frequency must be positive.")
    if args.log_frequency > args.iterations:
        raise ValueError("log_frequency cannot exceed iterations.")
    if args.iterations % args.log_frequency != 0:
        raise ValueError("iterations must be divisible by log_frequency for FSHA log allocation.")
    if args.accuracy_examples > 0 and args.accuracy_classifier_epochs <= 0:
        raise ValueError("accuracy-classifier-epochs must be positive when accuracy-examples is enabled.")

    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S_default_grid")
    root = Path(args.output_root) / run_name
    cases = build_cases(
        args.cr,
        args.wcc,
        args.raw_per_cr,
        args.modes,
        include_raw=not args.skip_raw,
    )

    if args.dry_run:
        print("Output root: %s" % root)
        for case in cases:
            print("%s -> %s" % (case_id(case), case_dir(root, case)))
        return

    root.mkdir(parents=True, exist_ok=True)
    dump_json(
        root / "manifest.json",
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "command": sys.argv,
            "defaults": DEFAULT_HPARAMS,
            "args": vars(args),
            "cases": cases,
        },
    )

    x_train, y_train, x_test, y_test, parser = load_arrays(args.dataset, args.examples, args.seed)
    arrays = (x_train, y_train, x_test, y_test)
    accuracy_classifier, accuracy_info = train_accuracy_classifier(args, arrays, parser, root)
    if accuracy_info:
        manifest_path = root / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["accuracy_classifier"] = accuracy_info
        dump_json(manifest_path, manifest)

    rows = []
    for case in cases:
        row = {
            "case_id": case_id(case),
            "mode": case["mode"],
            "compression_ratio": case["compression_ratio"],
            "wcc_lambda": case["wcc_lambda"],
            "run_dir": str(case_dir(root, case)),
        }
        try:
            row.update(run_one(args, arrays, parser, case, root, accuracy_classifier=accuracy_classifier))
        except Exception as exc:
            row.update({"status": "failed", "error": repr(exc)})
            error_dir = case_dir(root, case)
            error_dir.mkdir(parents=True, exist_ok=True)
            (error_dir / "error.txt").write_text(traceback.format_exc())
            print("[FAILED] %s: %r" % (case_id(case), exc))
            if not args.keep_going:
                rows.append(row)
                write_summary(root / "summary.csv", rows)
                raise
        rows.append(row)
        write_summary(root / "summary.csv", rows)

    dump_json(root / "summary.json", rows)
    write_summary(root / "summary.csv", rows)
    print("[GRID DONE] %s" % root)


if __name__ == "__main__":
    main()
