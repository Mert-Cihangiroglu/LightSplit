#!/usr/bin/env python3
import argparse
import csv
import json
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
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

import architectures
import bottlenecks
from run_bottleneck_grid import (
    Tee,
    build_cases,
    case_dir,
    case_id,
    dump_json,
    load_arrays,
    make_dataset,
)


def build_classifier(input_shape, num_classes, hidden_units):
    xin = tf.keras.layers.Input(input_shape)
    x = tf.keras.layers.Flatten()(xin)
    if hidden_units > 0:
        x = tf.keras.layers.Dense(hidden_units, activation="relu")(x)
    x = tf.keras.layers.Dense(num_classes)(x)
    return tf.keras.Model(xin, x, name="server_classifier")


class CleanSplitModel:
    def __init__(self, input_shape, num_classes, id_setup, case, hidden_units):
        make_f = architectures.SETUPS[id_setup][0]
        self.f = make_f(input_shape)
        self.raw_z_shape = tuple(self.f.output.shape.as_list()[1:])
        self.bottleneck = bottlenecks.make_bottleneck(
            case["mode"],
            self.raw_z_shape,
            compression_ratio=case["compression_ratio"],
            bottleneck_k=None,
            block_size=int(case.get("block_size", 2)),
        )
        self.classifier = build_classifier(
            self.bottleneck.output_shape_,
            num_classes,
            hidden_units,
        )
        # DP on the transmitted payload (clip to median L2 then Gaussian noise), applied
        # during train AND eval so the reported accuracy reflects the noised channel.
        self.dp_noise_multiplier = float(case.get("dp_noise_multiplier", 0.02)) if case.get("dp") else 0.0
        self.dp_clip = tf.Variable(0.0, trainable=False, dtype=tf.float32, name="dp_clip")

    @property
    def trainable_variables(self):
        return (
            self.f.trainable_variables
            + self.bottleneck.trainable_variables
            + self.classifier.trainable_variables
        )

    def _apply_dp(self, payload):
        if self.dp_noise_multiplier <= 0.0:
            return payload
        flat = tf.reshape(payload, [tf.shape(payload)[0], -1])
        norms = tf.norm(flat, axis=1, keepdims=True)
        clip = tf.cast(self.dp_clip, flat.dtype)
        factor = tf.minimum(clip / (norms + 1e-8), 1.0)
        flat = flat * factor + tf.random.normal(tf.shape(flat), dtype=flat.dtype) * (
            tf.cast(self.dp_noise_multiplier, flat.dtype) * clip
        )
        return tf.reshape(flat, tf.shape(payload))

    def calibrate_dp(self, dataset, num_batches=8):
        if self.dp_noise_multiplier <= 0.0:
            return
        norms = []
        for i, (x, _y) in enumerate(dataset):
            if i >= num_batches:
                break
            payload = self.bottleneck(self.f(x, training=False), training=False)
            flat = tf.reshape(payload, [tf.shape(payload)[0], -1])
            norms.append(tf.norm(flat, axis=1).numpy())
        med = float(np.median(np.concatenate(norms)))
        self.dp_clip.assign(med)
        print("[DP] clean clip=%0.4f (median payload L2); noise_std=%0.5f" % (
            med, self.dp_noise_multiplier * med))

    def __call__(self, x, training=False):
        z_raw = self.f(x, training=training)
        z = self.bottleneck(z_raw, training=training)
        z = self._apply_dp(z)
        logits = self.classifier(z, training=training)
        return logits, z


def make_batched_dataset(x, y, parser, seed, shuffle_buffer, batch_size, training):
    ds = make_dataset(x, y, parser, seed, shuffle_buffer)
    ds = ds.batch(batch_size, drop_remainder=False)
    if training:
        ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def evaluate(model, dataset):
    loss_metric = tf.keras.metrics.Mean()
    acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    for x, y in dataset:
        logits, _ = model(x, training=False)
        loss = loss_fn(y, logits)
        loss_metric.update_state(loss)
        acc_metric.update_state(y, logits)

    return float(loss_metric.result().numpy()), float(acc_metric.result().numpy())


def train_one_epoch(model, dataset, optimizer, wcc_lambda):
    total_loss_metric = tf.keras.metrics.Mean()
    task_loss_metric = tf.keras.metrics.Mean()
    wcc_loss_metric = tf.keras.metrics.Mean()
    acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    for x, y in dataset:
        with tf.GradientTape() as tape:
            logits, z = model(x, training=True)
            task_loss = loss_fn(y, logits)
            wcc_loss = tf.constant(0.0, dtype=task_loss.dtype)
            if wcc_lambda > 0.0 and model.bottleneck.supports_wcc:
                wcc_loss = tf.cast(wcc_lambda, task_loss.dtype) * bottlenecks.within_class_compaction_loss(z, y)
            total_loss = task_loss + wcc_loss

        grads = tape.gradient(total_loss, model.trainable_variables)
        grads_and_vars = [
            (grad, var)
            for grad, var in zip(grads, model.trainable_variables)
            if grad is not None
        ]
        optimizer.apply_gradients(grads_and_vars)

        total_loss_metric.update_state(total_loss)
        task_loss_metric.update_state(task_loss)
        wcc_loss_metric.update_state(wcc_loss)
        acc_metric.update_state(y, logits)

    return {
        "train_loss": float(total_loss_metric.result().numpy()),
        "train_task_loss": float(task_loss_metric.result().numpy()),
        "train_wcc_loss": float(wcc_loss_metric.result().numpy()),
        "train_accuracy": float(acc_metric.result().numpy()),
    }


def write_epoch_log(path, rows):
    fields = [
        "epoch",
        "train_loss",
        "train_task_loss",
        "train_wcc_loss",
        "train_accuracy",
        "test_loss",
        "test_accuracy",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def run_one(args, arrays, parser, case, root):
    x_train, y_train, x_test, y_test = arrays
    run_dir = case_dir(root, case)
    run_dir.mkdir(parents=True, exist_ok=True)
    num_classes = int(max(np.max(y_train), np.max(y_test))) + 1
    input_shape = make_dataset(x_train[:1], y_train[:1], parser, args.seed, args.shuffle_buffer).element_spec[0].shape

    config = {
        "case": case,
        "case_id": case_id(case),
        "dataset": args.dataset,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "id_setup_zero_based": args.id_setup,
        "id_setup_notebook": args.id_setup + 1,
        "learning_rate": args.learning_rate,
        "classifier_hidden_units": args.classifier_hidden_units,
        "seed": args.seed,
        "metric_description": "Clean supervised test accuracy of f -> bottleneck -> server classifier.",
    }
    dump_json(run_dir / "config.json", config)

    with (run_dir / "run.log").open("w") as log_file:
        tee_out = Tee(sys.stdout, log_file)
        tee_err = Tee(sys.stderr, log_file)
        with redirect_stdout(tee_out), redirect_stderr(tee_err):
            started = time.time()
            print("[RUN] %s" % case_id(case))
            print("[RUN_DIR] %s" % run_dir)

            train_ds = make_batched_dataset(
                x_train,
                y_train,
                parser,
                args.seed,
                args.shuffle_buffer,
                args.batch_size,
                training=True,
            )
            test_ds = make_batched_dataset(
                x_test,
                y_test,
                parser,
                args.seed + 1,
                args.shuffle_buffer,
                args.batch_size,
                training=False,
            )

            model = CleanSplitModel(
                input_shape,
                num_classes,
                args.id_setup,
                case,
                args.classifier_hidden_units,
            )
            dummy = tf.zeros([1] + [int(v) for v in input_shape], dtype=tf.float32)
            model(dummy, training=False)
            model.calibrate_dp(train_ds)  # no-op unless dp is set on the case
            optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)

            print("[BOTTLENECK] mode=%s raw_shape=%s defended_shape=%s compression_ratio=%0.4f k=%s wcc_lambda=%0.4f" % (
                case["mode"],
                model.raw_z_shape,
                model.bottleneck.output_shape_,
                model.bottleneck.effective_compression_ratio,
                model.bottleneck.k,
                case["wcc_lambda"],
            ))

            epoch_rows = []
            best_test_accuracy = 0.0
            for epoch in range(1, args.epochs + 1):
                train_metrics = train_one_epoch(
                    model,
                    train_ds,
                    optimizer,
                    case["wcc_lambda"] if case["mode"] == "fixed_random" else 0.0,
                )
                test_loss, test_accuracy = evaluate(model, test_ds)
                best_test_accuracy = max(best_test_accuracy, test_accuracy)
                row = {
                    "epoch": epoch,
                    "test_loss": test_loss,
                    "test_accuracy": test_accuracy,
                }
                row.update(train_metrics)
                epoch_rows.append(row)
                print(
                    "epoch=%03d train_acc=%0.4f test_acc=%0.4f train_loss=%0.4f test_loss=%0.4f"
                    % (
                        epoch,
                        row["train_accuracy"],
                        row["test_accuracy"],
                        row["train_loss"],
                        row["test_loss"],
                    )
                )
                write_epoch_log(run_dir / "epoch_log.csv", epoch_rows)

            final = epoch_rows[-1]
            metrics = {
                "case_id": case_id(case),
                "status": "ok",
                "duration_sec": time.time() - started,
                "final_train_accuracy": final["train_accuracy"],
                "clean_test_accuracy": final["test_accuracy"],
                "best_clean_test_accuracy": best_test_accuracy,
                "final_train_loss": final["train_loss"],
                "final_test_loss": final["test_loss"],
            }
            dump_json(run_dir / "metrics.json", metrics)
            print("[DONE] %s clean_test_accuracy=%0.4f best_clean_test_accuracy=%0.4f" % (
                case_id(case),
                metrics["clean_test_accuracy"],
                metrics["best_clean_test_accuracy"],
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
        "final_train_accuracy",
        "clean_test_accuracy",
        "best_clean_test_accuracy",
        "final_train_loss",
        "final_test_loss",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def parse_args():
    parser = argparse.ArgumentParser(description="Run clean classification accuracy for FSHA bottleneck defenses.")
    parser.add_argument("--dataset", choices=["mnist", "fashion_mnist", "cifar10", "synthetic"], default="mnist")
    parser.add_argument("--output-root", default=str(REPO_DIR / "results" / "clean_accuracy_grid"))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--id-setup", type=int, default=3, help="Zero-based setup index. Notebook setup 4 is index 3.")
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--classifier-hidden-units", type=int, default=256)
    parser.add_argument("--cr", type=float, nargs="+", default=[8.0, 16.0, 32.0])
    parser.add_argument("--wcc", type=float, nargs="+", default=[0.0, 0.01, 0.1, 0.3, 1.0, 3.0])
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["fixed_random", "topk_sparse", "learned_1x1"],
        default=["fixed_random", "learned_1x1"],
    )
    parser.add_argument("--raw-per-cr", action="store_true")
    parser.add_argument("--skip-raw", action="store_true")
    parser.add_argument("--examples", type=int, default=0, help="Limit examples per split. 0 uses the full dataset.")
    parser.add_argument("--shuffle-buffer", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.epochs <= 0:
        raise ValueError("epochs must be positive.")

    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S_clean_accuracy")
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
            "args": vars(args),
            "cases": cases,
        },
    )

    x_train, y_train, x_test, y_test, parser = load_arrays(args.dataset, args.examples, args.seed)
    arrays = (x_train, y_train, x_test, y_test)

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
            row.update(run_one(args, arrays, parser, case, root))
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
