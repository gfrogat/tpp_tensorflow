from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import numpy as np
import pandas as pd
import json
from sklearn.metrics import roc_auc_score
from tensorflow.python.client import device_lib


def compute_mean_auc(labels, predictions, mask, label_mask_fold):
    num_labels = len(label_mask_fold)
    auc_vec = np.empty(num_labels, dtype=np.float32)

    for ivec, idx in enumerate(label_mask_fold):
        mask_ids = mask[:, idx].astype(np.bool)
        masked_labels = labels[mask_ids, idx]
        masked_predictions = predictions[mask_ids, idx]

        if np.unique(masked_labels).size == 2:
            auc_vec[ivec] = roc_auc_score(masked_labels, masked_predictions)
        else:
            auc_vec[ivec] = np.nan

    mean_auc = np.nanmean(auc_vec)

    return mean_auc, auc_vec


def compute_mean_auc_fold(models, ds, eval_fn, params, label_mask):
    loss = tf.keras.metrics.Mean(name="test_loss")
    accuracy = tf.keras.metrics.Accuracy(name="test_accuracy")
    metrics = (loss, accuracy)

    # Concatenate all features
    y_true = []
    y_pred = []
    mask = []
    for features, labels in ds:
        pred = eval_fn(features, labels, models, metrics, params, output=True)
        y_true.append(pred["labels"].numpy())
        y_pred.append(pred["probabilities"].numpy())
        mask.append(pred["mask"].numpy())

    y_true = np.vstack(y_true)
    y_pred = np.vstack(y_pred)
    mask = np.vstack(mask)

    mean_auc, auc_vec = compute_mean_auc(y_true, y_pred, mask, label_mask)
    return mean_auc, auc_vec


def print_config(rparam, hparam):
    print(f"Tensorflow Version: {tf.__version__}")
    print(device_lib.list_local_devices())

    print(rparam)
    print(hparam)

def export_run_logs(run_type, run_log_dir, rparam, hparam):
    run_logs = {
        "run_type": run_type,
    }
    run_logs.update(hparam._asdict())
    run_logs.update(rparam._asdict())

    if not run_log_dir.exists():
        run_log_dir.mkdir()

    with open(run_log_dir / f"{rparam.run_id}.json", "w") as outfile:
        json.dump(run_logs, outfile)
