#!/usr/bin/env python
# coding: utf-8
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from tpp_tensorflow.config import get_params_semisparse, get_params_sparse, parser
from tpp_tensorflow.datautils import get_tfrecords_input_fn
from tpp_tensorflow.models.dense import DenseHead
from tpp_tensorflow.models.semisparse import SemiSparseInput
from tpp_tensorflow.models.sparse import SparseInput
from tpp_tensorflow.steps import eval_step, train_step
from tpp_tensorflow.utils import compute_mean_auc_fold, export_run_logs, print_config


if __name__ == "__main__":
    args = parser.parse_args()

    # Set hyperparameter type
    input_fn = get_tfrecords_input_fn(args.run_type)
    get_params = (
        get_params_semisparse if args.run_type == "semisparse" else get_params_sparse
    )

    # Arg Namespace to NamedTuple for autocomplete in IDE / Jupyter
    rparam, hparam = get_params(args)
    print_config(rparam, hparam)

    # Setup dataloaders
    train_ds = input_fn(
        rparam.records_train,
        mode="train",
        cache=True,
        split_train_eval=True,
        train_set_size=rparam.train_set_size,
        num_epochs=1,
        batch_size=hparam.batch_size,
        shuffle=True,
        rparams=rparam,
    )
    val_ds = input_fn(
        rparam.records_train,
        mode="eval",
        cache=True,
        split_train_eval=True,
        train_set_size=rparam.train_set_size,
        num_epochs=1,
        batch_size=hparam.batch_size,
        rparams=rparam,
    )
    train_ds_auc = input_fn(
        rparam.records_train,
        mode="train",
        cache=True,
        split_train_eval=True,
        train_set_size=rparam.train_set_size,
        num_epochs=1,
        batch_size=hparam.batch_size,
        rparams=rparam,
    )

    # Setup model
    if args.run_type == "sparse":
        feature_size = getattr(rparam, f"{rparam.feature}_size")
        input_model = SparseInput(hparam)
        # build model because SparseTensors lack shape
        input_model.build((hparam.batch_size, feature_size))
    else:
        assert args.run_type == "semisparse"
        input_model = SemiSparseInput(hparam)

    output_model = DenseHead(hparam)
    models = (input_model, output_model)

    # Setup optimizer
    lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
        hparam.lr, hparam.lr_decay_steps, hparam.lr_decay_rate, staircase=True
    )
    optimizer = tf.keras.optimizers.SGD(lr=hparam.lr)

    # Setup metrics
    train_loss = tf.keras.metrics.Mean(name="train_loss")
    train_accuracy = tf.keras.metrics.Accuracy(name="train_accuracy")
    train_metrics = (train_loss, train_accuracy)

    val_loss = tf.keras.metrics.Mean(name="val_loss")
    val_accuracy = tf.keras.metrics.Accuracy(name="val_accuracy")
    val_metrics = (val_loss, val_accuracy)

    # Setup experiment logging
    logdir = Path(rparam.model_dir) / "logs"
    run_id = rparam.run_id

    train_log_dir = logdir / "train"
    val_log_dir = logdir / "val"
    train_summary_writer = tf.summary.create_file_writer(train_log_dir.as_posix())
    val_summary_writer = tf.summary.create_file_writer(val_log_dir.as_posix())

    # Load label masks (we only compute AUC where we have
    # at least one active and inactive label)
    label_mask_train = pd.read_parquet(rparam.auc_mask_train_path).key.values
    label_mask_val = pd.read_parquet(rparam.auc_mask_train_path).key.values

    # New TensorFlow 2.0 training loop using GradientTape
    global_step = 0
    for epoch in range(hparam.num_epochs):
        for features, labels in train_ds:
            optimizer.learning_rate = lr_scheduler(global_step)
            train_step(features, labels, models, train_metrics, optimizer, hparam)
            global_step += 1

        train_mean_auc, _ = compute_mean_auc_fold(
            models, train_ds_auc, eval_step, hparam, label_mask_train
        )

        with train_summary_writer.as_default():
            tf.summary.scalar("loss", train_loss.result(), step=epoch)
            tf.summary.scalar("accuracy", train_accuracy.result(), step=epoch)
            tf.summary.scalar("mean_auc", train_mean_auc, step=epoch)

        for features, labels in val_ds:
            eval_step(features, labels, models, val_metrics, hparam)

        val_mean_auc, _ = compute_mean_auc_fold(
            models, val_ds, eval_step, hparam, label_mask_val
        )

        with val_summary_writer.as_default():
            tf.summary.scalar("loss", val_loss.result(), step=epoch)
            tf.summary.scalar("accuracy", val_accuracy.result(), step=epoch)
            tf.summary.scalar("mean_auc", val_mean_auc, step=epoch)

        template = (
            "Epoch {0}, Loss: {1:.5g}, Accuracy: {2:.5g}, meanAUC: {3:.8g}, "
            "Val Loss: {4:.5g}, Val Accuracy: {5:.5g}, Val meanAUC: {6:.8g}"
        )
        print(
            template.format(
                epoch,
                train_loss.result(),
                train_accuracy.result(),
                train_mean_auc,
                val_loss.result(),
                val_accuracy.result(),
                val_mean_auc,
            )
        )

        train_loss.reset_states()
        val_loss.reset_states()
        train_accuracy.reset_states()
        val_accuracy.reset_states()

    # Export logs for experiment
    export_run_logs(args.run_type, args.run_log_dir, rparam, hparam)

    # Get test set meanAUC
    label_mask_test = pd.read_parquet(rparam.auc_mask_test_path).key.values
    common_assays_mask = pd.read_parquet(rparam.common_assays_mask_path)
    label_mask_test = set(label_mask_test) & set(
        np.nonzero(common_assays_mask.assay_id.values)[0]
    )

    test_ds = input_fn(
        rparam.records_test,
        mode="eval",
        cache=False,
        split_train_eval=False,
        num_epochs=1,
        batch_size=hparam.batch_size,
        rparams=rparam,
    )
    test_mean_auc, _ = compute_mean_auc_fold(
        models, test_ds, eval_step, hparam, label_mask_test
    )

    print(f"Test meanAUC: {test_mean_auc}")
