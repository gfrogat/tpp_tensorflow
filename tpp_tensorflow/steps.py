from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tpp_tensorflow.losses import masked_cross_entropy


@tf.function
def train_step(features, labels, models, metrics, optimizer, params):
    train_loss, train_accuracy = metrics
    input_model, output_model = models

    with tf.GradientTape() as tape:
        x = input_model(features, training=True)
        logits = output_model(x, training=True)

        with tf.name_scope("compute_probabilities"):
            probabilities = tf.sigmoid(logits)

        with tf.name_scope("compute_predictions"):
            predictions = tf.round(probabilities)

        with tf.name_scope("create_mask"):
            mask = tf.cast(tf.not_equal(labels, params.mask_value), tf.float32)

        with tf.name_scope("format_labels"):
            labels_fmt = labels + tf.cast(tf.equal(labels, -1), tf.float32)

        loss = masked_cross_entropy(labels_fmt, logits, mask)

    gradients = tape.gradient(loss, input_model.trainable_variables + output_model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, input_model.trainable_variables + output_model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels_fmt, predictions, mask)


@tf.function
def eval_step(features, labels, models, metrics, params, output=False):
    eval_loss, eval_accuracy = metrics
    input_model, output_model = models

    x = input_model(features, training=True)
    logits = output_model(x, training=True)

    with tf.name_scope("compute_probabilities"):
        probabilities = tf.sigmoid(logits)

    with tf.name_scope("compute_predictions"):
        predictions = tf.round(probabilities)

    assert labels is not None
    with tf.name_scope("create_mask"):
        mask = tf.cast(tf.not_equal(labels, params.mask_value), tf.float32)

    with tf.name_scope("format_labels"):
        labels_fmt = labels + tf.cast(tf.equal(labels, -1), tf.float32)

    loss = masked_cross_entropy(labels_fmt, logits, mask)

    eval_loss(loss)
    eval_accuracy(labels_fmt, predictions, mask)

    if output is True:
        return {"labels": labels_fmt, "probabilities": probabilities, "mask": mask}
