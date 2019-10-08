import tensorflow as tf


def masked_cross_entropy(labels, logits, mask):
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=labels, logits=logits
    )
    loss = tf.reduce_sum(cross_entropy * mask) / tf.reduce_sum(mask)
    return loss
