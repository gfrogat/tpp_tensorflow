from __future__ import absolute_import, division, print_function, unicode_literals

from functools import partial

import tensorflow as tf

feature_description_semisparse = {
    "maccs_fp": tf.io.VarLenFeature(tf.int64),
    "labels_val": tf.io.VarLenFeature(tf.int64),
    "labels_idx": tf.io.VarLenFeature(tf.int64),
    "rdkit_fp": tf.io.VarLenFeature(tf.int64),
    "PubChem": tf.io.VarLenFeature(tf.int64),
    "CATS2D_clean_val": tf.io.VarLenFeature(tf.float32),
    "CATS2D_clean_idx": tf.io.VarLenFeature(tf.int64),
    "SHED_clean_idx": tf.io.VarLenFeature(tf.int64),
    "SHED_clean_val": tf.io.VarLenFeature(tf.float32),
}

feature_description_sparse = {
    "labels_val": tf.io.VarLenFeature(tf.int64),
    "labels_idx": tf.io.VarLenFeature(tf.int64),
    "ECFC4_clean_val": tf.io.VarLenFeature(tf.float32),
    "ECFC4_clean_idx": tf.io.VarLenFeature(tf.int64),
    "ECFC6_clean_val": tf.io.VarLenFeature(tf.float32),
    "ECFC6_clean_idx": tf.io.VarLenFeature(tf.int64),
    "DFS8_clean_idx": tf.io.VarLenFeature(tf.int64),
    "DFS8_clean_val": tf.io.VarLenFeature(tf.float32),
}


def parse_single_tf_example(example_proto, feature_description):
    parsed_features = tf.io.parse_single_example(example_proto, feature_description)
    return parsed_features


def extract_sparse_features(parsed_features, idx_name, val_name, feature_len):
    labels_idx = parsed_features.pop(idx_name)
    labels_val = parsed_features.pop(val_name)

    labels_idx = tf.stack([labels_idx.values], axis=-1)
    labels_val = tf.cast(labels_val.values, tf.float32)

    target = tf.SparseTensor(labels_idx, labels_val, dense_shape=[feature_len])

    return target


def process_single_tf_example_semisparse(parsed_features, rparams):
    target = tf.sparse.to_dense(
        extract_sparse_features(
            parsed_features, "labels_idx", "labels_val", rparams.num_classes
        )
    )

    parsed_features["shed"] = tf.sparse.to_dense(
        extract_sparse_features(
            parsed_features, "SHED_clean_idx", "SHED_clean_val", rparams.shed_size
        )
    )
    parsed_features["cats2d"] = tf.sparse.to_dense(
        extract_sparse_features(
            parsed_features, "CATS2D_clean_idx", "CATS2D_clean_val", rparams.cats2d_size
        )
    )

    maccs_fp = parsed_features["maccs_fp"]
    maccs_indices = tf.reshape([maccs_fp.values], shape=(-1, 1))
    parsed_features["maccs_fp"] = tf.scatter_nd(
        maccs_indices,
        tf.ones_like(maccs_fp.values, dtype=tf.float32),
        shape=(rparams.maccs_fp_size,),
    )

    rdkit_fp = parsed_features["rdkit_fp"]
    rdkit_indices = tf.reshape([rdkit_fp.values], shape=(-1, 1))
    parsed_features["rdkit_fp"] = tf.scatter_nd(
        rdkit_indices,
        tf.ones_like(rdkit_fp.values, dtype=tf.float32),
        shape=(rparams.rdkit_fp_size,),
    )

    pubchem_fp = parsed_features.pop("PubChem")
    pubchem_indices = tf.reshape([pubchem_fp.values], shape=(-1, 1))
    parsed_features["pubchem_fp"] = tf.scatter_nd(
        pubchem_indices,
        tf.ones_like(pubchem_fp.values, dtype=tf.float32),
        shape=(rparams.pubchem_fp_size,),
    )

    return (parsed_features, target)


def process_single_tf_example_sparse(parsed_features, rparams):
    target = tf.sparse.to_dense(
        extract_sparse_features(
            parsed_features, "labels_idx", "labels_val", rparams.num_classes
        )
    )

    parsed_features["ecfc4"] = extract_sparse_features(
        parsed_features, "ECFC4_clean_idx", "ECFC4_clean_val", rparams.ecfc4_size
    )
    parsed_features["ecfc6"] = extract_sparse_features(
        parsed_features, "ECFC6_clean_idx", "ECFC6_clean_val", rparams.ecfc6_size
    )
    parsed_features["dfs8"] = extract_sparse_features(
        parsed_features, "DFS8_clean_idx", "DFS8_clean_val", rparams.dfs8_size
    )

    features = parsed_features.pop(rparams.feature)

    return (features, target)


def get_tfrecords_input_fn(feature):
    if feature == "semisparse":
        feature_description = feature_description_semisparse
        process_single_tf_example = process_single_tf_example_semisparse
    elif feature == "sparse":
        feature_description = feature_description_sparse
        process_single_tf_example = process_single_tf_example_sparse

    def tfrecords_input_fn(
        files_name_pattern,
        mode="train",
        batch_size=10,
        num_epochs=1,
        cache=False,
        split_train_eval=False,
        train_set_size=278578,
        shuffle=False,
        rparams=None,
    ):
        num_calls = (
            tf.data.experimental.AUTOTUNE
            if rparams.num_calls == -1
            else rparams.num_calls
        )

        file_names = tf.io.matching_files(files_name_pattern)

        with tf.name_scope("input_pipeline"):
            dataset = tf.data.TFRecordDataset(
                filenames=file_names, num_parallel_reads=rparams.num_reads
            )

            if split_train_eval is True:
                if mode == "train":
                    dataset = dataset.take(train_set_size)
                else:
                    assert mode == "eval"
                    dataset = dataset.skip(train_set_size)

            dataset = dataset.map(
                partial(
                    parse_single_tf_example, feature_description=feature_description
                ),
                num_parallel_calls=num_calls,
            )
            if cache is True:
                dataset = dataset.cache()

            if shuffle:
                dataset = dataset.shuffle(
                    buffer_size=4 * batch_size + 1, seed=rparams.shuffle_seed
                )

            dataset = dataset.map(
                partial(process_single_tf_example, rparams=rparams),
                num_parallel_calls=num_calls,
            )

            dataset = dataset.batch(batch_size, drop_remainder=False)
            dataset = dataset.prefetch(1)
            dataset = dataset.repeat(num_epochs)

        return dataset

    return tfrecords_input_fn
