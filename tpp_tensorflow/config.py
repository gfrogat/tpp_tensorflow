from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import uuid
from collections import namedtuple
from pathlib import Path

RParam = namedtuple(
    "RParam",
    [
        "records_train",
        "records_test",
        "auc_mask_train_path",
        "auc_mask_test_path",
        "model_dir",
        "train_set_size",
        "num_classes",
        "num_calls",
        "num_reads",
        "shuffle_seed",
        "run_id",
        "feature"
    ],
)

HParam = namedtuple(
    "HParam",
    [
        "hidden_units",
        "hidden_layers",
        "activation",
        "lr",
        "lr_decay_steps",
        "lr_decay_rate",
        "batch_size",
        "num_epochs",
        "num_classes",
        "dropout_rate",
        "dropout_seed",
        "input_dropout_rate",
        "input_dropout_seed",
        "mask_value",
        "feature",
        "embedding_size",
        "kernel_init",
        "reg_l2_rate",
    ],
)


SemiSparseParam = namedtuple(
    "SemiSparseParam",
    ["maccs_fp_size", "rdkit_fp_size", "pubchem_fp_size", "cats2d_size", "shed_size"]
    + list(RParam._fields),
)


SparseParam = namedtuple(
    "SparseParam", ["ecfc4_size", "ecfc6_size", "dfs8_size"] + list(RParam._fields)
)


parser_runparam = argparse.ArgumentParser("RunParam", add_help=False)
parser_runparam.add_argument("--run-log-dir", type=Path, required=True)

parser_runparam.add_argument("--records-train", type=Path, required=True)
parser_runparam.add_argument("--records-test", type=Path, required=True)
parser_runparam.add_argument("--auc-mask-train-path", type=Path, required=True)
parser_runparam.add_argument("--auc-mask-test-path", type=Path, required=True)
parser_runparam.add_argument("--model-dir", type=Path, required=True)
parser_runparam.add_argument("--checkpoint-secs", type=int, default=60)
parser_runparam.add_argument("--train-set-size", type=int, default=278578)
parser_runparam.add_argument("--num-classes", type=int, default=2193)
parser_runparam.add_argument("--num-calls", type=int, default=-1)
parser_runparam.add_argument("--num-reads", type=int, default=12)
parser_runparam.add_argument("--shuffle-seed", type=int, default=42)

parser_hyperparam = argparse.ArgumentParser("HyperParam", add_help=False)
parser_hyperparam.add_argument("--hidden-units", type=int, default=2000)
parser_hyperparam.add_argument("--hidden-layers", type=int, default=3)
parser_hyperparam.add_argument("--activation", type=str, default="relu")
parser_hyperparam.add_argument("--lr", type=float, default=0.1)
parser_hyperparam.add_argument("--lr-decay-steps", type=int, default=500000)
parser_hyperparam.add_argument("--lr-decay-rate", type=float, default=0.96)
parser_hyperparam.add_argument("--batch-size", type=int, default=128)
parser_hyperparam.add_argument("--num-epochs", type=int, default=150)
parser_hyperparam.add_argument("--dropout-rate", type=float, default=0.5)
parser_hyperparam.add_argument("--dropout-seed", type=int, default=42)
parser_hyperparam.add_argument("--input-dropout-rate", type=float, default=0.2)
parser_hyperparam.add_argument("--input-dropout-seed", type=int, default=42)
parser_hyperparam.add_argument("--mask-value", type=float, default=0.0)
parser_hyperparam.add_argument("--kernel-init", type=str, default="glorot_uniform")
parser_hyperparam.add_argument("--reg-l2-rate", type=float, default=0.001)
# only used by sparse version
parser_hyperparam.add_argument("--feature", type=str, default="dfs8")
parser_hyperparam.add_argument("--embedding-size", type=int, default=512)


parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(help="sub-command --help", dest="run_type")
parser_semisparse = subparsers.add_parser(
    "semisparse", parents=[parser_runparam, parser_hyperparam], help="semisparse --help"
)

parser_semisparse.add_argument("--maccs-fp-size", type=int, default=167)
parser_semisparse.add_argument("--rdkit-fp-size", type=int, default=2048)
parser_semisparse.add_argument("--pubchem-fp-size", type=int, default=881)
parser_semisparse.add_argument("--cats2d-size", type=int, default=29)
parser_semisparse.add_argument("--shed-size", type=int, default=3)

parser_sparse = subparsers.add_parser(
    "sparse", parents=[parser_runparam, parser_hyperparam], help="sparse --help"
)
parser_sparse.add_argument("--ecfc4-size", type=int, default=82361)
parser_sparse.add_argument("--ecfc6-size", type=int, default=214795)
parser_sparse.add_argument("--dfs8-size", type=int, default=82854)


def get_hparams(args):
    hparam = HParam(
        hidden_units=args.hidden_units,
        hidden_layers=args.hidden_layers,
        activation=args.activation,
        lr=args.lr,
        lr_decay_steps=args.lr_decay_steps,
        lr_decay_rate=args.lr_decay_rate,
        batch_size=args.batch_size,
        num_classes=args.num_classes,
        num_epochs=args.num_epochs,
        dropout_rate=args.dropout_rate,
        dropout_seed=args.dropout_seed,
        input_dropout_rate=args.input_dropout_rate,
        input_dropout_seed=args.input_dropout_seed,
        mask_value=args.mask_value,
        feature=args.feature,
        embedding_size=args.embedding_size,
        kernel_init=args.kernel_init,
        reg_l2_rate=args.reg_l2_rate,
    )

    return hparam


def get_params_semisparse(args):
    run_id = str(uuid.uuid4())

    rparam = SemiSparseParam(
        run_id=run_id,
        records_train=(args.records_train / "part-*").as_posix(),
        records_test=(args.records_test / "part-*").as_posix(),
        auc_mask_train_path=args.auc_mask_train_path.as_posix(),
        auc_mask_test_path=args.auc_mask_test_path.as_posix(),
        model_dir=(args.model_dir / run_id).as_posix(),
        train_set_size=args.train_set_size,
        num_classes=args.num_classes,
        num_calls=args.num_calls,
        num_reads=args.num_reads,
        shuffle_seed=args.shuffle_seed,
        maccs_fp_size=args.maccs_fp_size,
        rdkit_fp_size=args.rdkit_fp_size,
        pubchem_fp_size=args.pubchem_fp_size,
        cats2d_size=args.cats2d_size,
        shed_size=args.shed_size,
        feature=None
    )

    hparam = get_hparams(args)

    return rparam, hparam


def get_params_sparse(args):
    run_id = str(uuid.uuid4())

    rparam = SparseParam(
        run_id=run_id,
        records_train=(args.records_train / "part-*").as_posix(),
        records_test=(args.records_test / "part-*").as_posix(),
        auc_mask_train_path=args.auc_mask_train_path.as_posix(),
        auc_mask_test_path=args.auc_mask_test_path.as_posix(),
        model_dir=(args.model_dir / run_id).as_posix(),
        train_set_size=args.train_set_size,
        num_classes=args.num_classes,
        num_calls=args.num_calls,
        num_reads=args.num_reads,
        shuffle_seed=args.shuffle_seed,
        ecfc4_size=args.ecfc4_size,
        ecfc6_size=args.ecfc6_size,
        dfs8_size=args.dfs8_size,
        feature=args.feature
    )

    hparam = get_hparams(args)

    return rparam, hparam
