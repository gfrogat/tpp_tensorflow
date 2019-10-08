#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from pathlib import Path
import json
import datetime
import os


# In[ ]:


os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# In[ ]:


import tensorflow as tf


# In[ ]:


from tpp_tensorflow.config import get_params_semisparse, get_params_sparse, parser
from tpp_tensorflow.datautils import get_tfrecords_input_fn
from tpp_tensorflow.models.semisparse import SemiSparseInput
from tpp_tensorflow.models.sparse import SparseInput
from tpp_tensorflow.models.dense import DenseHead
from tpp_tensorflow.utils import compute_mean_auc_fold, print_config, export_run_logs

from tpp_tensorflow.steps import train_step, eval_step


# In[ ]:


TENSORBOARD_PORT = 6711


# In[ ]:


FEATURE_TYPE = "semisparse"


# In[ ]:


DATA_ROOT_DISK0 = Path("/local00/bioinf/tpp/")
DATA_ROOT_DISK1 = Path("/local01/bioinf/tpp/")


# In[ ]:


MODEL_DIR = (DATA_ROOT_DISK0 / "models").as_posix()
RUN_LOG_DIR = (DATA_ROOT_DISK0 / "run_logs").as_posix()


# In[ ]:


RECORDS_TRAIN = (DATA_ROOT_DISK1 / f"runs/thesis_chembl25/records_{FEATURE_TYPE}/train.tfrecords").as_posix()
RECORDS_TEST = (DATA_ROOT_DISK1 / f"runs/thesis_chembl25/records_{FEATURE_TYPE}/test.tfrecords").as_posix()


# In[ ]:


AUC_MASK_TRAIN_PATH = (DATA_ROOT_DISK1 / f"runs/thesis_chembl25/records_{FEATURE_TYPE}/label_mask_train.parquet/").as_posix()
AUC_MASK_TEST_PATH = (DATA_ROOT_DISK1 / f"runs/thesis_chembl25/records_{FEATURE_TYPE}/label_mask_test.parquet/").as_posix()


# In[ ]:


with open(DATA_ROOT_DISK1 / f"runs/thesis_chembl25/records_{FEATURE_TYPE}/metadata.json", "r") as infile:
    metadata = json.load(infile)


# In[ ]:


metadata


# In[ ]:


# Somewhat arbitrary number < nitems for train / val split
TRAIN_SET_SIZE = int(metadata["num_items_train"] * 0.6)
NUM_CLASSES = metadata["labels_size"]

if FEATURE_TYPE == "semisparse":
    CATS2D_SIZE = metadata["CATS2D_clean_size"]
    SHED_SIZE = metadata["SHED_clean_size"]
    
    custom_args = f"""
    --cats2d-size {CATS2D_SIZE} --shed-size {SHED_SIZE} 
    """
    
elif FEATURE_TYPE == "sparse":
    DFS8_SIZE = metadata["DFS8_clean_size"]
    ECFC4_SIZE = metadata["ECFC4_clean_size"]
    ECFC6_SIZE = metadata["ECFC6_clean_size"]
    
    FEATURE = "dfs8"
    FEATURE_SIZE = DFS8_SIZE
    EMBEDDING_SIZE = 2048
    
    custom_args = f"""
    --feature {FEATURE} --feature-size {FEATURE_SIZE} --embedding-size {EMBEDDING_SIZE} 
    --ecfc4-size {ECFC4_SIZE} --ecfc6-size {ECFC6_SIZE} --dfs8-size {DFS8_SIZE} 
    """


# In[ ]:


TRAIN_SET_SIZE


# In[ ]:


NUM_EPOCHS = 150
BATCH_SIZE = 64
DROPOUT_RATE = 0.4
INPUT_DROPOUT_RATE = 0.2
ACTIVATION = "selu"
REG_L2_RATE = 0.01
LR = 0.1
LR_DECAY_STEPS = 500000
LR_DECAY_RATE = 0.96


# In[ ]:


args = parser.parse_args((f"""
    {FEATURE_TYPE} 
    --model-dir {MODEL_DIR} 
    --run-log-dir {RUN_LOG_DIR} 
    --records-train {RECORDS_TRAIN} 
    --records-test {RECORDS_TEST} 
    --auc-mask-train-path {AUC_MASK_TRAIN_PATH} 
    --auc-mask-test-path {AUC_MASK_TEST_PATH} 
    --num-epochs {NUM_EPOCHS} --batch-size {BATCH_SIZE} --dropout-rate {DROPOUT_RATE} 
    --lr {LR} --lr-decay-steps {LR_DECAY_STEPS} --lr-decay-rate {LR_DECAY_RATE}
    --input-dropout-rate {INPUT_DROPOUT_RATE} --activation {ACTIVATION} --reg-l2-rate {REG_L2_RATE}
    --train-set-size {TRAIN_SET_SIZE} 
    """ + custom_args).split())


# In[ ]:


input_fn = get_tfrecords_input_fn(args.run_type)
get_params = get_params_semisparse if args.run_type == "semisparse" else get_params_sparse


# In[ ]:


# Arg Namespace to NamedTuple for autocomplete in IDE / Jupyter
rparam, hparam = get_params(args)


# In[ ]:


train_ds = input_fn(rparam.records_train, mode="train", cache=True, split_train_eval=True, train_set_size=rparam.train_set_size, num_epochs=1, batch_size=hparam.batch_size, rparams=rparam)
val_ds = input_fn(rparam.records_train, mode="eval", cache=True, split_train_eval=True, train_set_size=rparam.train_set_size, num_epochs=1, batch_size=hparam.batch_size, rparams=rparam)

train_ds_auc = input_fn(rparam.records_train, mode="eval", cache=True, split_train_eval=True, train_set_size=rparam.train_set_size, num_epochs=1, batch_size=hparam.batch_size, rparams=rparam)


# In[ ]:


input_model = SemiSparseInput(hparam) if args.run_type == "semisparse" else SparseInput(hparam)
output_model = DenseHead(hparam)


# In[ ]:


models = (input_model, output_model)


# In[ ]:


lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(hparam.lr, hparam.lr_decay_steps, hparam.lr_decay_rate, staircase=True)
optimizer = tf.keras.optimizers.SGD(lr=hparam.lr)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.Accuracy(name='train_accuracy')
train_metrics = (train_loss, train_accuracy)

val_loss = tf.keras.metrics.Mean(name='val_loss')
val_accuracy = tf.keras.metrics.Accuracy(name='val_accuracy')
val_metrics = (val_loss, val_accuracy)


# In[ ]:


logdir = Path(rparam.model_dir) / "logs"


# In[ ]:


#%load_ext tensorboard


# In[ ]:


#%tensorboard --logdir $logdir --port 6711


# In[ ]:


run_id = rparam.run_id

train_log_dir = logdir / "train"
val_log_dir = logdir / "val"
train_summary_writer = tf.summary.create_file_writer(train_log_dir.as_posix())
val_summary_writer = tf.summary.create_file_writer(val_log_dir.as_posix())


# In[ ]:


global_step = 0
for epoch in range(hparam.num_epochs):
    for features, labels in train_ds:        
        optimizer.learning_rate = lr_scheduler(global_step)
        train_step(features, labels, models, train_metrics, optimizer, hparam)
        global_step += 1
        
    train_mean_auc, _ = compute_mean_auc_fold(models, train_ds_auc, eval_step, hparam, rparam.auc_mask_train_path)
    
    with train_summary_writer.as_default():
        tf.summary.scalar('loss', train_loss.result(), step=epoch)
        tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)
        tf.summary.scalar("mean_auc", train_mean_auc, step=epoch)

    for features, labels in val_ds:
        eval_step(features, labels, models, val_metrics, hparam)
        
    val_mean_auc, _ = compute_mean_auc_fold(models, val_ds, eval_step, hparam, rparam.auc_mask_test_path)
    
    with val_summary_writer.as_default():
        tf.summary.scalar('loss', val_loss.result(), step=epoch)
        tf.summary.scalar('accuracy', val_accuracy.result(), step=epoch)
        tf.summary.scalar("mean_auc", val_mean_auc, step=epoch)
        
    template = 'Epoch {0}, Loss: {1:.5g}, Accuracy: {2:.5g}, meanAUC: {3:.5g}, Val Loss: {4:.5g}, Val Accuracy: {5:.5g}, Val meanAUC: {6:.5g}'
    print(template.format(
        epoch,
        train_loss.result(),
        train_accuracy.result(),
        train_mean_auc,
        val_loss.result(),
        val_accuracy.result(),
        val_mean_auc))
    
    train_loss.reset_states()
    val_loss.reset_states()
    train_accuracy.reset_states()
    val_accuracy.reset_states()


# In[ ]:


model.save(rparam.model_dir)


# In[ ]:


export_run_logs(args.run_type, args.run_log_dir, rparam, hparam)


# In[ ]:


test_ds = input_fn(rparam.records_test, mode="eval", cache=False, split_train_eval=False, num_epochs=1, batch_size=hparam.batch_size, rparams=rparam)
test_mean_auc, _ = compute_mean_auc_fold(models, test_ds, eval_step, hparam, rparam.auc_mask_test_path)


# In[ ]:


print(f"Test meanAUC: {test_mean_auc}")


# In[ ]:





