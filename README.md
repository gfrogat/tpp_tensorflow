# TPP TensorFlow

## Code

The repository contains [a script](./scripts/tpp_tf2.py) and the identical [Jupyter Notebook](./notebooks/tpp_tf2.ipynb) for training networks using TensorFlow using the different features computed using the TPP Spark jobs.
All scripts come with optimized `tf.data` input pipelines based TFRecord Datasets. The code was updated to TensorFlow 2.0.0. While the GPU utilization for the initial Tensorflow `v1.14` code was reasonable (`90%`), the new `v2.0.0` code has worse performance (`70%`). Dataloading generally should not be the bottleneck.

The scripts only provide templates you can build on. The hyperparameters and architectures were not optimized and should only be used as guidance.
As a warning: As soon you are using sparse tensors most optimizers might behave differently (e.g. due to wrong handling of momentum). I recommend to work with classic Gradient Descent or to have a look at Adagrad which is supposed to work better with sparse data (I did not look into this further due to time reasons).

The configuration of the code utilizes Python `named_tuples` which enable auto-completion in your IDE or Jupyter Notebook. It's a bit annoying to extend (you have to update multiple places) but auto-completion is worth it.

## Conda

The repo includes a conda environment file that can be used to setup TensorFlow `1.14` with Python `3.6`.

```bash
conda env create -f tools/conda/tf_v2.0.0_py3.7.yml -n "tpp_tf"
```

## Requirements

The general requirements are pretty minimal:

```bash
python=3.7
tensorflow-gpu=2.0
scikit-learn
pandas
pyarrow
```
