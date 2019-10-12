from tensorflow.keras import Model, layers, regularizers


class SemiSparseInput(Model):
    def __init__(self, params):
        super(SemiSparseInput, self).__init__()

        # Correctly handle SELU
        dropout = layers.AlphaDropout if params.activation == "selu" else layers.Dropout
        kernel_init = (
            "lecun_normal" if params.activation == "selu" else params.kernel_init
        )

        kernel_reg = (
            regularizers.l2(params.reg_l2_rate)
            if params.reg_l2_rate is not None
            else None
        )

        self.input_dropout_maccs_fp = dropout(
            rate=params.input_dropout_rate,
            seed=params.input_dropout_seed,
            name="input_dropout_maccs_fp",
        )
        self.input_maccs_fp = layers.Dense(
            256,
            activation=params.activation,
            kernel_initializer=kernel_init,
            kernel_regularizer=kernel_reg,
            name="dense_maccs_fp",
        )

        self.input_dropout_rdkit_fp = dropout(
            rate=params.input_dropout_rate,
            seed=params.input_dropout_seed,
            name="input_dropout_rdkit_fp",
        )
        self.input_rdkit_fp = layers.Dense(
            2048,
            activation=params.activation,
            kernel_initializer=kernel_init,
            kernel_regularizer=kernel_reg,
            name="dense_rdkit_fp",
        )

        self.input_dropout_pubchem_fp = dropout(
            rate=params.input_dropout_rate,
            seed=params.input_dropout_seed,
            name="input_dropout_pubchem_fp",
        )
        self.input_pubchem_fp = layers.Dense(
            1024,
            activation=params.activation,
            kernel_initializer=kernel_init,
            kernel_regularizer=kernel_reg,
            name="dense_pubchem_fp",
        )

        self.input_shed = layers.Dense(
            8,
            activation=params.activation,
            kernel_initializer=kernel_init,
            kernel_regularizer=kernel_reg,
            name="dense_shed",
        )

        self.input_dropout_cats2d = dropout(
            rate=params.input_dropout_rate,
            seed=params.input_dropout_seed,
            name="input_dropout_cats2d",
        )
        self.input_cats2d = layers.Dense(
            32,
            activation=params.activation,
            kernel_initializer=kernel_init,
            kernel_regularizer=kernel_reg,
            name="dense_cats2d",
        )

    def call(self, features, training=False):
        x1 = self.input_dropout_maccs_fp(features["maccs_fp"])
        x1 = self.input_maccs_fp(x1)

        x2 = self.input_dropout_rdkit_fp(features["rdkit_fp"])
        x2 = self.input_rdkit_fp(x2)

        x3 = self.input_dropout_pubchem_fp(features["pubchem_fp"])
        x3 = self.input_pubchem_fp(x3)

        x4 = self.input_shed(features["shed"])

        x5 = self.input_dropout_cats2d(features["cats2d"])
        x5 = self.input_cats2d(x5)

        x = layers.concatenate([x1, x2, x3, x4, x5], axis=1)
        return x
