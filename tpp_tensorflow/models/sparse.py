from tensorflow.keras import Model, layers, regularizers


class SparseInput(Model):
    def __init__(self, params):
        super(SparseInput, self).__init__()

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

        self.input_layer = layers.Dense(
            params.embedding_size,
            activation=params.activation,
            kernel_initializer=kernel_init,
            kernel_regularizer=kernel_reg,
            name="input_{}".format(params.feature),
        )


    def call(self, features, training=False):
        x = self.input_layer(features)
        return x