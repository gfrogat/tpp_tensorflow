
from tensorflow.keras import layers, Model, regularizers


class DenseHead(Model):
    def __init__(self, params):
        super(DenseHead, self).__init__()

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

        self.hidden_layers = []
        self.dropout_layers = []
        for i in range(params.hidden_layers):
            self.hidden_layers.append(
                layers.Dense(
                    params.hidden_units,
                    activation=params.activation,
                    kernel_initializer=kernel_init,
                    kernel_regularizer=kernel_reg,
                    name="dense_{}".format(i),
                )
            )
            self.dropout_layers.append(
                dropout(
                    params.dropout_rate,
                    seed=params.dropout_seed,
                    name="dropout_{}".format(i),
                )
            )

        self.output_layer = layers.Dense(
            params.num_classes, activation=None, name="dense_output"
        )

    def call(self, x, training=False):
        for h_layer, d_layer in zip(self.hidden_layers, self.dropout_layers):
            x = d_layer(x, training)
            x = h_layer(x)

        return self.output_layer(x)
