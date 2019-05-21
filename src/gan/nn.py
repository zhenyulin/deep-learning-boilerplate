from tensorflow import nn, variable_scope, layers, sigmoid, reduce_mean

### Networks
def generator(seed, reuse=None):
    with variable_scope("generator", reuse=reuse):
        layer1 = layers.dense(inputs=seed, units=128, activation=nn.leaky_relu)
        layer2 = layers.dense(inputs=layer1, units=128, activation=nn.leaky_relu)
        output = layers.dense(inputs=layer2, units=784, activation=nn.tanh)

        return output


def discriminator(input, reuse=None):
    with variable_scope("discriminator", reuse=reuse):
        layer1 = layers.dense(inputs=input, units=128, activation=nn.leaky_relu)
        layer2 = layers.dense(inputs=layer1, units=128, activation=nn.leaky_relu)
        logits = layers.dense(layer2, units=1)
        output = sigmoid(logits)

        return output, logits


def loss_func(logits_in, labels_in):
    return reduce_mean(
        nn.sigmoid_cross_entropy_with_logits(logits=logits_in, labels=labels_in)
    )
