import tensorflow as tf

from tensorflow import nn, layers, sigmoid

# Networks
# define the input, output and layers
def generator(seed, reuse=None):
    with tf.variable_scope("generator", reuse=reuse):
        layer1 = layers.dense(inputs=seed, units=128, activation=nn.leaky_relu)
        layer2 = layers.dense(inputs=layer1, units=128, activation=nn.leaky_relu)
        output = layers.dense(inputs=layer2, units=784, activation=nn.tanh)

        return output


def discriminator(input, reuse=None):
    with tf.variable_scope("discriminator", reuse=reuse):
        layer1 = layers.dense(inputs=input, units=128, activation=nn.leaky_relu)
        layer2 = layers.dense(inputs=layer1, units=128, activation=nn.leaky_relu)
        logits = layers.dense(layer2, units=1)
        output = sigmoid(logits)

        return output, logits


# logits is a tensor [None, {R}], labels (= [None, {0, 1}]
# sigmoid function typically use a logistic function to turn R into (-1,1)
# cross_entropy is an advanced alternative to squared_error
# reduce_mean turns the tensor(vector here) into a number
def loss_func(logits_in, labels_in):
    return tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_in, labels=labels_in)
    )


tf.reset_default_graph()  # stop any on-going tf session

# placeholder defines the structure of a tensor to be fed with feed_dict
real_images = tf.placeholder(tf.float32, shape=[None, 784])

seed = tf.placeholder(tf.float32, shape=[None, 100])
generation = generator(seed)

d_real_output, d_real_logits = discriminator(real_images)
d_fake_output, d_fake_logits = discriminator(generation, reuse=True)

# discriminator to mimimize the loss in judging real/fake(1/0)
d_real_loss = loss_func(d_real_logits, tf.ones_like(d_real_logits) * 0.9)
d_fake_loss = loss_func(d_fake_logits, tf.zeros_like(d_real_logits))
d_loss = d_real_loss + d_fake_loss

# generator to mimimize the loss in generating a real-like(1) output
g_loss = loss_func(d_fake_logits, tf.ones_like(d_fake_logits))
