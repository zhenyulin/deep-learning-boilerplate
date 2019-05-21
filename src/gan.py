import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

### training data set
mnist = input_data.read_data_sets("MNIST_data")

### Networks
def generator(seed, reuse=None):
    with tf.variable_scope("gen", reuse=reuse):
        hidden1 = tf.layers.dense(inputs=seed, units=128, activation=tf.nn.leaky_relu)
        hidden2 = tf.layers.dense(
            inputs=hidden1, units=128, activation=tf.nn.leaky_relu
        )
        output = tf.layers.dense(inputs=hidden2, units=784, activation=tf.nn.tanh)

        return output


def discriminator(input, reuse=None):
    with tf.variable_scope("dis", reuse=reuse):
        hidden1 = tf.layers.dense(inputs=input, units=128, activation=tf.nn.leaky_relu)
        hidden2 = tf.layers.dense(
            inputs=hidden1, units=128, activation=tf.nn.leaky_relu
        )
        logits = tf.layers.dense(hidden2, units=1)
        output = tf.sigmoid(logits)

        return output, logits  ### QUESTION: what is logits?


### Initialise Network
tf.reset_default_graph()

real_images = tf.placeholder(tf.float32, shape=[None, 784])
seed = tf.placeholder(tf.float32, shape=[None, 100])

G = generator(seed)
D_output_real, D_logits_real = discriminator(real_images)
D_output_fake, D_logits_fake = discriminator(G, reuse=True)

### Loss Function
def loss_func(logits_in, labels_in):
    return tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_in, labels=labels_in)
    )


D_real_loss = loss_func(D_logits_real, tf.ones_like(D_logits_real) * 0.9)
D_fake_loss = loss_func(D_logits_fake, tf.zeros_like(D_logits_real))
D_loss = D_real_loss + D_fake_loss

G_loss = loss_func(D_logits_fake, tf.ones_like(D_logits_fake))

### Training Settings
lr = 0.001

tvars = tf.trainable_variables()
d_vars = [var for var in tvars if "dis" in var.name]
g_vars = [var for var in tvars if "gen" in var.name]

D_trainer = tf.train.AdamOptimizer(lr).minimize(D_loss, var_list=d_vars)
G_trainer = tf.train.AdamOptimizer(lr).minimize(G_loss, var_list=g_vars)

BATCH_SIZE = 100
EPOCHS = 100
init = tf.global_variables_initializer()

### Training Process
samples = []

with tf.Session() as s:
    s.run(init)
    for epoch in range(EPOCHS):
        num_batches = mnist.train.num_examples // BATCH_SIZE
        for i in range(num_batches):
            batch = mnist.train.next_batch(BATCH_SIZE)
            batch_images = batch[0].reshape(
                (BATCH_SIZE, 784)
            )  ### is it some shorthand?
            batch_images = batch_images * 2 - 1  ### what this is doing?
            batch_seed = np.random.uniform(-1, 1, size=(BATCH_SIZE, 100))
            _ = s.run(
                D_trainer, feed_dict={real_images: batch_images, seed: batch_seed}
            )  ### alternative function invoke?
            _ = s.run(G_trainer, feed_dict={seed: batch_seed})

        print("on epoch{}".format(epoch))

        sample_seed = np.random.uniform(-1, 1, size=(1, 100))
        gen_sample = s.run(generator(seed, reuse=True), feed_dict={seed: sample_seed})

        samples.append(gen_sample)

plt.imshow(samples[0].reshape(28, 28))
plt.imshow(samples[99].reshape(28, 28))
