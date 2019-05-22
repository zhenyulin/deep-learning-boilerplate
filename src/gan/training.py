import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.examples.tutorials.mnist.input_data as mnist

from networks import generator, d_loss, g_loss, real_images, seed

### hyperparameters
LEFTOUT_RATE = 0.001
BATCH_SIZE = 100
EPOCHS = 100

### trainer function
tvars = tf.trainable_variables()
d_vars = [var for var in tvars if "discriminator" in var.name]
g_vars = [var for var in tvars if "generator" in var.name]

d_trainer = tf.train.AdamOptimizer(LEFTOUT_RATE).minimize(d_loss, var_list=d_vars)
g_trainer = tf.train.AdamOptimizer(LEFTOUT_RATE).minimize(g_loss, var_list=g_vars)

### load training dataset
training_set = mnist.read_data_sets("MNIST_data")
samples = []

with tf.Session() as s:
    s.run(tf.global_variables_initializer())
    for epoch in range(EPOCHS):
        batches = training_set.train.num_examples // BATCH_SIZE
        for i in range(batches):
            batch = training_set.train.next_batch(BATCH_SIZE)
            batch_images = batch[0].reshape(BATCH_SIZE, 784)
            batch_images = batch_images * 2 - 1
            batch_seed = np.random.uniform(-1, 1, size=(BATCH_SIZE, 100))
            _ = s.run(
                d_trainer, feed_dict={real_images: batch_images, seed: batch_seed}
            )
            _ = s.run(g_trainer, feed_dict={seed: batch_seed})

        print("on epoch{}".format(epoch))

        sample_seed = np.random.uniform(-1, 1, size=(1, 100))
        gen_sample = s.run(generator(seed, reuse=True), feed_dict={seed: sample_seed})

        samples.append(gen_sample)

plt.imshow(samples[0].reshape(28, 28))
plt.imshow(samples[99].reshape(28, 28))
