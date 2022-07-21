from __future__ import absolute_import
from matplotlib import pyplot as plt
from preprocess import get_data
from convolution import conv2d

import os
import tensorflow as tf
import numpy as np
import random

class Model(tf.keras.Model):
    def __init__(self):
        """
    This model class will contain the architecture for your CNN that
        classifies images. Do not modify the constructor, as doing so
        will break the autograder. We have left in variables in the constructor
        for you to fill out, but you are welcome to change them if you'd like.
        """
        super(Model, self).__init__()

        self.batch_size = 64
        self.num_classes = 2

        # TODO: Initialize all hyperparameters

        self.epsilon = tf.math.exp(tf.cast(-3, tf.float32))
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.epochs = 15



        # TODO: Initialize all trainable parameters

        self.filter1 = tf.Variable(tf.random.truncated_normal(shape = [5, 5, 3, 16], stddev = 0.1,
                                                              dtype = tf.float32))
        self.filter1_b = tf.Variable(tf.random.truncated_normal([16]))

        self.filter2 = tf.Variable(tf.random.truncated_normal(shape = [5, 5, 16, 20], stddev = 0.1,
                                                              dtype = tf.float32))
        self.filter2_b = tf.Variable(tf.random.truncated_normal([20]))

        self.filter3 = tf.Variable(tf.random.truncated_normal(shape = [5, 5, 20, 20], stddev = 0.1,
                                                              dtype = tf.float32))
        self.filter3_b = tf.Variable(tf.random.truncated_normal([20]))

        self.W1 = tf.Variable(tf.random.truncated_normal(shape = [4 * 4 * 20, 20], stddev = 0.1,
                                                         dtype = tf.float32))
        self.b1 = tf.Variable(tf.random.truncated_normal(shape = [20], stddev = 0.1, dtype=tf.float32))

        self.W2 = tf.Variable(tf.random.truncated_normal(shape = [20, 4 * 4 * 20], stddev = 0.1,
                                                         dtype = tf.float32))
        self.b2 = tf.Variable(tf.random.truncated_normal(shape = [4 * 4 * 20],
                                                         stddev = 0.1, dtype=tf.float32))

        self.W3 = tf.Variable(tf.random.truncated_normal(shape = [4 * 4 * 20, 2], stddev = 0.1,
                                                         dtype = tf.float32))
        self.b3 = tf.Variable(tf.random.truncated_normal(shape = [2], stddev = 0.1, dtype=tf.float32))


    def call(self, inputs, is_testing=False):
        """
        Runs a forward pass on an input batch of images.
        :param inputs: images, shape of (num_inputs, 32, 32, 3); during training, the shape is (batch_size, 32, 32, 3)
        :param is_testing: a boolean that should be set to True only when you're doing Part 2 of the assignment and this function is being called during testing
        :return: logits - a matrix of shape (num_inputs, num_classes); during training, it would be (batch_size, 2)
        """
        # Remember that
        # shape of input = (num_inputs (or batch_size), in_height, in_width, in_channels)
        # shape of filter = (filter_height, filter_width, in_channels, out_channels)
        # shape of strides = (batch_stride, height_stride, width_stride, channels_stride)


        conv1 = tf.nn.conv2d(inputs, self.filter1, [1, 2, 2, 1], "SAME")
        conv1b = tf.nn.bias_add(conv1, self.filter1_b)

        mean1, var1 = tf.nn.moments(conv1b, axes = [0, 1, 2])
        norm1 = tf.nn.batch_normalization(conv1b, mean1, var1, None, None, self.epsilon)

        max_pool1 = tf.nn.max_pool(tf.nn.relu(norm1), [3, 3], [1, 2, 2, 1], "SAME")


        conv2 = tf.nn.conv2d(max_pool1, self.filter2, [1, 1, 1, 1], "SAME")
        conv2b = tf.nn.bias_add(conv2, self.filter2_b)

        mean2, var2 = tf.nn.moments(conv2b, axes = [0, 1, 2])
        norm2 = tf.nn.batch_normalization(conv2b, mean2, var2, None, None, self.epsilon)

        max_pool2 = tf.nn.max_pool(tf.nn.relu(norm2), [2, 2], [1, 2, 2, 1], "SAME")

        if is_testing:
            conv3 = conv2d(max_pool2, self.filter3, [1, 1, 1, 1], "SAME")
        else:
            conv3 = tf.nn.conv2d(max_pool2, self.filter3, [1, 1, 1, 1], "SAME")
        conv3b = tf.nn.bias_add(conv3, self.filter3_b)

        mean3, var3 = tf.nn.moments(conv3b, axes = [0, 1, 2])
        norm3 = tf.nn.batch_normalization(conv3b, mean3, var3, None, None, self.epsilon)

        flattened = tf.reshape(tf.nn.relu(norm3), [-1, 4 * 4 * 20])

        dense1 = tf.matmul(flattened, self.W1) + self.b1
        drop1 = tf.nn.dropout(tf.nn.relu(dense1), 0.3)

        dense2 = tf.matmul(drop1, self.W2) + self.b2
        drop2 = tf.nn.dropout(tf.nn.relu(dense2), 0.3)

        dense3 = tf.matmul(drop2, self.W3) + self.b3
        return dense3




    def loss(self, logits, labels):
        """
        Calculates the model cross-entropy loss after one forward pass.
        :param logits: during training, a matrix of shape (batch_size, self.num_classes)
        containing the result of multiple convolution and feed forward layers
        Softmax is applied in this function.
        :param labels: during training, matrix of shape (batch_size, self.num_classes) containing the train labels
        :return: the loss of the model as a Tensor
        """
        #print(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels, logits)))
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels, logits))


    def accuracy(self, logits, labels):
        """
        Calculates the model's prediction accuracy by comparing
        logits to correct labels – no need to modify this.
        :param logits: a matrix of size (num_inputs, self.num_classes); during training, this will be (batch_size, self.num_classes)
        containing the result of multiple convolution and feed forward layers
        :param labels: matrix of size (num_labels, self.num_classes) containing the answers, during training, this will be (batch_size, self.num_classes)

        NOTE: DO NOT EDIT

        :return: the accuracy of the model as a Tensor
        """
        correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

def train(model, train_inputs, train_labels):
    '''
    Trains the model on all of the inputs and labels for one epoch. You should shuffle your inputs
    and labels - ensure that they are shuffled in the same order using tf.gather.
    To increase accuracy, you may want to use tf.image.random_flip_left_right on your
    inputs before doing the forward pass. You should batch your inputs.
    :param model: the initialized model to use for the forward pass and backward pass
    :param train_inputs: train inputs (all inputs to use for training),
    shape (num_inputs, width, height, num_channels)
    :param train_labels: train labels (all labels to use for training),
    shape (num_labels, num_classes)
    :return: None
    '''


    rand_ind = tf.random.shuffle(range(len(train_labels)))
    shuffled_inputs = tf.gather(train_inputs, rand_ind)
    shuffled_labels = tf.gather(train_labels, rand_ind)
    for i in range(0, len(train_labels), model.batch_size):
        with tf.GradientTape() as tape:
            num_ex = len(shuffled_inputs[i:i+model.batch_size])
            inputs = tf.reshape(shuffled_inputs[i:i+model.batch_size], [num_ex, 32, 32, 3])
            inputs = tf.image.random_flip_left_right(inputs)
            inputs = tf.clip_by_value(inputs * tf.random.uniform(shape = tf.shape(inputs), maxval = 2.), 0 , 1)
            logits = model.call(inputs)
            loss = model.loss(logits, shuffled_labels[i:i+model.batch_size])
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))




def test(model, test_inputs, test_labels):
    """
    Tests the model on the test inputs and labels. You should NOT randomly
    flip images or do any extra preprocessing.
    :param test_inputs: test data (all images to be tested),
    shape (num_inputs, width, height, num_channels)
    :param test_labels: test labels (all corresponding labels),
    shape (num_labels, num_classes)
    :return: test accuracy - this can be the average accuracy across
    all batches or the sum as long as you eventually divide it by batch_size
    """

    logits = model.call(test_inputs, is_testing = True)
    return model.accuracy(logits, test_labels)


def visualize_results(image_inputs, probabilities, image_labels, first_label, second_label):
    """
    Uses Matplotlib to visualize the results of our model.
    :param image_inputs: image data from get_data(), limited to 10 images, shape (10, 32, 32, 3)
    :param probabilities: the output of model.call(), shape (10, num_classes)
    :param image_labels: the labels from get_data(), shape (10, num_classes)
    :param first_label: the name of the first class, "dog"
    :param second_label: the name of the second class, "cat"

    NOTE: DO NOT EDIT

    :return: doesn't return anything, a plot should pop-up
    """
    predicted_labels = np.argmax(probabilities, axis=1)
    num_images = image_inputs.shape[0]

    fig, axs = plt.subplots(ncols=num_images)
    fig.suptitle("PL = Predicted Label\nAL = Actual Label")
    for ind, ax in enumerate(axs):
            ax.imshow(image_inputs[ind], cmap="Greys")
            pl = first_label if predicted_labels[ind] == 0.0 else second_label
            al = first_label if np.argmax(image_labels[ind], axis=0) == 0 else second_label
            ax.set(title="PL: {}\nAL: {}".format(pl, al))
            plt.setp(ax.get_xticklabels(), visible=False)
            plt.setp(ax.get_yticklabels(), visible=False)
            ax.tick_params(axis='both', which='both', length=0)
    plt.show()


def main():
    '''
    Read in CIFAR10 data (limited to 2 classes), initialize your model, and train and
    test your model for a number of epochs. We recommend that you train for
    10 epochs and at most 25 epochs. For CS2470 students, you must train within 10 epochs.
    You should receive a final accuracy on the testing examples for cat and dog of >=70%.
    :return: None
    '''
    train_i, train_l = get_data("CIFAR_data_compressed/train", 5, 3)
    test_i, test_l = get_data("CIFAR_data_compressed/test", 5, 3)

    model = Model()

    for i in range(model.epochs):
        print(i)
        train(model, train_i, train_l)

    print(test(model, test_i, test_l))

    visualize_results(test_i[:5], model.call(test_i)[:5], test_l[:5], 5, 3)



if __name__ == '__main__':
    main()
