from __future__ import absolute_import

import os
import tensorflow as tf
import numpy as np
import random
import math

def conv2d(inputs, filters, strides, padding):
    """
    Performs 2D convolution given 4D inputs and filter Tensors.
    :param inputs: tensor with shape [num_examples, in_height, in_width, in_channels]
    :param filters: tensor with shape [filter_height, filter_width, in_channels, out_channels]
    :param strides: MUST BE [1, 1, 1, 1] - list of strides, with each stride corresponding to each dimension in input
    :param padding: either "SAME" or "VALID", capitalization matters
    :return: outputs, NumPy array or Tensor with shape [num_examples, output_height, output_width, output_channels]
    """
    input_size = inputs.shape
    num_examples = input_size[0]
    in_height = input_size[1]
    in_width = input_size[2]
    input_in_channels = input_size[3]

    filter_size = filters.shape
    filter_height = filter_size[0]
    filter_width = filter_size[1]
    filter_in_channels = filter_size[2]
    filter_out_channels = filter_size[3]

    num_examples_stride = 1
    strideY = 1
    strideX = 1
    channels_stride = 1
    
    
    padX = 0
    padY = 0
    
    # Cleaning padding input
    output_height = (in_height - filter_height) + 1
    output_width = in_width - filter_width + 1
    if padding == "SAME":
        padX = int(np.floor((filter_width - 1) / 2))
        padY = int(np.floor((filter_width - 1) / 2))
        output_height = in_height - filter_height + 2 * padY + 1
        output_width = in_width - filter_width + 2 * padX + 1
        paddings = tf.constant([[0, 0], [padX, padX], [padY, padY], [0, 0]])
        inputs = tf.pad(inputs, paddings, 'CONSTANT')

    # Calculate output dimensions
    output_channels = filter_out_channels
    
    g = np.zeros([num_examples, output_height, output_width, output_channels])
    for im_num in range(num_examples):
        for h in range(output_height):
            for w in range(output_width):
                for oc in range(filter_out_channels):
                    
                    imgPatch = (inputs[im_num])[h: h + filter_height, w: w + filter_width]
                    #print(imgPatch.shape)
                    g[im_num, h, w, oc] = tf.tensordot(imgPatch, filters[:,:,:,oc], 
                     [[0, 1, 2], [0, 1, 2]]) 
    return g
                

def same_test_0():
    '''
    Simple test using SAME padding to check out differences between 
    own convolution function and TensorFlow's convolution function.

    NOTE: DO NOT EDIT
    '''
    imgs = np.array([[2,2,3,3,3],[0,1,3,0,3],[2,3,0,1,3],[3,3,2,1,2],[3,3,0,2,3]], dtype=np.float32)
    imgs = np.reshape(imgs, (1,5,5,1))
    filters = tf.Variable(tf.random.truncated_normal([2, 2, 1, 1],
                                dtype=tf.float32,
                                stddev=1e-1),
                                name="filters")
    my_conv = conv2d(imgs, filters, strides=[1, 1, 1, 1], padding="SAME")
    tf_conv = tf.nn.conv2d(imgs, filters, [1, 1, 1, 1], padding="SAME")
    print("SAME_TEST_0:", "my conv2d:", my_conv[0][0][0], "tf conv2d:", tf_conv[0][0][0].numpy())

def valid_test_0():
    '''
    Simple test using VALID padding to check out differences between 
    own convolution function and TensorFlow's convolution function.

    NOTE: DO NOT EDIT
    '''
    imgs = np.array([[2,2,3,3,3],[0,1,3,0,3],[2,3,0,1,3],[3,3,2,1,2],[3,3,0,2,3]], dtype=np.float32)
    imgs = np.reshape(imgs, (1,5,5,1))
    filters = tf.Variable(tf.random.truncated_normal([2, 2, 1, 1],
                                dtype=tf.float32,
                                stddev=1e-1),
                                name="filters")
    my_conv = conv2d(imgs, filters, strides=[1, 1, 1, 1], padding="VALID")
    tf_conv = tf.nn.conv2d(imgs, filters, [1, 1, 1, 1], padding="VALID")
    print("VALID_TEST_0:", "my conv2d:", my_conv[0][0], "tf conv2d:", tf_conv[0][0].numpy())

def valid_test_1():
    '''
    Simple test using VALID padding to check out differences between 
    own convolution function and TensorFlow's convolution function.

    NOTE: DO NOT EDIT
    '''
    imgs = np.array([[3,5,3,3],[5,1,4,5],[2,5,0,1],[3,3,2,1]], dtype=np.float32)
    imgs = np.reshape(imgs, (1,4,4,1))
    filters = tf.Variable(tf.random.truncated_normal([3, 3, 1, 1],
                                dtype=tf.float32,
                                stddev=1e-1),
                                name="filters")
    my_conv = conv2d(imgs, filters, strides=[1, 1, 1, 1], padding="VALID")
    tf_conv = tf.nn.conv2d(imgs, filters, [1, 1, 1, 1], padding="VALID")
    print("VALID_TEST_1:", "my conv2d:", my_conv[0][0], "tf conv2d:", tf_conv[0][0].numpy())

def valid_test_2():
    '''
    Simple test using VALID padding to check out differences between 
    own convolution function and TensorFlow's convolution function.

    NOTE: DO NOT EDIT
    '''
    imgs = np.array([[1,3,2,1],[1,3,3,1],[2,1,1,3],[3,2,3,3]], dtype=np.float32)
    imgs = np.reshape(imgs, (1,4,4,1))
    filters = np.array([[1,2,3],[0,1,0],[2,1,2]]).reshape((3,3,1,1)).astype(np.float32)
    my_conv = conv2d(imgs, filters, strides=[1, 1, 1, 1], padding="VALID")
    tf_conv = tf.nn.conv2d(imgs, filters, [1, 1, 1, 1], padding="VALID")
    print("VALID_TEST_1:", "my conv2d:", my_conv[0][0], "tf conv2d:", tf_conv[0][0].numpy())
    


def main():
    # TODO: Add in any tests you may want to use to view the differences between your and TensorFlow's output
    valid_test_0()
    valid_test_1()
    valid_test_2()
    same_test_0()
    return

if __name__ == '__main__':
    main()
