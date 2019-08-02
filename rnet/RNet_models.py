import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.contrib.tensorboard.plugins import projector
import numpy as np

num_keep_radio = 0.73       #0.7


# define prelu
def prelu(inputs):
    alphas = tf.get_variable("alphas", shape=inputs.get_shape()[-1], dtype=tf.float32, initializer=tf.constant_initializer(0.25))
    pos = tf.nn.relu(inputs)
    neg = alphas * (inputs-abs(inputs))*0.5
    return pos + neg


# cls_prob:batch*2
# label:batch
def cls_ohem(cls_prob, label, cls_num):
    zeros = tf.zeros_like(label)
    # label=-1 --> label=0net_factory
    # pos -> 1, neg -> 0, others -> 0
    # label_filter_invalid = tf.where(tf.less(label, 0), zeros, label)
    num_cls_prob = tf.size(cls_prob)
    cls_prob_reshape = tf.reshape(cls_prob, [num_cls_prob, -1])
    label_int = tf.cast(label, tf.int32)

    # get the number of rows of class_prob
    num_row = tf.to_int32(cls_prob.get_shape()[0])

    #row = [0,2,4.....]
    # row = tf.range(num_row)*2
    row = tf.range(num_row) * cls_num   # cls_num分几类
    indices_ = row + label_int
    label_prob = tf.squeeze(tf.gather(cls_prob_reshape, indices_))
    loss = -tf.log(label_prob+1e-10)
    zeros = tf.zeros_like(label_prob, dtype=tf.float32)
    ones = tf.ones_like(label_prob, dtype=tf.float32)

    # set pos and neg to be 1, rest to be 0
    valid_inds = tf.where(label < zeros, zeros, ones)
    # get the number of POS and NEG examples
    num_valid = tf.reduce_sum(valid_inds)
    keep_num = tf.cast(num_valid*num_keep_radio, dtype=tf.int32)
    # FILTER OUT PART AND LANDMARK DATA
    loss = loss * valid_inds
    loss, _ = tf.nn.top_k(loss, k=keep_num)
    return tf.reduce_mean(loss)


def cal_accuracy(cls_prob, label):

    '''
    :param cls_prob:
    :param label:
    :return:calculate classification accuracy for pos and neg examples only
    '''
    # get the index of maximum value along axis one from cls_prob
    # 0 for negative 1 for positive
    pred = tf.argmax(cls_prob, axis=1)
    label_int = tf.cast(label, tf.int64)
    # return the index of pos and neg examples
    cond = tf.where(tf.greater_equal(label_int, 0))
    picked = tf.squeeze(cond)
    # gather the label of pos and neg examples
    label_picked = tf.gather(label_int, picked)
    pred_picked = tf.gather(pred, picked)
    # calculate the mean value of rnet vector contains 1 and 0, 1 for correct classification, 0 for incorrect
    # ACC = (TP+FP)/total population
    accuracy_op = tf.reduce_mean(tf.cast(tf.equal(label_picked,pred_picked), tf.float32))
    return accuracy_op


def R_Net(inputs, label_hat=None, label_mask=None, label_block=None, label_blur=None, label_bow=None, label_illumination=None, training=True):
    with slim.arg_scope([slim.conv2d],
                        activation_fn=prelu,
                        weights_initializer=slim.xavier_initializer(),
                        biases_initializer=tf.zeros_initializer(),
                        weights_regularizer=slim.l2_regularizer(0.0005),
                        padding='valid'):
        print(inputs.get_shape())
        net = slim.conv2d(inputs, num_outputs=28, kernel_size=[3, 3], stride=1, scope="conv1")
        print(net.get_shape())
        net = slim.max_pool2d(net, kernel_size=[3, 3], stride=2, scope="pool1", padding='SAME')
        print(net.get_shape())
        net = slim.conv2d(net, num_outputs=48, kernel_size=[3, 3], stride=1, scope="conv2")
        print(net.get_shape())
        net = slim.max_pool2d(net, kernel_size=[3, 3], stride=2, scope="pool2")
        print(net.get_shape())
        net = slim.conv2d(net, num_outputs=64, kernel_size=[2, 2], stride=1, scope="conv3")
        print(net.get_shape())
        fc_flatten = slim.flatten(net)
        print(fc_flatten.get_shape())
        fc1 = slim.fully_connected(fc_flatten, num_outputs=128, scope="fc1")
        print(fc1.get_shape())
        # batch*2
        cls_hat_prob = slim.fully_connected(fc1, num_outputs=2, scope="cls_fc_hat", activation_fn=tf.nn.softmax)
        print(cls_hat_prob.get_shape())
        cls_mask_prob = slim.fully_connected(fc1, num_outputs=2, scope="cls_fc_mask", activation_fn=tf.nn.softmax)
        print(cls_mask_prob.get_shape())
        cls_block_prob = slim.fully_connected(fc1, num_outputs=2, scope="cls_fc_block", activation_fn=tf.nn.softmax)
        print(cls_block_prob.get_shape())
        cls_blur_prob = slim.fully_connected(fc1, num_outputs=2, scope="cls_fc_blur", activation_fn=tf.nn.softmax)
        print(cls_blur_prob.get_shape())
        cls_bow_prob = slim.fully_connected(fc1, num_outputs=2, scope="cls_fc_bow", activation_fn=tf.nn.softmax)
        print(cls_bow_prob.get_shape())
        cls_illumination_prob = slim.fully_connected(fc1, num_outputs=2, scope="cls_fc_illumination", activation_fn=tf.nn.softmax)
        print(cls_illumination_prob.get_shape())

        #train
        if training:
            loss_hat = cls_ohem(cls_hat_prob, label_hat, 2)
            accuracy_hat = cal_accuracy(cls_hat_prob, label_hat)

            loss_mask = cls_ohem(cls_mask_prob, label_mask, 2)
            accuracy_mask = cal_accuracy(cls_mask_prob, label_mask)

            loss_block = cls_ohem(cls_block_prob, label_block, 2)
            accuracy_block = cal_accuracy(cls_block_prob, label_block)

            loss_blur = cls_ohem(cls_blur_prob, label_blur, 2)
            accuracy_blur = cal_accuracy(cls_blur_prob, label_blur)

            loss_bow = cls_ohem(cls_bow_prob, label_bow, 2)
            accuracy_bow = cal_accuracy(cls_bow_prob, label_bow)

            loss_illumination = cls_ohem(cls_illumination_prob, label_illumination, 2)
            accuracy_illumination = cal_accuracy(cls_illumination_prob, label_illumination)

            L2_loss = tf.add_n(slim.losses.get_regularization_losses())
            return loss_hat, accuracy_hat, loss_mask, accuracy_mask, loss_block, accuracy_block, loss_blur, \
                   accuracy_blur, loss_bow, accuracy_bow, loss_illumination, accuracy_illumination, L2_loss
        else:
            return cls_hat_prob, cls_mask_prob, cls_block_prob, cls_blur_prob, cls_bow_prob, cls_illumination_prob
