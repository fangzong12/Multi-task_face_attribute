# coding:utf-8
import os
from datetime import datetime
import tensorflow as tf
from tensorboard.plugins import projector
from rnet.read_tfrecord import read_single_tfrecord

BATCH_SIZE = 384                  # 384
CLS_OHEM = True
CLS_OHEM_RATIO = 0.7
EPS = 1e-14
LR_EPOCH = [6, 14, 20]


def train_model(base_lr, loss, data_num):
    """
    train model
    :param base_lr: base learning rate
    :param loss: loss
    :param data_num:
    :return:
    train_op, lr_op

    """
    lr_factor = 0.1
    global_step = tf.Variable(0, trainable=False)
    # LR_EPOCH [8,14]
    # boundaried [num_batch,num_batch]
    boundaries = [int(epoch * data_num / BATCH_SIZE) for epoch in LR_EPOCH]
    # lr_values[0.01,0.001,0.0001,0.00001]
    lr_values = [base_lr * (lr_factor ** x) for x in range(0, len(LR_EPOCH) + 1)]
    # control learning rate
    lr_op = tf.train.piecewise_constant(global_step, boundaries, lr_values)
    optimizer = tf.train.MomentumOptimizer(lr_op, 0.9)
    train_op = optimizer.minimize(loss, global_step)
    return train_op, lr_op


def image_color_distort(inputs):
    inputs = tf.image.random_contrast(inputs, lower=0.5, upper=1.5)
    inputs = tf.image.random_brightness(inputs, max_delta=0.2)
    inputs = tf.image.random_hue(inputs, max_delta=0.2)
    inputs = tf.image.random_saturation(inputs, lower=0.5, upper=1.5)
    return inputs


def train(net_factory, prefix, end_epoch, base_dir, logs_dir,
          display=200, base_lr=0.01):
    """
    train PNet/RNet/ONet
    :param net_factory:
    :param prefix: model path
    :param end_epoch:
    :param dataset:
    :param display:
    :param base_lr:
    :return:
    """
    if not os.path.exists(prefix):
        os.makedirs(prefix)
    net = prefix.split('/')[-1]
    # label file
    label_file = os.path.join(base_dir, 'data/train_RNet_full.txt')
    #label_file = os.path.join(base_dir,'landmark_12_few.txt')
    print(label_file)
    f = open(label_file, 'r')
    # get number of training examples
    num = len(f.readlines())
    f.close()

   # num = 143348
    print("Total size of the dataset is: ", num)
    print(prefix)

    # dataset_dir = os.path.join(base_dir,'train_%s_ALL.tfrecord_shuffle' % net)
    dataset_dir = os.path.join(base_dir, 'data/RNet0801.tfrecord_shuffle')
    print('dataset dir is:', dataset_dir)
    image_batch, label_hat_batch, label_mask_batch, label_block_batch, label_blur_batch, label_bow_batch,\
    label_illumination_batch = read_single_tfrecord(dataset_dir, BATCH_SIZE, net)

        # landmark_dir
    image_size = 24
    radio_cls_loss = 1.0
    # radio_bbox_loss = 0.5
    # radio_landmark_loss = 0.5

    # define placeholder
    input_image = tf.placeholder(tf.float32, shape=[BATCH_SIZE, image_size, image_size, 3], name='input_image')
    label_hat = tf.placeholder(tf.float32, shape=[BATCH_SIZE], name='label_hat')
    label_mask = tf.placeholder(tf.float32, shape=[BATCH_SIZE], name='label_mask')
    label_block = tf.placeholder(tf.float32, shape=[BATCH_SIZE], name='label_block')
    label_blur = tf.placeholder(tf.float32, shape=[BATCH_SIZE], name='label_blur')
    label_bow = tf.placeholder(tf.float32, shape=[BATCH_SIZE], name='label_bow')
    label_illumination = tf.placeholder(tf.float32, shape=[BATCH_SIZE], name='label_illumination')

    # bbox_target = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 4], name='bbox_target')
    # landmark_target = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 10], name='landmark_target')
    # get loss and accuracy
    input_image = image_color_distort(input_image)
    cls_hat_loss_op, accuracy_hat_op, cls_mask_loss_op, accuracy_mask_op, cls_block_loss_op, accuracy_block_op, \
    cls_blur_loss_op, accuracy_blur_op, cls_bow_loss_op, accuracy_bow_op, cls_illumination_loss_op, accuracy_illumination_op,\
    L2_loss_op= net_factory(input_image, label_hat, label_mask, label_block, label_blur,
                            label_bow, label_illumination, training=True)
    # train,update learning rate(3 loss)
    total_loss_op = radio_cls_loss * cls_hat_loss_op+radio_cls_loss * cls_mask_loss_op+radio_cls_loss * \
                    cls_block_loss_op+radio_cls_loss * cls_blur_loss_op+radio_cls_loss * cls_bow_loss_op+\
                    radio_cls_loss * cls_illumination_loss_op
    train_op, lr_op = train_model(base_lr,
                                  total_loss_op,
                                  num)
    # init
    init = tf.global_variables_initializer()
    sess = tf.Session()
    # save model
    saver = tf.train.Saver(max_to_keep=0)
    sess.run(init)
    # visualize some variables
    tf.summary.scalar("cls_hat_loss", cls_hat_loss_op)  # cls_loss
    tf.summary.scalar("accuracy_hat", accuracy_hat_op)  # cls_acc
    tf.summary.scalar("cls_mask_loss", cls_mask_loss_op)  # cls_loss
    tf.summary.scalar("accuracy_mask", accuracy_mask_op)  # cls_acc
    tf.summary.scalar("cls_block_loss", cls_block_loss_op)  # cls_loss
    tf.summary.scalar("accuracy_block", accuracy_block_op)  # cls_acc
    tf.summary.scalar("cls_blur_loss", cls_blur_loss_op)  # cls_loss
    tf.summary.scalar("accuracy_blur", accuracy_blur_op)  # cls_acc
    tf.summary.scalar("cls_bow_loss", cls_bow_loss_op)  # cls_loss
    tf.summary.scalar("accuracy_bow", accuracy_bow_op)  # cls_acc
    tf.summary.scalar("cls_illumination_loss", cls_illumination_loss_op)  # cls_loss
    tf.summary.scalar("accuracy_illumination", accuracy_illumination_op)  # cls_acc
    tf.summary.scalar("total_loss", total_loss_op)  # cls_loss, bbox loss, landmark loss and L2 loss add together
    summary_op = tf.summary.merge_all()
    # logs_dir = "logs07091613"
    # logs_dir = os.path.join(base_dir, logs_dir)
    # logs_dir = logdir
    print('tensorboard logs_dir is :'+logs_dir)
    if not os.path.exists(logs_dir):
        os.mkdir(logs_dir)

    writer = tf.summary.FileWriter(logs_dir, sess.graph)
    projector_config = projector.ProjectorConfig()
    projector.visualize_embeddings(writer, projector_config)
    # begin
    coord = tf.train.Coordinator()
    # begin enqueue thread
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    i = 0
    # total steps
    MAX_STEP = int(num / BATCH_SIZE + 1) * end_epoch
    epoch = 0
    sess.graph.finalize()
    try:
        for step in range(MAX_STEP):
            i = i + 1
            if coord.should_stop():
                break
            image_batch_array, label_hat_batch_array, label_mask_batch_array, label_block_batch_array, \
            label_blur_batch_array, label_bow_batch_array, label_illumination_batch_array = sess.run(
                [image_batch, label_hat_batch, label_mask_batch, label_block_batch,
                 label_blur_batch, label_bow_batch, label_illumination_batch])
            '''
            print('im here')
            print(image_batch_array.shape)
            print(label_batch_array.shape)
            print(bbox_batch_array.shape)
            print(landmark_batch_array.shape)
            print(label_batch_array[0])
            print(bbox_batch_array[0])
            print(landmark_batch_array[0])
            '''
            _, _, summary = sess.run([train_op, lr_op, summary_op],
                                     feed_dict={input_image: image_batch_array, label_hat: label_hat_batch_array,
                                                label_mask: label_mask_batch_array, label_block: label_block_batch_array,
                                                label_blur: label_blur_batch_array, label_bow: label_bow_batch_array,
                                                label_illumination: label_illumination_batch_array})
            if (step + 1) % display == 0:
                # acc = accuracy(cls_pred, labels_batch)
                cls_hat_loss, accuracy_hat, cls_mask_loss, accuracy_mask, cls_block_loss, \
                accuracy_block, cls_blur_loss, accuracy_blur, cls_bow_loss, accuracy_bow, \
                cls_illumination_loss, accuracy_illumination, L2_loss, lr = sess.run(
                    [cls_hat_loss_op, accuracy_hat_op, cls_mask_loss_op, accuracy_mask_op, cls_block_loss_op,
                      accuracy_block_op, cls_blur_loss_op, accuracy_blur_op, cls_bow_loss_op, accuracy_bow_op,
                      cls_illumination_loss_op, accuracy_illumination_op, L2_loss_op, lr_op],
                    feed_dict={input_image: image_batch_array, label_hat: label_hat_batch_array,
                                                label_mask: label_mask_batch_array, label_block: label_block_batch_array,
                                                label_blur: label_blur_batch_array, label_bow: label_bow_batch_array,
                                                label_illumination: label_illumination_batch_array})
                total_loss = radio_cls_loss * cls_hat_loss + radio_cls_loss * cls_mask_loss + radio_cls_loss * \
                    cls_block_loss + radio_cls_loss * cls_blur_loss + radio_cls_loss * cls_bow_loss + \
                    radio_cls_loss * cls_illumination_loss + L2_loss
                # landmark loss: %4f,
                print("%s : Step: %d/%d, accuracy_hat: %3f, cls_hat loss:"
                      " %4f,accuracy_mask: %3f, cls_mask loss:"
                      " %4f,accuracy_block: %3f, cls_block loss:"
                      " %4f,accuracy_blur: %3f, cls_blur loss:"
                      " %4f,accuracy_bow: %3f, cls_bow loss:"
                      " %4f,accuracy_illumination: %3f, cls_illumination loss:"
                      " %4f, L2 loss: %4f, Total Loss: %4f ,lr:%f " % (
                        datetime.now(), step + 1, MAX_STEP,
                        accuracy_hat, cls_hat_loss, accuracy_mask, cls_mask_loss, accuracy_block, cls_block_loss,
                        accuracy_blur, cls_blur_loss, accuracy_bow, cls_bow_loss,
                        accuracy_illumination, cls_illumination_loss,  L2_loss, total_loss, lr))
            # save every two epochs
            if i * BATCH_SIZE > num * 2:
                epoch = epoch + 1
                i = 0
                path_prefix = saver.save(sess, prefix, global_step=epoch * 2)
                print('path prefix is :', path_prefix)
            writer.add_summary(summary, global_step=step)
    except tf.errors.OutOfRangeError:
        print("完成！！！")
    finally:
        coord.request_stop()
        writer.close()
    coord.join(threads)
    sess.close()
