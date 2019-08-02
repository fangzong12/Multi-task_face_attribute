# coding:utf-8

import sys
import zipfile

sys.path.append('..')

from face_cls.cls import rnet_cls
from face_cls.predict import Predict
from face_cls.RNet_models import *
from face_cls.dataLoader import TestLoader
import cv2
import os
import numpy as np

if __name__ == '__main__':
    test_mode = "RNet"
    thresh = [0.6, 0.7, 0.7]
    min_face_size = 20
    stride = 2
    slide_window = False
    shuffle = False
    nets = [None]
    prefix = ['/nfs/data/DRG/fz/face_classfy_model/08011734/RNet0801']
    epoch = [30, 14, 16]
    batch_size = [2048, 64, 16]
    model_path = ['%s-%s' % (x, y) for x, y in zip(prefix, epoch)]

    # RNet = Predict(R_Net, 24, batch_size[1], model_path[0])
    RNet = Predict(R_Net, 24, batch_size[1], model_path[0])
    nets[0] = RNet
    rnetCls = rnet_cls(cls_nets=nets, min_face_size=min_face_size,
                       stride=stride, threshold=thresh, slide_window=slide_window)
    gt_imdb = []
    # gt_imdb.append("35_Basketball_Basketball_35_515.jpg")
    # imdb_ = dict()"
    # imdb_['image'] = im_path
    # imdb_['label'] = 5
    hat_labels = []
    mask_labels = []
    block_labels = []
    blur_labels = []
    bow_labels = []
    illumination_labels = []

    zip_file = '/nfs/data/DRG/fz/data/val_celebA.zip'
    unzip_dir = '/nfs/data/DRG/fz/data'
    image_dir = '/nfs/data/DRG/fz/data /val_celebA'
    path = '/nfs/data/DRG/fz/data/val_RNet_full.txt'
    # os.system('tree /nfs/data/DRG/fz/data/val_celebA')
    # zip_tool = zipfile.ZipFile(zip_file, 'r')
    # zip_tool.extractall(unzip_dir)
    # zip_tool.close()
    # os.system('tree /nfs/data/DRG/fz/data/val_celebA')

    reader = open(path, 'r')
    line = reader.readline().strip()
    while line:
        item = line.split(' ')
        gt_imdb.append(os.path.join(image_dir, item[0]))
        hat_labels.append(int(item[1]))
        mask_labels.append(int(item[2]))
        block_labels.append(int(item[3]))
        blur_labels.append(int(item[4]))
        bow_labels.append(int(item[5]))
        illumination_labels.append(int(item[6]))
        line = reader.readline().strip()
        # if not os.path.exists(os.path.join(image_dir, item[0])):
        #     print(os.path.join(image_dir, item[0]))
    test_data = TestLoader(gt_imdb)

    hat_probs, mask_probs, block_probs, blur_probs, bow_probs, illumination_probs = rnetCls.cls_face(test_data)

    cls_hat_prob = tf.convert_to_tensor(hat_probs, tf.float32, name='cls_hat_probs')
    cls_mask_prob = tf.convert_to_tensor(mask_probs, tf.float32, name='cls_mask_probs')
    cls_block_prob = tf.convert_to_tensor(block_probs, tf.float32, name='cls_block_probs')
    cls_blur_prob = tf.convert_to_tensor(blur_probs, tf.float32, name='cls_blur_probs')
    cls_bow_prob = tf.convert_to_tensor(bow_probs, tf.float32, name='cls_bow_probs')
    cls_illumination_prob = tf.convert_to_tensor(illumination_probs, tf.float32, name='cls_illumination_probs')

    # print(cls_prob)
    # cls_probs = np.squeeze(np.array(cls_probs))
    # print(labels)
    reader.close()

    cls_hat_probs = tf.squeeze(cls_hat_prob)
    hat_acc = cal_accuracy(cls_hat_probs, hat_labels)
    hat_loss = cls_ohem(cls_hat_probs, hat_labels, 2)

    cls_mask_probs = tf.squeeze(cls_mask_prob)
    mask_acc = cal_accuracy(cls_mask_probs, mask_labels)
    mask_loss = cls_ohem(cls_mask_probs, mask_labels, 2)

    cls_block_probs = tf.squeeze(cls_block_prob)
    block_acc = cal_accuracy(cls_block_probs, block_labels)
    block_loss = cls_ohem(cls_block_probs, block_labels, 2)

    cls_blur_probs = tf.squeeze(cls_blur_prob)
    blur_acc = cal_accuracy(cls_blur_probs, blur_labels)
    blur_loss = cls_ohem(cls_blur_probs, blur_labels, 2)

    cls_bow_probs = tf.squeeze(cls_bow_prob)
    bow_acc = cal_accuracy(cls_bow_probs, bow_labels)
    bow_loss = cls_ohem(cls_bow_probs, bow_labels, 2)

    cls_illumination_probs = tf.squeeze(cls_illumination_prob)
    illumination_acc = cal_accuracy(cls_illumination_probs, illumination_labels)
    illumination_loss = cls_ohem(cls_illumination_probs, illumination_labels, 2)

    sess = tf.Session()
    # print(sess.run(cls_prob))
    print('hat_loss is %4f, ' % sess.run(hat_loss))
    print('hat_acc is %4f\n' % sess.run(hat_acc))

    print('mask_loss is %4f, ' % sess.run(mask_loss))
    print('mask_acc is %4f\n' % sess.run(mask_acc))

    print('block_loss is %4f, ' % sess.run(block_loss))
    print('block_acc is %4f\n' % sess.run(block_acc))

    print('blur_loss is %4f, ' % sess.run(blur_loss))
    print('blur_acc is %4f\n' % sess.run(blur_acc))

    print('bow_loss is %4f, ' % sess.run(bow_loss))
    print('bow_acc is %4f\n' % sess.run(bow_acc))

    print('illumination_loss is %4f, ' % sess.run(illumination_loss))
    print('illumination_acc is %4f\n' % sess.run(illumination_acc))
