#coding:utf-8
from rnet.RNet_models import R_Net
from rnet.train import train
import sys
import os


def train_RNet(base_dir, prefix, end_epoch, logDir, display, lr):
    """
    train PNet
    :param dataset_dir: tfrecord path
    :param prefix:
    :param end_epoch:
    :param display:
    :param lr:
    :return:
    """
    net_factory = R_Net
    train(net_factory, prefix, end_epoch, base_dir, logDir, display=display, base_lr=lr)


if __name__ == '__main__':
    base_dir = '/nfs/data/DRG/fz'
    model_name = 'RNet0801'
    model_path = '/nfs/data/DRG/fz/face_classfy_model/08011734/%s' % model_name
    prefix = model_path
    logDir = str(sys.argv[-2])
    print(logDir)

    end_epoch = 44
    display = 100
    lr = 0.001
    # os.system('tree /nfs/data/DRG/fz/')
    # size = os.path.getsize(r'/nfs/data/DRG/fz/data/RNet0801.tfrecord_shuffle')
    # print(size)
    train_RNet(base_dir, prefix, end_epoch, logDir, display, lr)
