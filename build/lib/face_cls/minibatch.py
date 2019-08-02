import cv2
import numpy as np


def get_minibatch(imdb, num_classes, im_size):
    # im_size: 12, 24 or 48
    num_images = len(imdb)
    processed_ims = list()
    cls_hat_label = list()
    cls_mask_label = list()
    cls_block_label = list()
    cls_blur_label = list()
    cls_bow_label = list()
    cls_illumination_label = list()

    # bbox_reg_target = list()
    for i in range(num_images):
        im = cv2.imread(imdb[i]['image'])
        h, w, c = im.shape
        cls_hat = imdb[i]['label_hat']
        cls_mask = imdb[i]['label_mask']
        cls_block = imdb[i]['label_block']
        cls_blur = imdb[i]['label_blur']
        cls_bow = imdb[i]['label_bow']
        cls_illumination = imdb[i]['label_illumination']
        # bbox_target = imdb[i]['bbox_target']

        assert h == w == im_size, "image size wrong"
        if imdb[i]['flipped']:
            im = im[:, ::-1, :]

        im_tensor = im / 127.5
        processed_ims.append(im_tensor)
        cls_hat_label.append(cls_hat)
        cls_mask_label.append(cls_mask)
        cls_block_label.append(cls_block)
        cls_blur_label.append(cls_blur)
        cls_bow_label.append(cls_bow)
        cls_illumination_label.append(cls_illumination)
        # bbox_reg_target.append(bbox_target)

    im_array = np.asarray(processed_ims)
    label_hat_array = np.array(cls_hat_label)
    label_mask_array = np.array(cls_mask_label)
    label_block_array = np.array(cls_block_label)
    label_blur_array = np.array(cls_blur_label)
    label_bow_array = np.array(cls_bow_label)
    label_illumination_array = np.array(cls_illumination_label)
    # bbox_target_array = np.vstack(bbox_reg_target)
    '''
    bbox_reg_weight = np.ones(label_array.shape)
    invalid = np.where(label_array == 0)[0]
    bbox_reg_weight[invalid] = 0
    bbox_reg_weight = np.repeat(bbox_reg_weight, 4, axis=1)
    '''

    data = {'data': im_array}
    label = {'label_hat': label_hat_array, 'label_mask': label_mask_array, 'label_block': label_block_array,
             'label_blur': label_blur_array, 'label_bow': label_bow_array, 'label_illumination': label_illumination_array}

    return data, label


def get_testbatch(imdb):
    # print(len(imdb))
    assert len(imdb) == 1, "Single batch only"
    # im = cv2.imread(imdb[0])
    im = cv2.imread(imdb)
    im_array = im
    data = {'data': im_array}
    return data
