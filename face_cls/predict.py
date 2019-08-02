import tensorflow as tf
import numpy as np


class Predict(object):
    #net_factory:rnet or onet
    #datasize:24 or 48
    def __init__(self, net_factory, data_size, batch_size, model_path):
        graph = tf.Graph()
        with graph.as_default():
            self.image_op = tf.placeholder(tf.float32, shape=[batch_size, data_size, data_size, 3], name='input_image')
            #figure out landmark
            self.cls_hat_prob, self.cls_mask_prob, self.cls_block_prob, self.cls_blur_prob, \
            self.cls_bow_prob, self.cls_illumination_prob = net_factory(self.image_op, training=False)
            # self.sess = tf.Session(
            #     config=tf.ConfigProto(allow_soft_placement=True, gpu_options=tf.GPUOptions(allow_growth=True)))
            self.sess = tf.Session()
            saver = tf.train.Saver()
            #check whether the dictionary is valid
            model_dict = '/'.join(model_path.split('/')[:-1])
            ckpt = tf.train.get_checkpoint_state(model_dict)
            print(model_path)
            readstate = ckpt and ckpt.model_checkpoint_path
            assert  readstate, "the params dictionary is not valid"
            print("restore models' param")
            saver.restore(self.sess, model_path)

        self.data_size = data_size
        self.batch_size = batch_size

    #rnet and onet minibatch(test)
    def predict(self, databatch):
        # access data
        # databatch: N x 3 x data_size x data_size
        databatch = np.expand_dims(databatch,axis=0)
        scores = []
        batch_size = self.batch_size

        minibatch = []
        cur = 0
        #num of all_data
        n = databatch.shape[0]
        while cur < n:
            #split mini-batch
            # minibatch.append(databatch[cur:min(cur + batch_size, n), :, :, :])

            minibatch.append(databatch[cur:min(cur + batch_size, n)])
            cur += batch_size
        #every batch prediction result
        cls_hat_prob_list = []
        cls_mask_prob_list = []
        cls_block_prob_list = []
        cls_blur_prob_list = []
        cls_bow_prob_list = []
        cls_illumination_prob_list = []
        for idx, data in enumerate(minibatch):
            m = data.shape[0]
            real_size = self.batch_size
            #the last batch
            if m < batch_size:
                keep_inds = np.arange(m)
                #gap (difference)
                gap = self.batch_size - m
                while gap >= len(keep_inds):
                    gap -= len(keep_inds)
                    keep_inds = np.concatenate((keep_inds, keep_inds))
                if gap != 0:
                    keep_inds = np.concatenate((keep_inds, keep_inds[:gap]))
                data = data[keep_inds]
                real_size = m
            #cls_prob batch*2
            #bbox_pred batch*4
            cls_hat_prob, cls_mask_prob, cls_block_prob, cls_blur_prob, cls_bow_prob, cls_illumination_prob = \
                self.sess.run([self.cls_hat_prob, self.cls_mask_prob, self.cls_block_prob, self.cls_blur_prob,
                                 self.cls_bow_prob, self.cls_illumination_prob], feed_dict={self.image_op: data})
            #num_batch * batch_size *2
            cls_hat_prob_list.append(cls_hat_prob[:real_size])
            cls_mask_prob_list.append(cls_mask_prob[:real_size])
            cls_block_prob_list.append(cls_block_prob[:real_size])
            cls_blur_prob_list.append(cls_blur_prob[:real_size])
            cls_bow_prob_list.append(cls_bow_prob[:real_size])
            cls_illumination_prob_list.append(cls_illumination_prob[:real_size])

            #num_of_data*2,num_of_data*4,num_of_data*10
        return np.concatenate(cls_hat_prob_list, axis=0), np.concatenate(cls_mask_prob_list, axis=0),\
               np.concatenate(cls_block_prob_list, axis=0), np.concatenate(cls_blur_prob_list, axis=0), \
               np.concatenate(cls_bow_prob_list, axis=0), np.concatenate(cls_illumination_prob_list, axis=0)
