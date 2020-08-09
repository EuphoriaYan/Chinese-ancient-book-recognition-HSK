# -*- encoding: utf-8 -*-
# Author: hushukai

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import backend as K


class SegmentTarget(layers.Layer):
    def __init__(self, feat_stride=16, label_smoothing=0.1,
                 pos_weight=2., neg_weight=1., pad_weight=1., cls_score_thresh=0.7, segment_task="book_page", **kwargs):
        self.feat_stride = feat_stride
        self.label_smoothing = label_smoothing
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight
        self.pad_weight = pad_weight
        self.cls_score_thresh = cls_score_thresh
        self.segment_task = segment_task
        super(SegmentTarget, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'feat_stride': self.feat_stride,
            'label_smoothing': self.label_smoothing,
            'pos_weight': self.pos_weight,
            'neg_weight': self.neg_weight,
            'pad_weight': self.pad_weight,
            'cls_score_thresh': self.cls_score_thresh,
            'segment_task': self.segment_task
        })
        return config

    def call(self, inputs, **kwargs):
        split_line_pos, feat_width, real_features_width, pred_cls_logit = inputs  # 要求划分位置是有序的，从小到大; padding value -1

        batch_size = split_line_pos.shape[0]
        split_line_x1, split_line_x2 = split_line_pos[..., 0], split_line_pos[..., 1]
        split_line_center = (split_line_x1 + split_line_x2) * 0.5

        x_interval_num = tf.floor(split_line_center / self.feat_stride)

        # 如果多条划分线落在同一区间，该区间只负责预测第一条划分线，其它的忽略；这种情况几乎不可能发生
        _first_col = tf.constant(-1., shape=[batch_size, 1])
        prev_interval_num = tf.concat([_first_col, x_interval_num[:, :-1]], axis=1)
        x_interval_num = tf.where(x_interval_num == prev_interval_num, -1., x_interval_num)

        split_line_indices = tf.where(x_interval_num >= 0)

        x_interval_num = tf.gather_nd(x_interval_num, split_line_indices)
        split_line_pos = tf.gather_nd(split_line_pos, split_line_indices)

        batch_indices = split_line_indices[:, 0]
        x_interval_num = tf.cast(x_interval_num, tf.int64)
        target_indices = tf.stack([batch_indices, x_interval_num], axis=1)
        pre_mask = tf.ones_like(target_indices[:, 0], dtype=tf.float32)

        interval_mask = tf.scatter_nd(indices=target_indices, updates=pre_mask, shape=[batch_size, feat_width])  # 0, 1
        interval_split_line = tf.scatter_nd(indices=target_indices, updates=split_line_pos,
                                            shape=[batch_size, feat_width, 2])

        # 标签平滑
        interval_cls_goals = tf.where(interval_mask == 0., self.label_smoothing, 1. - self.label_smoothing)

        # 计算回归目标
        interval_center = (tf.range(0., feat_width) + 0.5) * self.feat_stride
        interval_center = tf.tile(interval_center[:, tf.newaxis], multiples=[1, 2])
        interval_center = interval_center[tf.newaxis, ...]  # shape (1, feat_width, 2)

        split_line_delta = (interval_split_line - interval_center) / self.feat_stride

        # 筛选负类
        if self.segment_task in ("text_line", "mix_line"):
            nearby_maximum = tf.nn.max_pool1d(interval_mask[..., tf.newaxis], ksize=3, strides=1,
                                              padding="SAME")  # zero padding
            nearby_maximum = tf.reshape(nearby_maximum, shape=tf.shape(nearby_maximum)[:2])
            # neg_indices = tf.where(nearby_maximum == 0.)    # sample method 2, pure neg indices

            pred_scores = K.sigmoid(pred_cls_logit)  # sample method 3, 抽样那些预测出错的负类，针对性学习
            neg_indices_wrong = tf.where(tf.logical_and(nearby_maximum == 0., pred_scores >= self.cls_score_thresh))
            neg_indices_right = tf.where(tf.logical_and(nearby_maximum == 0., pred_scores < self.cls_score_thresh))
            neg_indices = tf.concat([neg_indices_wrong, tf.random.shuffle(neg_indices_right)], axis=0)
        else:
            neg_indices = tf.where(interval_mask == 0.)  # sample method 1
            neg_indices = tf.random.shuffle(neg_indices)

        num_positive = tf.shape(target_indices)[0]
        num_negative = tf.shape(neg_indices)[0]

        # 抽样，使正负样本均衡
        num_samples = tf.minimum(num_positive, num_negative)
        pos_indices = target_indices[:num_samples]
        neg_indices = neg_indices[:num_samples]

        ones = tf.ones_like(neg_indices[:, 0], dtype=tf.float32)
        twos = tf.ones_like(pos_indices[:, 0], dtype=tf.float32) + 1.
        flag = tf.scatter_nd(indices=neg_indices, updates=ones, shape=[batch_size, feat_width])  # 0, 1
        flag = tf.tensor_scatter_nd_add(tensor=flag, indices=pos_indices, updates=twos)  # 0, 1, 2

        # 不同类别占损失的权重
        feat_region = tf.expand_dims(tf.range(0, feat_width, dtype=tf.int32), axis=0)
        real_features_width = tf.expand_dims(tf.cast(real_features_width, tf.int32), axis=1)
        inside_weights = tf.where(feat_region <= real_features_width, self.neg_weight, self.pad_weight)
        inside_weights = tf.where(flag == 1., inside_weights, 0.)
        inside_weights = tf.where(flag == 2., self.pos_weight, inside_weights)

        # summary, 用作度量的必须是浮点类型
        num_pos = tf.cast(num_positive, tf.float32)
        num_neg = tf.cast(num_negative, tf.float32)

        return interval_cls_goals, split_line_delta, interval_mask, inside_weights, num_pos, num_neg


if __name__ == '__main__':
    print("Done !")
