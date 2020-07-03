# -*- encoding: utf-8 -*-
# Author: hushukai

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import backend as K

from .utils import pad_to_fixed_size_tf, remove_pad_tf


def nms(split_positions, scores, score_thresh=0.7, distance_thresh=16, max_outputs=50):
    """Non-Maximum-Suppression"""
    
    indices = tf.where(scores >= score_thresh)[:, 0]
    scores = tf.gather(scores, indices)
    split_positions = tf.gather(split_positions, indices)

    # 获取自适应的distance_thresh
    if distance_thresh <= 1:
        distance_thresh_ratio = distance_thresh
        split_num = tf.cast(tf.shape(split_positions)[0], tf.float32)
        split_cent = tf.reduce_mean(split_positions, axis=1)
        split_minimum = tf.reduce_min(split_cent)
        split_maximum = tf.reduce_max(split_cent)
        distance_thresh = distance_thresh_ratio * (split_maximum - split_minimum) / (split_num - 1.)
    
    ordered_indices = tf.argsort(scores)[::-1]
    ordered_scores = tf.gather(scores, ordered_indices)
    ordered_positions = tf.gather(split_positions, ordered_indices)
    
    nms_scores = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    nms_positions = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

    def loop_condition(j, ordered_scores, *args):
        return tf.shape(ordered_scores)[0] > 0

    def loop_body(j, ordered_scores, ordered_positions, nms_scores, nms_positions):
        curr_score = ordered_scores[0]
        curr_positions = ordered_positions[0]
        nms_scores = nms_scores.write(j, curr_score)
        nms_positions = nms_positions.write(j, curr_positions)
        
        distances = tf.reduce_mean(ordered_positions[1:], axis=1) - tf.reduce_mean(curr_positions, keepdims=True)
        _indices = tf.where(tf.abs(distances) > distance_thresh)[:, 0] + 1
        
        ordered_scores = tf.gather(ordered_scores, _indices)
        ordered_positions = tf.gather(ordered_positions, _indices)
        return j + 1, ordered_scores, ordered_positions, nms_scores, nms_positions

    _, _, _, nms_scores, nms_positions = tf.while_loop(cond=loop_condition, body=loop_body,
                                                       loop_vars=[0, ordered_scores, ordered_positions, nms_scores, nms_positions])

    nms_scores = nms_scores.stack()
    nms_positions = nms_positions.stack()
    
    nms_scores = pad_to_fixed_size_tf(nms_scores[:, tf.newaxis], max_outputs)
    nms_positions = pad_to_fixed_size_tf(nms_positions, max_outputs)
    
    return [nms_positions, nms_scores]


class ExtractSplitPosition(layers.Layer):
    
    def __init__(self, feat_stride=16, cls_score_thresh=0.7, distance_thresh=16, nms_max_outputs=50, **kwargs):
        self.feat_stride = feat_stride
        self.cls_score_thresh = cls_score_thresh
        self.distance_thresh = distance_thresh
        self.nms_max_outputs = nms_max_outputs
        super(ExtractSplitPosition, self).__init__(**kwargs)
    
    def call(self, inputs, **kwargs):
        pred_cls_logit, pred_delta, img_width, real_images_width = inputs
        batch_size = tf.shape(pred_cls_logit)[0]
        feat_width = img_width // self.feat_stride

        interval_center = (tf.range(0., feat_width) + 0.5) * self.feat_stride
        interval_center = tf.tile(interval_center[:, tf.newaxis], multiples=[1, 2])
        interval_center = interval_center[tf.newaxis, ...]  # shape (1, feat_width, 2)
        
        pred_split_positions = pred_delta * self.feat_stride + interval_center
        pred_scores = K.sigmoid(pred_cls_logit)

        max_width = real_images_width[:, tf.newaxis, tf.newaxis] - 1.
        pred_split_positions = tf.where(pred_split_positions < 0., 0., pred_split_positions)
        pred_split_positions = tf.where(pred_split_positions > max_width, max_width, pred_split_positions)

        # 非极大抑制
        options = {"score_thresh": self.cls_score_thresh,
                   "distance_thresh": self.distance_thresh,
                   "max_outputs": self.nms_max_outputs}
        nms_split_positions, nms_scores = tf.map_fn(fn=lambda x: nms(*x, **options),
                                                    elems=[pred_split_positions, pred_scores],
                                                    dtype=[tf.float32, tf.float32])
        
        # In order to compute accuracy
        nms_center = tf.reduce_mean(nms_split_positions[..., :2], axis=2)
        x_interval_num = tf.floor(nms_center / self.feat_stride)
        
        nms_indices = tf.where(nms_split_positions[..., 2] == 1.)
        x_interval_num = tf.gather_nd(x_interval_num, nms_indices)

        batch_indices = nms_indices[:, 0]
        x_interval_num = tf.cast(x_interval_num, tf.int64)
        target_indices = tf.stack([batch_indices, x_interval_num], axis=1)
        pre_nms_cls = tf.ones_like(target_indices[:, 0], dtype=tf.float32)
        
        nms_cls_ids = tf.scatter_nd(indices=target_indices, updates=pre_nms_cls, shape=[batch_size, feat_width])  # 0, 1
        
        return nms_split_positions, nms_scores, nms_cls_ids


if __name__ == '__main__':
    print("Done !")
