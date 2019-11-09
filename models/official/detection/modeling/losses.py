# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Losses used for Mask-RCNN."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf


def focal_loss(logits, targets, alpha, gamma, normalizer):
  """Compute the focal loss between `logits` and the golden `target` values.

  Focal loss = -(1-pt)^gamma * log(pt)
  where pt is the probability of being classified to the true class.

  Args:
    logits: A float32 tensor of size
      [batch, height_in, width_in, num_predictions].
    targets: A float32 tensor of size
      [batch, height_in, width_in, num_predictions].
    alpha: A float32 scalar multiplying alpha to the loss from positive examples
      and (1-alpha) to the loss from negative examples.
    gamma: A float32 scalar modulating loss from hard and easy examples.
    normalizer: A float32 scalar normalizes the total loss from all examples.

  Returns:
    loss: A float32 Tensor of size [batch, height_in, width_in, num_predictions]
      representing normalized loss on the prediction map.
  """
  with tf.name_scope('focal_loss'):
    positive_label_mask = tf.equal(targets, 1.0)
    cross_entropy = (
        tf.nn.sigmoid_cross_entropy_with_logits(labels=targets, logits=logits))
    # Below are comments/derivations for computing modulator.
    # For brevity, let x = logits,  z = targets, r = gamma, and p_t = sigmod(x)
    # for positive samples and 1 - sigmoid(x) for negative examples.
    #
    # The modulator, defined as (1 - P_t)^r, is a critical part in focal loss
    # computation. For r > 0, it puts more weights on hard examples, and less
    # weights on easier ones. However if it is directly computed as (1 - P_t)^r,
    # its back-propagation is not stable when r < 1. The implementation here
    # resolves the issue.
    #
    # For positive samples (labels being 1),
    #    (1 - p_t)^r
    #  = (1 - sigmoid(x))^r
    #  = (1 - (1 / (1 + exp(-x))))^r
    #  = (exp(-x) / (1 + exp(-x)))^r
    #  = exp(log((exp(-x) / (1 + exp(-x)))^r))
    #  = exp(r * log(exp(-x)) - r * log(1 + exp(-x)))
    #  = exp(- r * x - r * log(1 + exp(-x)))
    #
    # For negative samples (labels being 0),
    #    (1 - p_t)^r
    #  = (sigmoid(x))^r
    #  = (1 / (1 + exp(-x)))^r
    #  = exp(log((1 / (1 + exp(-x)))^r))
    #  = exp(-r * log(1 + exp(-x)))
    #
    # Therefore one unified form for positive (z = 1) and negative (z = 0)
    # samples is:
    #      (1 - p_t)^r = exp(-r * z * x - r * log(1 + exp(-x))).
    neg_logits = -1.0 * logits
    modulator = tf.exp(gamma * targets * neg_logits - gamma * tf.log1p(
        tf.exp(neg_logits)))
    loss = modulator * cross_entropy
    weighted_loss = tf.where(positive_label_mask, alpha * loss,
                             (1.0 - alpha) * loss)
    weighted_loss /= normalizer
  return weighted_loss


class RpnScoreLoss(object):
  """Region Proposal Network score loss function."""

  def __init__(self, params):
    self._rpn_batch_size_per_im = params.rpn_batch_size_per_im

  def __call__(self, score_outputs, labels):
    """Computes total RPN detection loss.

    Computes total RPN detection loss including box and score from all levels.

    Args:
      score_outputs: an OrderDict with keys representing levels and values
        representing scores in [batch_size, height, width, num_anchors].
      labels: the dictionary that returned from dataloader that includes
        groundturth targets.

    Returns:
      rpn_score_loss: a scalar tensor representing total score loss.
    """
    with tf.name_scope('rpn_loss'):
      levels = sorted(score_outputs.keys())

      score_losses = []
      for level in levels:
        score_losses.append(
            self._rpn_score_loss(
                score_outputs[level],
                labels[level],
                normalizer=tf.cast(
                    tf.shape(score_outputs[level])[0] *
                    self._rpn_batch_size_per_im, dtype=tf.float32)))

      # Sums per level losses to total loss.
      return tf.add_n(score_losses)

  def _rpn_score_loss(self, score_outputs, score_targets, normalizer=1.0):
    """Computes score loss."""
    # score_targets has three values:
    # (1) score_targets[i]=1, the anchor is a positive sample.
    # (2) score_targets[i]=0, negative.
    # (3) score_targets[i]=-1, the anchor is don't care (ignore).
    with tf.name_scope('rpn_score_loss'):
      mask = tf.logical_or(tf.equal(score_targets, 1),
                           tf.equal(score_targets, 0))
      score_targets = tf.maximum(score_targets, tf.zeros_like(score_targets))
      # RPN score loss is sum over all except ignored samples.
      score_loss = tf.losses.sigmoid_cross_entropy(
          score_targets, score_outputs, weights=mask,
          reduction=tf.losses.Reduction.SUM)
      score_loss /= normalizer
      return score_loss


class RpnBoxLoss(object):
  """Region Proposal Network box regression loss function."""

  def __init__(self, params):
    self._delta = params.huber_loss_delta

  def __call__(self, box_outputs, labels):
    """Computes total RPN detection loss.

    Computes total RPN detection loss including box and score from all levels.

    Args:
      box_outputs: an OrderDict with keys representing levels and values
        representing box regression targets in
        [batch_size, height, width, num_anchors * 4].
      labels: the dictionary that returned from dataloader that includes
        groundturth targets.

    Returns:
      rpn_box_loss: a scalar tensor representing total box regression loss.
    """
    with tf.name_scope('rpn_loss'):
      levels = sorted(box_outputs.keys())

      box_losses = []
      for level in levels:
        box_losses.append(
            self._rpn_box_loss(
                box_outputs[level], labels[level], delta=self._delta))

      # Sum per level losses to total loss.
      return tf.add_n(box_losses)

  def _rpn_box_loss(self, box_outputs, box_targets, normalizer=1.0, delta=1./9):
    """Computes box regression loss."""
    # The delta is typically around the mean value of regression target.
    # for instances, the regression targets of 512x512 input with 6 anchors on
    # P2-P6 pyramid is about [0.1, 0.1, 0.2, 0.2].
    with tf.name_scope('rpn_box_loss'):
      mask = tf.not_equal(box_targets, 0.0)
      # The loss is normalized by the sum of non-zero weights before additional
      # normalizer provided by the function caller.
      box_loss = tf.losses.huber_loss(
          box_targets,
          box_outputs,
          weights=mask,
          delta=delta,
          reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)
      box_loss /= normalizer
      return box_loss


class FastrcnnClassLoss(object):
  """Fast R-CNN classification loss function."""

  def __call__(self, class_outputs, class_targets):
    """Computes the class loss (Fast-RCNN branch) of Mask-RCNN.

    This function implements the classification loss of the Fast-RCNN.

    The classification loss is softmax on all RoIs.
    Reference: https://github.com/facebookresearch/Detectron/blob/master/detectron/modeling/fast_rcnn_heads.py  # pylint: disable=line-too-long

    Args:
      class_outputs: a float tensor representing the class prediction for each box
        with a shape of [batch_size, num_boxes, num_classes].
      class_targets: a float tensor representing the class label for each box
        with a shape of [batch_size, num_boxes].

    Returns:
      a scalar tensor representing total class loss.
    """
    with tf.name_scope('fast_rcnn_loss'):
      _, _, num_classes = class_outputs.get_shape().as_list()
      class_targets = tf.to_int32(class_targets)
      class_targets_one_hot = tf.one_hot(class_targets, num_classes)
      return self._fast_rcnn_class_loss(class_outputs, class_targets_one_hot)

  def _fast_rcnn_class_loss(self, class_outputs, class_targets_one_hot,
                            normalizer=1.0):
    """Computes classification loss."""
    with tf.name_scope('fast_rcnn_class_loss'):
      # The loss is normalized by the sum of non-zero weights before additional
      # normalizer provided by the function caller.
      class_loss = tf.losses.softmax_cross_entropy(
          class_targets_one_hot, class_outputs,
          reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)
      class_loss /= normalizer
      return class_loss


class FastrcnnBoxLoss(object):
  """Fast R-CNN box regression loss function."""

  def __init__(self, params):
    self._delta = params.huber_loss_delta

  def __call__(self, box_outputs, class_targets, box_targets):
    """Computes the box loss (Fast-RCNN branch) of Mask-RCNN.

    This function implements the box regression loss of the Fast-RCNN. As the
    `box_outputs` produces `num_classes` boxes for each RoI, the reference model
    expands `box_targets` to match the shape of `box_outputs` and selects only
    the target that the RoI has a maximum overlap. (Reference: https://github.com/facebookresearch/Detectron/blob/master/detectron/roi_data/fast_rcnn.py)  # pylint: disable=line-too-long
    Instead, this function selects the `box_outputs` by the `class_targets` so
    that it doesn't expand `box_targets`.

    The box loss is smooth L1-loss on only positive samples of RoIs.
    Reference: https://github.com/facebookresearch/Detectron/blob/master/detectron/modeling/fast_rcnn_heads.py  # pylint: disable=line-too-long

    Args:
      box_outputs: a float tensor representing the box prediction for each box
        with a shape of [batch_size, num_boxes, num_classes * 4].
      class_targets: a float tensor representing the class label for each box
        with a shape of [batch_size, num_boxes].
      box_targets: a float tensor representing the box label for each box
        with a shape of [batch_size, num_boxes, 4].

    Returns:
      box_loss: a scalar tensor representing total box regression loss.
    """
    with tf.name_scope('fast_rcnn_loss'):
      class_targets = tf.to_int32(class_targets)

      # Selects the box from `box_outputs` based on `class_targets`, with which
      # the box has the maximum overlap.
      (batch_size, num_rois,
       num_class_specific_boxes) = box_outputs.get_shape().as_list()
      num_classes = num_class_specific_boxes // 4
      box_outputs = tf.reshape(box_outputs,
                               [batch_size, num_rois, num_classes, 4])

      box_indices = tf.reshape(
          class_targets + tf.tile(
              tf.expand_dims(
                  tf.range(batch_size) * num_rois * num_classes, 1),
              [1, num_rois]) + tf.tile(
                  tf.expand_dims(tf.range(num_rois) * num_classes, 0),
                  [batch_size, 1]), [-1])

      box_outputs = tf.matmul(
          tf.one_hot(
              box_indices,
              batch_size * num_rois * num_classes,
              dtype=box_outputs.dtype), tf.reshape(box_outputs, [-1, 4]))
      box_outputs = tf.reshape(box_outputs, [batch_size, -1, 4])

      return self._fast_rcnn_box_loss(box_outputs, box_targets, class_targets,
                                      delta=self._delta)

  def _fast_rcnn_box_loss(self, box_outputs, box_targets, class_targets,
                          normalizer=1.0, delta=1.):
    """Computes box regression loss."""
    # The delta is typically around the mean value of regression target.
    # for instances, the regression targets of 512x512 input with 6 anchors on
    # P2-P6 pyramid is about [0.1, 0.1, 0.2, 0.2].
    with tf.name_scope('fast_rcnn_box_loss'):
      mask = tf.tile(tf.expand_dims(tf.greater(class_targets, 0), axis=2),
                     [1, 1, 4])
      # The loss is normalized by the sum of non-zero weights before additional
      # normalizer provided by the function caller.
      box_loss = tf.losses.huber_loss(
          box_targets,
          box_outputs,
          weights=mask,
          delta=delta,
          reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)
      box_loss /= normalizer
      return box_loss


class MaskrcnnLoss(object):
  """Mask R-CNN instance segmentation mask loss function."""

  def __call__(self, mask_outputs, mask_targets, select_class_targets):
    """Computes the mask loss of Mask-RCNN.

    This function implements the mask loss of Mask-RCNN. As the `mask_outputs`
    produces `num_classes` masks for each RoI, the reference model expands
    `mask_targets` to match the shape of `mask_outputs` and selects only the
    target that the RoI has a maximum overlap. (Reference: https://github.com/facebookresearch/Detectron/blob/master/detectron/roi_data/mask_rcnn.py)  # pylint: disable=line-too-long
    Instead, this implementation selects the `mask_outputs` by the `class_targets`
    so that it doesn't expand `mask_targets`. Note that the selection logic is
    done in the post-processing of mask_rcnn_fn in mask_rcnn_architecture.py.

    Args:
      mask_outputs: a float tensor representing the prediction for each mask,
        with a shape of
        [batch_size, num_masks, mask_height, mask_width].
      mask_targets: a float tensor representing the binary mask of ground truth
        labels for each mask with a shape of
        [batch_size, num_masks, mask_height, mask_width].
      select_class_targets: a tensor with a shape of [batch_size, num_masks],
        representing the foreground mask targets.

    Returns:
      mask_loss: a float tensor representing total mask loss.
    """
    with tf.name_scope('mask_rcnn_loss'):
      (batch_size, num_masks, mask_height,
       mask_width) = mask_outputs.get_shape().as_list()

      weights = tf.tile(
          tf.reshape(tf.greater(select_class_targets, 0),
                     [batch_size, num_masks, 1, 1]),
          [1, 1, mask_height, mask_width])
      return tf.losses.sigmoid_cross_entropy(
          mask_targets, mask_outputs, weights=weights,
          reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)


class RetinanetClassLoss(object):
  """RetinaNet class loss."""

  def __init__(self, params):
    self._num_classes = params.num_classes
    self._focal_loss_alpha = params.focal_loss_alpha
    self._focal_loss_gamma = params.focal_loss_gamma

  def __call__(self, cls_outputs, labels, num_positives):
    """Computes class loss from all levels.

    Args:
      cls_outputs: an OrderDict with keys representing levels and values
        representing logits in [batch_size, height, width,
        num_anchors * num_classes].
      labels: the dictionary that returned from dataloader that includes
        class groundturth targets.
      num_positives: number of positive examples in the minibatch.

    Returns:
      an scalar tensor representing total class loss.
    """
    # Sums all positives in a batch for normalization and avoids zero
    # num_positives_sum, which would lead to inf loss during training
    num_positives_sum = tf.reduce_sum(num_positives) + 1.0

    cls_losses = []
    for level in cls_outputs.keys():
      cls_losses.append(self.class_loss(
          cls_outputs[level], labels[level], num_positives_sum))
    # Sums per level losses to total loss.
    return tf.add_n(cls_losses)

  def class_loss(self, cls_outputs, cls_targets, num_positives,
                 ignore_label=-2):
    """Computes RetinaNet classification loss."""
    # Onehot encoding for classification labels.
    cls_targets_one_hot = tf.one_hot(cls_targets, self._num_classes)
    bs, height, width, _, _ = cls_targets_one_hot.get_shape().as_list()
    cls_targets_one_hot = tf.reshape(cls_targets_one_hot,
                                     [bs, height, width, -1])
    loss = focal_loss(cls_outputs, cls_targets_one_hot,
                      self._focal_loss_alpha, self._focal_loss_gamma,
                      num_positives)

    ignore_loss = tf.where(tf.equal(cls_targets, ignore_label),
                           tf.zeros_like(cls_targets, dtype=tf.float32),
                           tf.ones_like(cls_targets, dtype=tf.float32),)
    ignore_loss = tf.expand_dims(ignore_loss, -1)
    ignore_loss = tf.tile(ignore_loss, [1, 1, 1, 1, self._num_classes])
    ignore_loss = tf.reshape(ignore_loss, tf.shape(loss))
    return tf.reduce_sum(ignore_loss * loss)


class RetinanetBoxLoss(object):
  """RetinaNet box loss."""

  def __init__(self, params):
    self._huber_loss_delta = params.huber_loss_delta

  def __call__(self, box_outputs, labels, num_positives):
    """Computes box detection loss from all levels.

    Args:
      box_outputs: an OrderDict with keys representing levels and values
        representing box regression targets in [batch_size, height, width,
        num_anchors * 4].
      labels: the dictionary that returned from dataloader that includes
        box groundturth targets.
      num_positives: number of positive examples in the minibatch.

    Returns:
      an scalar tensor representing total box regression loss.
    """
    # Sums all positives in a batch for normalization and avoids zero
    # num_positives_sum, which would lead to inf loss during training
    num_positives_sum = tf.reduce_sum(num_positives) + 1.0

    box_losses = []
    for level in box_outputs.keys():
      # Onehot encoding for classification labels.
      box_targets_l = labels[level]
      box_losses.append(
          self.box_loss(box_outputs[level], box_targets_l, num_positives_sum))
    # Sums per level losses to total loss.
    return tf.add_n(box_losses)

  def box_loss(self, box_outputs, box_targets, num_positives):
    """Computes RetinaNet box regression loss."""
    # The delta is typically around the mean value of regression target.
    # for instances, the regression targets of 512x512 input with 6 anchors on
    # P3-P7 pyramid is about [0.1, 0.1, 0.2, 0.2].
    normalizer = num_positives * 4.0
    mask = tf.not_equal(box_targets, 0.0)
    box_loss = tf.losses.huber_loss(
        box_targets,
        box_outputs,
        weights=mask,
        delta=self._huber_loss_delta,
        reduction=tf.losses.Reduction.SUM)
    box_loss /= normalizer
    return box_loss


class RetinanetCuboidLoss(object):
  """RetinaNet cuboid loss."""

  def __init__(self, params):
    self._cuboid_yaw_num_bins = params.cuboid_yaw_num_bins
    self._cuboid_yaw_loss_weight = params.cuboid_yaw_loss_weight
    self._cuboid_huber_loss_delta = params.cuboid_huber_loss_delta

  def __call__(self, cuboid_outputs, labels, num_positives):
    """Computes cuboid estimation loss.

    Computes cuboid losses from all levels.

    Args:
      cuboid_outputs: a dict of ouput property to outputs.  Each individual
        output is an OrderDict with keys representing levels and values
        representing logits in [batch_size, height, width,
        num_anchors * output_dimension].
      labels: the dictionary that returned from dataloader that includes
        cuboid groundturth targets.
      num_positives: number of positive examples in the minibatch.

    Returns:
      an dict of loss to scalar tensor representing cuboid regression losses.
    """
    # Sums all positives in a batch for normalization and avoids zero
    # num_positives_sum, which would lead to inf loss during training
    num_positives_sum = tf.reduce_sum(num_positives) + 1.0

    cuboid_losses = {
      'cuboid_center':
        self.center_loss(cuboid_outputs, labels, num_positives_sum),
      'cuboid_depth': 
        self.depth_loss(cuboid_outputs, labels, num_positives_sum),
      'cuboid_yaw': 
        self.yaw_loss(cuboid_outputs, labels, num_positives_sum),
      'cuboid_size': 
        self.size_loss(cuboid_outputs, labels, num_positives_sum),
    }
    return cuboid_losses
  
  def center_loss(self, cuboid_outputs, labels, num_positives_sum):
    center_losses = []
    for level in cuboid_outputs.keys():
      pred_xy = cuboid_outputs[level]['cuboid_center']
      label_x = tf.expand_dims(labels['cuboid/box/center_x'][level], -1)
      label_y = tf.expand_dims(labels['cuboid/box/center_y'][level], -1)
      label_xy = tf.concat([label_x, label_y], axis=-1)
      center_losses.append(
        self._regression_loss(pred_xy, label_xy, num_positives_sum))
    # Sums per level losses to total loss.
    return tf.add_n(center_losses)

  def depth_loss(self, cuboid_outputs, labels, num_positives_sum):
    depth_losses = []
    for level in cuboid_outputs.keys():
      pred_depth = cuboid_outputs[level]['cuboid_depth']
      label_depth = tf.expand_dims(
        labels['cuboid/box/center_depth'][level], -1)

      # Following Hu et al., we predict inverse depth.
      # https://github.com/ucbdrive/3d-vehicle-tracking/blob/ce54b2461c8983aef265ed043dec976c6035d431/3d-tracking/utils/network_utils.py#L115
      target_depth = 1. / label_depth - 1.

      depth_losses.append(
        self._regression_loss(pred_depth, target_depth, num_positives_sum))
    # Sums per level losses to total loss.
    return tf.add_n(depth_losses)

  def yaw_loss(self, cuboid_outputs, labels, num_positives_sum,
                        ignore_label=float('-inf')):
    n_bins = self._cuboid_yaw_num_bins
    yaw_losses = []
    for level in cuboid_outputs.keys():
      preds = cuboid_outputs[level]['cuboid_yaw']

      # preds = tf.check_numerics(preds, 'preds')
      label_yaw = labels['cuboid/box/perspective_yaw'][level]

      # Following both Hu et al. and Drago et al. (originators), we use
      # the multibin loss:
      # https://arxiv.org/pdf/1612.00496.pdf
      # https://cs.gmu.edu/~amousavi/papers/3D-Deepbox-Supplementary.pdf
      # NOPE Adapted from:
      # NOPE https://github.com/ucbdrive/3d-vehicle-tracking/blob/2bb6b23fcfa1fbb5982ce7fe1a8471df18777518/3d-tracking/utils/network_utils.py#L24

      PI = 3.141592653589793
        
      # Normalize to [0, 2pi]
      target_yaw = (label_yaw + 2*PI) % (2*PI)
      target_yaw = tf.expand_dims(target_yaw, -1)
      target_yaw = tf.tile(target_yaw, [1, 1, 1, 1, n_bins])
      target_yaw = tf.identity(target_yaw, name='target_yaw_%s' % level)

      bin_size_rad = (2*PI) / n_bins
      theta_bin = tf.range(0, 2*PI, bin_size_rad)
      cos_thb = tf.math.cos(theta_bin)
      sin_thb = tf.math.sin(theta_bin)

      target_bin = tf.math.abs(target_yaw - theta_bin) / bin_size_rad
      target_residual_cos_th = tf.math.cos(target_yaw) - cos_thb
      target_residual_sin_th = tf.math.sin(target_yaw) - sin_thb
      
      yaw_head_n_vals = preds.get_shape().as_list()[-1]
      assert 3 * n_bins == yaw_head_n_vals
      # GUH https://github.com/tensorflow/tensorflow/issues/2540
      # Even if we mask the loss below, TF will backprop some NaN gradients
      # into the net weights unless we mask them here.
      mask = tf.expand_dims(label_yaw, -1)
      mask = tf.tile(mask, [1, 1, 1, 1, yaw_head_n_vals])
      preds = tf.where(
                tf.equal(mask, ignore_label), tf.stop_gradient(preds), preds)

      pred_bin =    preds[:, :, :, :, 0:n_bins]
      pred_cos_th = preds[:, :, :, :, n_bins:2*n_bins]
      pred_sin_th = preds[:, :, :, :, 2*n_bins:]
      normalizer = tf.sqrt(pred_cos_th ** 2 + pred_sin_th **2 + 1e-9)
      pred_cos_th /= normalizer
      pred_sin_th /= normalizer

      # pred_cos_th = tf.check_numerics(pred_cos_th, 'pred_cos_th')
      # target_residual_cos_th = tf.check_numerics(target_residual_cos_th, 'target_residual_cos_th')
      # tf.stop_gradient
      # # Less numerical stability, but our dimensionality is small
      # pred_bin_softmax = tf.nn.softmax(pred_bin)
      # bin_loss = -1 * target_bin * tf.log(pred_bin_softmax)
      bin_loss = tf.nn.softmax_cross_entropy_with_logits(
                        labels=target_bin, logits=pred_bin,
                        name='bin_loss_%s' % level)

      cos_resid_loss = tf.losses.huber_loss(
                          target_residual_cos_th,
                          pred_cos_th,
                          delta=0.01, # TODO tune to bin_size_rad ~~~~~~~~~~~~~~~~~~~~
                          reduction=tf.losses.Reduction.NONE)
      cos_resid_loss = tf.identity(
                          cos_resid_loss,
                          name='cos_resid_loss_%s' % level)
      sin_resid_loss = tf.losses.huber_loss(
                          target_residual_sin_th,
                          pred_sin_th,
                          delta=0.01, # TODO tune to bin_size_rad ~~~~~~~~~~~~~~~~~~~~~~~
                          reduction=tf.losses.Reduction.NONE)
      sin_resid_loss = tf.identity(
                          sin_resid_loss,
                          name='sin_resid_loss_%s' % level)
      
      # # bin_loss = tf.check_numerics(filter_loss(bin_loss), 'moofbin')
      # cos_resid_loss = tf.check_numerics(filter_loss(cos_resid_loss), 'cos_resid_loss')
      # sin_resid_loss = tf.check_numerics(filter_loss(sin_resid_loss), 'sin_resid_loss')

      yaw_loss = (
        bin_loss + 
        tf.reduce_mean(cos_resid_loss, axis=-1) + 
        tf.reduce_mean(sin_resid_loss, axis=-1))

      yaw_loss = tf.where(tf.equal(label_yaw, ignore_label),
                           tf.zeros_like(label_yaw, dtype=tf.float32),
                           yaw_loss)
      
      # yaw_loss = tf.where(tf.equal(yaw_loss, float('nan')),
      #                      tf.zeros_like(yaw_loss, dtype=tf.float32),
      #                      yaw_loss)

      # import pdb; pdb.set_trace()
      # label_yaw = tf.check_numerics(label_yaw, 'label_yaw')
      # yaw_loss = tf.check_numerics(yaw_loss, 'yaw_loss')
      yaw_loss = tf.reduce_sum(yaw_loss)

      # ignore_loss = tf.where(tf.equal(label_yaw, ignore_label),
      #                      tf.zeros_like(label_yaw, dtype=tf.float32),
      #                      tf.ones_like(label_yaw, dtype=tf.float32))
      # # ignore_loss = tf.where(tf.equal(label_yaw, ignore_label),
      # #                       tf.zeros_like(label_yaw, dtype=tf.float32),
      # #                       tf.ones_like(label_yaw, dtype=tf.float32),)
      # # ignore_loss = tf.expand_dims(ignore_loss, -1)
      # # ignore_loss = tf.tile(ignore_loss, [1, 1, 1, 1, yaw_head_n_vals])
      # # import pdb; pdb.set_trace()
      # # ignore_loss = tf.reshape(ignore_loss, tf.shape(yaw_loss))

      # filtered_yaw_loss = tf.reduce_sum(
      #   ignore_loss * yaw_loss, name='filtered_yaw_loss_%s' % level)

      yaw_losses.append(yaw_loss)
    
    # Sums per level losses to total loss.
    total_loss = tf.add_n(yaw_losses) / num_positives_sum
    return self._cuboid_yaw_loss_weight * total_loss

  def size_loss(self, cuboid_outputs, labels, num_positives_sum):
    size_losses = []
    for level in cuboid_outputs.keys():
      pred_lwh = cuboid_outputs[level]['cuboid_size']
      label_l = labels['cuboid/length'][level]
      label_w = labels['cuboid/width'][level]
      label_h = labels['cuboid/height'][level]
      label_lwh = tf.concat([
                    tf.expand_dims(label_l, axis=-1),
                    tf.expand_dims(label_w, axis=-1),
                    tf.expand_dims(label_h, axis=-1)],
                    axis=-1)
      size_losses.append(
        self._regression_loss(pred_lwh, label_lwh, num_positives_sum))
    # Sums per level losses to total loss.
    return tf.add_n(size_losses)

  def _regression_loss(self, r_outputs, r_targets, num_positives,
                 ignore_label=float('-inf')):
    """Computes a RetinaNet regression loss."""
    r_dims = r_targets.get_shape().as_list()[-1]

    normalizer = num_positives * r_dims
    r_loss = tf.losses.huber_loss(
        r_targets,
        r_outputs,
        delta=self._cuboid_huber_loss_delta,
        reduction=tf.losses.Reduction.NONE)
    r_loss /= normalizer
    r_loss = tf.where(tf.equal(r_targets, ignore_label),
                           tf.zeros_like(r_targets, dtype=tf.float32),
                           r_loss)
    return tf.reduce_sum(r_loss)
    
    # ignore_loss = tf.where(tf.equal(r_targets, ignore_label),
    #                        tf.zeros_like(r_targets, dtype=tf.float32),
    #                        tf.ones_like(r_targets, dtype=tf.float32))
    # ignore_loss = tf.expand_dims(ignore_loss, -1)
    # # ignore_loss = tf.tile(ignore_loss, [1, 1, 1, 1, r_dims])
    # # ignore_loss = tf.reshape(ignore_loss, tf.shape(r_loss))
    # return tf.reduce_sum(ignore_loss * r_loss)


class ShapemaskMseLoss(object):
  """ShapeMask mask Mean Squared Error loss function wrapper."""

  def __call__(self, probs, labels, valid_mask):
    """Compute instance segmentation loss.

    Args:
      probs: A Tensor of shape [batch_size * num_points, height, width,
        num_classes]. The logits are not necessarily between 0 and 1.
      labels: A float16 Tensor of shape [batch_size, num_instances,
          mask_size, mask_size], where mask_size =
          mask_crop_size * gt_upsample_scale for fine mask, or mask_crop_size
          for coarse masks and shape priors.
      valid_mask: a binary mask indicating valid training masks.

    Returns:
      loss: an float tensor representing total mask classification loss.
    """
    with tf.name_scope('shapemask_prior_loss'):
      batch_size, num_instances = valid_mask.get_shape().as_list()[:2]
      diff = labels - probs
      diff *= tf.cast(tf.reshape(
          valid_mask, [batch_size, num_instances, 1, 1]), diff.dtype)
      loss = tf.nn.l2_loss(diff) / tf.reduce_sum(labels)
    return loss


class ShapemaskLoss(object):
  """ShapeMask mask loss function wrapper."""

  def __call__(self, logits, labels, valid_mask):
    """ShapeMask mask cross entropy loss function wrapper.

    Args:
      logits: A Tensor of shape [batch_size * num_instances, height, width,
        num_classes]. The logits are not necessarily between 0 and 1.
      labels: A float16 Tensor of shape [batch_size, num_instances,
        mask_size, mask_size], where mask_size =
        mask_crop_size * gt_upsample_scale for fine mask, or mask_crop_size
        for coarse masks and shape priors.
      valid_mask: a binary mask of shape [batch_size, num_instances]
        indicating valid training masks.
    Returns:
      loss: an float tensor representing total mask classification loss.
    """
    with tf.name_scope('shapemask_loss'):
      batch_size, num_instances = valid_mask.get_shape().as_list()[:2]
      loss = tf.nn.sigmoid_cross_entropy_with_logits(
          labels=labels, logits=logits)
      loss *= tf.cast(tf.reshape(
          valid_mask, [batch_size, num_instances, 1, 1]), loss.dtype)
      loss = tf.reduce_sum(loss) / tf.reduce_sum(labels)
    return loss


class SegmentationLoss(object):
  """Semantic segmentationloss function."""

  def __init__(self, params):
    self._ignore_label = params.ignore_label

  def __call__(self, logits, labels):
    _, height, width, _ = logits.get_shape().as_list()
    labels = tf.image.resize_images(
        labels, (height, width), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    valid_mask = tf.not_equal(labels, self._ignore_label)
    normalizer = tf.reduce_sum(tf.to_float(valid_mask))
    # Assign pixel with ignore label to class 0 (background). The loss on the
    # pixel will later be masked out.
    labels = tf.where(valid_mask, labels, tf.zeros_like(labels))

    labels = tf.squeeze(tf.cast(labels, tf.int32), axis=3)
    valid_mask = tf.squeeze(tf.cast(valid_mask, tf.float32), axis=3)
    cross_entropy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits)
    cross_entropy_loss *= tf.to_float(valid_mask)
    loss = tf.reduce_sum(cross_entropy_loss) / normalizer
    return loss
