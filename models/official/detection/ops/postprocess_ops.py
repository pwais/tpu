# Lint as: python2, python3
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Post-processing model outputs to generate detection."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
from six.moves import range
import tensorflow.compat.v1 as tf

from utils import box_utils


def generate_detections_factory(params):
  """Factory to select function to generate detection."""
  if params.use_batched_nms:
    func = functools.partial(
        _generate_detections_batched,
        max_total_size=params.max_total_size,
        nms_iou_threshold=params.nms_iou_threshold,
        score_threshold=params.score_threshold)
  else:
    func = functools.partial(
        _generate_detections,
        max_total_size=params.max_total_size,
        nms_iou_threshold=params.nms_iou_threshold,
        score_threshold=params.score_threshold,
        pre_nms_num_boxes=params.pre_nms_num_boxes)
  return func


def _generate_detections(boxes,
                         scores,
                         cuboids=None,
                         max_total_size=100,
                         nms_iou_threshold=0.3,
                         score_threshold=0.05,
                         pre_nms_num_boxes=5000):
  """Generate the final detections given the model outputs.

  This uses batch unrolling, which is TPU compatible.

  Args:
    boxes: a tensor with shape [batch_size, N, num_classes, 4] or
      [batch_size, N, 1, 4], which box predictions on all feature levels. The N
      is the number of total anchors on all levels.
    scores: a tensor with shape [batch_size, N, num_classes], which
      stacks class probability on all feature levels. The N is the number of
      total anchors on all levels. The num_classes is the number of classes
      predicted by the model. Note that the class_outputs here is the raw score.
    cuboids: a dict of cuboid property to tensor containing scores for
      that cuboid property.
    max_total_size: a scalar representing maximum number of boxes retained over
      all classes.
    nms_iou_threshold: a float representing the threshold for deciding whether
      boxes overlap too much with respect to IOU.
    score_threshold: a float representing the threshold for deciding when to
      remove boxes based on score.
    pre_nms_num_boxes: an int number of top candidate detections per class
      before NMS.

  Returns:
    nms_boxes: `float` Tensor of shape [batch_size, max_total_size, 4]
      representing top detected boxes in [y1, x1, y2, x2].
    nms_scores: `float` Tensor of shape [batch_size, max_total_size]
      representing sorted confidence scores for detected boxes. The values are
      between [0, 1].
    nms_classes: `int` Tensor of shape [batch_size, max_total_size] representing
      classes for detected boxes.
    valid_detections: `int` Tensor of shape [batch_size] only the top
      `valid_detections` boxes are valid detections.
    nms_cuboids: (optional) dict of key -> `float` Tensor of shape
      [batch_size, max_total_size, ?] representing predictions of 
      detected cuboids.
  """
  with tf.name_scope('generate_detections'):
    batch_size = scores.get_shape().as_list()[0]
    nmsed_boxes = []
    nmsed_classes = []
    nmsed_scores = []
    nmsed_cuboids = {}
    valid_detections = []
    for i in range(batch_size):
      cuboids_i = {}
      if cuboids:
        cuboids_i.update(dict((k, preds[i]) for k, preds in cuboids.items()))
      (nmsed_boxes_i, nmsed_scores_i, nmsed_classes_i,
       valid_detections_i, nmsed_cuboids_i) = _generate_detections_per_image(
          boxes[i],
          scores[i],
          cuboids=cuboids_i,
          max_total_size=max_total_size,
          nms_iou_threshold=nms_iou_threshold,
          score_threshold=score_threshold,
          pre_nms_num_boxes=pre_nms_num_boxes)
      nmsed_boxes.append(nmsed_boxes_i)
      nmsed_scores.append(nmsed_scores_i)
      nmsed_classes.append(nmsed_classes_i)
      for prop, prop_preds_i in nmsed_cuboids_i.items():
        nmsed_cuboids.setdefault(prop, [])
        nmsed_cuboids[prop].append(prop_preds_i)
      valid_detections.append(valid_detections_i)
  nmsed_boxes = tf.stack(nmsed_boxes, axis=0)
  nmsed_scores = tf.stack(nmsed_scores, axis=0)
  nmsed_classes = tf.stack(nmsed_classes, axis=0)
  nmsed_cuboids = dict(
    (prop, tf.stack(all_preds, axis=0))
    for prop, all_preds in nmsed_cuboids.items())
  valid_detections = tf.stack(valid_detections, axis=0)
  return (
    nmsed_boxes, nmsed_scores, nmsed_classes, valid_detections, nmsed_cuboids)


def _generate_detections_per_image(boxes,
                                   scores,
                                   cuboids=None,
                                   max_total_size=100,
                                   nms_iou_threshold=0.3,
                                   score_threshold=0.05,
                                   pre_nms_num_boxes=5000):
  """Generate the final detections per image given the model outputs.

  Args:
    boxes: a tensor with shape [N, num_classes, 4] or [N, 1, 4], which box
      predictions on all feature levels. The N is the number of total anchors on
      all levels.
    scores: a tensor with shape [N, num_classes], which stacks class probability
      on all feature levels. The N is the number of total anchors on all levels.
      The num_classes is the number of classes predicted by the model. Note that
      the class_outputs here is the raw score.
    cuboids: a dict of cuboid property to tensor containing scores for
      that cuboid property.
    max_total_size: a scalar representing maximum number of boxes retained over
      all classes.
    nms_iou_threshold: a float representing the threshold for deciding whether
      boxes overlap too much with respect to IOU.
    score_threshold: a float representing the threshold for deciding when to
      remove boxes based on score.
    pre_nms_num_boxes: an int number of top candidate detections per class
      before NMS.

  Returns:
    nms_boxes: `float` Tensor of shape [max_total_size, 4] representing top
      detected boxes in [y1, x1, y2, x2].
    nms_scores: `float` Tensor of shape [max_total_size] representing sorted
      confidence scores for detected boxes. The values are between [0, 1].
    nms_classes: `int` Tensor of shape [max_total_size] representing classes for
      detected boxes.
    valid_detections: `int` Tensor of shape [1] only the top `valid_detections`
      boxes are valid detections.
    nmsed_cuboids: (optional) dict of key -> `float` Tensor of shape
      [max_total_size, ?] representing predictions of top detected cuboids.
  """
  nmsed_boxes = []
  nmsed_scores = []
  nmsed_classes = []
  nmsed_cuboids = {}
  num_classes_for_box = boxes.get_shape().as_list()[1]
  num_classes = scores.get_shape().as_list()[1]
  for i in range(num_classes):
    boxes_i = boxes[:, min(num_classes_for_box - 1, i)]
    scores_i = scores[:, i]

    # Obtains pre_nms_num_boxes before running NMS.
    scores_i, indices = tf.nn.top_k(
        scores_i, k=tf.minimum(tf.shape(scores_i)[-1], pre_nms_num_boxes))
    boxes_i = tf.gather(boxes_i, indices)

    cuboid_preds_i = {}
    if cuboids:
      # Although cuboid estimates are class-agnostic, so are boxes; so we'll
      # combine them using the same procedure applied to boxes.
      assert num_classes_for_box == 1, "No per-class cuboid support"
      for prop, preds in cuboids.items():
        cuboid_preds_i[prop] = tf.gather(preds, indices)

    (nmsed_indices_i,
     nmsed_num_valid_i) = tf.image.non_max_suppression_padded(
         tf.cast(boxes_i, tf.float32),
         tf.cast(scores_i, tf.float32),
         max_total_size,
         iou_threshold=nms_iou_threshold,
         score_threshold=score_threshold,
         pad_to_max_output_size=True,
         name='nms_detections_' + str(i))
    nmsed_boxes_i = tf.gather(boxes_i, nmsed_indices_i)
    nmsed_scores_i = tf.gather(scores_i, nmsed_indices_i)
    # Sets scores of invalid boxes to -1.
    nmsed_scores_i = tf.where(
        tf.less(tf.range(max_total_size), [nmsed_num_valid_i]),
        nmsed_scores_i, -tf.ones_like(nmsed_scores_i))
    nmsed_classes_i = tf.fill([max_total_size], i)
    nmsed_boxes.append(nmsed_boxes_i)
    nmsed_scores.append(nmsed_scores_i)
    nmsed_classes.append(nmsed_classes_i)
    
    if cuboid_preds_i:
      for prop, preds_i in cuboid_preds_i.items():
        nmsed_preds_i = tf.gather(preds_i, nmsed_indices_i)
        # Sets scores of invalid boxes to -inf.
        nmsed_preds_i = tf.where(
                    tf.less(tf.range(max_total_size), [nmsed_num_valid_i]),
                    nmsed_preds_i,
                    float('-inf') * tf.ones_like(nmsed_preds_i))

        nmsed_cuboids.setdefault(prop, [])
        nmsed_cuboids[prop].append(nmsed_preds_i)

  # Concats results from all classes
  nmsed_boxes = tf.concat(nmsed_boxes, axis=0)
  nmsed_scores = tf.concat(nmsed_scores, axis=0)
  nmsed_classes = tf.concat(nmsed_classes, axis=0)
  nmsed_cuboids = dict(
                    (prop, tf.concat(nmsed_preds, axis=0))
                    for prop, nmsed_preds in nmsed_cuboids.items())

  # Sort results and take top k overall
  nmsed_scores, indices = tf.nn.top_k(
      nmsed_scores, k=max_total_size, sorted=True)
  nmsed_boxes = tf.gather(nmsed_boxes, indices)
  nmsed_classes = tf.gather(nmsed_classes, indices)
  nmsed_cuboids = dict(
                    (prop, tf.gather(preds, indices))
                    for prop, preds in nmsed_cuboids.items())
  valid_detections = tf.reduce_sum(
      tf.cast(tf.greater(nmsed_scores, -1), tf.int32))
  return (
    nmsed_boxes, nmsed_scores, nmsed_classes, valid_detections, nmsed_cuboids)


def _generate_detections_batched(boxes,
                                 scores,
                                 max_total_size,
                                 nms_iou_threshold,
                                 score_threshold,
                                 cuboids=None):
  """Generates detected boxes with scores and classes for one-stage detector.

  The function takes output of multi-level ConvNets and anchor boxes and
  generates detected boxes. Note that this used batched nms, which is not
  supported on TPU currently.

  Args:
    boxes: a tensor with shape [batch_size, N, num_classes, 4] or
      [batch_size, N, 1, 4], which box predictions on all feature levels. The N
      is the number of total anchors on all levels.
    scores: a tensor with shape [batch_size, N, num_classes], which
      stacks class probability on all feature levels. The N is the number of
      total anchors on all levels. The num_classes is the number of classes
      predicted by the model. Note that the class_outputs here is the raw score.
    max_total_size: a scalar representing maximum number of boxes retained over
      all classes.
    nms_iou_threshold: a float representing the threshold for deciding whether
      boxes overlap too much with respect to IOU.
    score_threshold: a float representing the threshold for deciding when to
      remove boxes based on score.
    cuboids: (ignored for now)
  Returns:
    nms_boxes: `float` Tensor of shape [batch_size, max_total_size, 4]
      representing top detected boxes in [y1, x1, y2, x2].
    nms_scores: `float` Tensor of shape [batch_size, max_total_size]
      representing sorted confidence scores for detected boxes. The values are
      between [0, 1].
    nms_classes: `int` Tensor of shape [batch_size, max_total_size] representing
      classes for detected boxes.
    valid_detections: `int` Tensor of shape [batch_size] only the top
      `valid_detections` boxes are valid detections.
    nms_cuboids: None (ignored)
  """
  with tf.name_scope('generate_detections'):
    # TODO(tsungyi): Removes normalization/denomalization once the
    # tf.image.combined_non_max_suppression is coordinate system agnostic.
    # Normalizes maximum box coordinates to 1.
    normalizer = tf.reduce_max(boxes)
    boxes /= normalizer
    (nmsed_boxes, nmsed_scores, nmsed_classes,
     valid_detections) = tf.image.combined_non_max_suppression(
         boxes,
         scores,
         max_output_size_per_class=max_total_size,
         max_total_size=max_total_size,
         iou_threshold=nms_iou_threshold,
         score_threshold=score_threshold,
         pad_per_class=False,)
    # De-normalizes box coordinates.
    nmsed_boxes *= normalizer
  return nmsed_boxes, nmsed_scores, nmsed_classes, valid_detections, None


class MultilevelDetectionGenerator(object):
  """Generates detected boxes with scores and classes for one-stage detector."""

  def __init__(self, params):
    self._generate_detections = generate_detections_factory(params)
    self._min_level = params.min_level
    self._max_level = params.max_level

  def __call__(self,
               box_outputs,
               class_outputs,
               anchor_boxes,
               image_shape,
               cuboid_outputs=None):
    # Collects outputs from all levels into a list.
    boxes = []
    scores = []
    cuboids = {}
    for i in range(self._min_level, self._max_level + 1):
      box_outputs_i_shape = tf.shape(box_outputs[i])
      batch_size = box_outputs_i_shape[0]
      num_anchors_per_locations = box_outputs_i_shape[-1] // 4
      num_classes = tf.shape(class_outputs[i])[-1] // num_anchors_per_locations

      # Applies score transformation and remove the implicit background class.
      scores_i = tf.sigmoid(
          tf.reshape(class_outputs[i], [batch_size, -1, num_classes]))
      scores_i = tf.slice(scores_i, [0, 0, 1], [-1, -1, -1])

      # Box decoding.
      # The anchor boxes are shared for all data in a batch.
      # One stage detector only supports class agnostic box regression.
      anchor_boxes_i = tf.reshape(anchor_boxes[i], [batch_size, -1, 4])
      box_outputs_i = tf.reshape(box_outputs[i], [batch_size, -1, 4])
      boxes_i = box_utils.decode_boxes(box_outputs_i, anchor_boxes_i)

      # Box clipping.
      boxes_i = box_utils.clip_boxes(boxes_i, image_shape)
      boxes.append(boxes_i)
      scores.append(scores_i)

      # Pack cuboids
      if cuboid_outputs:
        level_outputs = cuboid_outputs[i]
        for prop, preds in level_outputs.items():
          # Stack all predictions to match layout of boxes and scores
          preds_per_box = tf.shape(preds)[-1]
          preds_i = tf.reshape(preds, [batch_size, -1, preds_per_box])
          cuboids.setdefault(prop, [])
          cuboids[prop].append(preds_i)

    boxes = tf.concat(boxes, axis=1)
    scores = tf.concat(scores, axis=1)
    if cuboids:
      cuboids = dict(
                  (prop, tf.concat(preds, axis=1))
                  for prop, preds in cuboids.items())

    result = self._generate_detections(
                tf.expand_dims(boxes, axis=2),
                  # Callee expects perhaps one box per class
                scores,
                cuboids=cuboids)
    (nmsed_boxes, nmsed_scores, nmsed_classes,
      valid_detections, nmsed_cuboids) = result

    # Adds 1 to offset the background class which has index 0.
    nmsed_classes += 1
    return (
      nmsed_boxes, nmsed_scores, nmsed_classes, valid_detections, nmsed_cuboids)


class GenericDetectionGenerator(object):
  """Generates the final detected boxes with scores and classes."""

  def __init__(self, params):
    self._generate_detections = generate_detections_factory(params)

  def __call__(self,
               box_outputs,
               class_outputs,
               anchor_boxes,
               image_shape,
               cuboid_outputs=None):
    """Generate final detections.

    Args:
      box_outputs: a tensor of shape of [batch_size, K, num_classes * 4]
        representing the class-specific box coordinates relative to anchors.
      class_outputs: a tensor of shape of [batch_size, K, num_classes]
        representing the class logits before applying score activiation.
      anchor_boxes: a tensor of shape of [batch_size, K, 4] representing the
        corresponding anchor boxes w.r.t `box_outputs`.
      image_shape: a tensor of shape of [batch_size, 2] storing the image height
        and width w.r.t. the scaled image, i.e. the same image space as
        `box_outputs` and `anchor_boxes`.
      cuboid_outputs: (optional) dict of key -> tensor of cuboid outputs,
        with each tensor of shape [batch_size, K, num_outputs]

    Returns:
      nms_boxes: `float` Tensor of shape [batch_size, max_total_size, 4]
        representing top detected boxes in [y1, x1, y2, x2].
      nms_scores: `float` Tensor of shape [batch_size, max_total_size]
        representing sorted confidence scores for detected boxes. The values are
        between [0, 1].
      nms_classes: `int` Tensor of shape [batch_size, max_total_size]
        representing classes for detected boxes.
      valid_detections: `int` Tensor of shape [batch_size] only the top
        `valid_detections` boxes are valid detections.
      nmsed_cuboids: (optional) dict of key -> `float` Tensor of shape
        [batch_size, max_total_size, ?] representing predictions of
        top detected cuboids.
    """
    class_outputs = tf.nn.softmax(class_outputs, axis=-1)

    # Removes the background class.
    class_outputs_shape = tf.shape(class_outputs)
    batch_size = class_outputs_shape[0]
    num_locations = class_outputs_shape[1]
    num_classes = class_outputs_shape[-1]
    num_detections = num_locations * (num_classes - 1)

    class_outputs = tf.slice(class_outputs, [0, 0, 1], [-1, -1, -1])
    box_outputs = tf.reshape(
        box_outputs,
        tf.stack([batch_size, num_locations, num_classes, 4], axis=-1))
    box_outputs = tf.slice(
        box_outputs, [0, 0, 1, 0], [-1, -1, -1, -1])
    anchor_boxes = tf.tile(
        tf.expand_dims(anchor_boxes, axis=2), [1, 1, num_classes - 1, 1])
    box_outputs = tf.reshape(
        box_outputs,
        tf.stack([batch_size, num_detections, 4], axis=-1))
    anchor_boxes = tf.reshape(
        anchor_boxes,
        tf.stack([batch_size, num_detections, 4], axis=-1))

    # Box decoding.
    decoded_boxes = box_utils.decode_boxes(
        box_outputs, anchor_boxes, weights=[10.0, 10.0, 5.0, 5.0])

    # Box clipping
    decoded_boxes = box_utils.clip_boxes(decoded_boxes, image_shape)

    decoded_boxes = tf.reshape(
        decoded_boxes,
        tf.stack([batch_size, num_locations, num_classes - 1, 4], axis=-1))

    #cuboid_outputs needs tf.slice ???
    res = self._generate_detections(
                  decoded_boxes, class_outputs, cuboids=cuboid_outputs)
    (nmsed_boxes, nmsed_scores, nmsed_classes,
      valid_detections, nmsed_cuboids) = res

    # Adds 1 to offset the background class which has index 0.
    nmsed_classes += 1

    return (nmsed_boxes, nmsed_scores, nmsed_classes,
      valid_detections, nmsed_cuboids)
