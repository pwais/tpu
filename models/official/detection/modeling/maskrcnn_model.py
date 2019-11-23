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
"""Model defination for the Mask R-CNN Model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf

from dataloader import mode_keys
from modeling import base_model
from modeling import losses
from modeling.architecture import factory
from ops import postprocess_ops
from ops import roi_ops
from ops import sampling_ops
from ops import spatial_transform_ops
from utils import benchmark_utils
from utils import box_utils


class MaskrcnnModel(base_model.Model):
  """RetinaNet model function."""

  def __init__(self, params):
    super(MaskrcnnModel, self).__init__(params)

    self._include_mask = params.architecture.include_mask
    self._include_cuboids = params.architecture.include_cuboids

    # Architecture generators.
    self._backbone_fn = factory.backbone_generator(params)
    self._fpn_fn = factory.multilevel_features_generator(params)
    self._rpn_head_fn = factory.rpn_head_generator(params.rpn_head)
    self._generate_rois_fn = roi_ops.ROIGenerator(params.roi_proposal)
    self._sample_rois_fn = sampling_ops.ROISampler(params.roi_sampling)
    self._sample_masks_fn = sampling_ops.MaskSampler(params.mask_sampling)
    self._sample_cuboids_fn = sampling_ops.CuboidSampler(params.cuboid_sampling)

    self._frcnn_head_fn = factory.fast_rcnn_head_generator(params.frcnn_head)
    if self._include_mask:
      self._mrcnn_head_fn = factory.mask_rcnn_head_generator(params.mrcnn_head)
    self._cuboid_head_fn = None
    if self._include_cuboids:
      self._cuboid_head_fn = (
        factory.fast_rcnn_cuboid_head_generator(params.frcnn_cuboid_head))

    # Loss function.
    self._rpn_score_loss_fn = losses.RpnScoreLoss(params.rpn_score_loss)
    self._rpn_box_loss_fn = losses.RpnBoxLoss(params.rpn_box_loss)
    self._frcnn_class_loss_fn = losses.FastrcnnClassLoss()
    self._frcnn_box_loss_fn = losses.FastrcnnBoxLoss(params.frcnn_box_loss)
    if self._include_mask:
      self._mask_loss_fn = losses.MaskrcnnLoss()
    if self._include_cuboids:
      self._cuboid_loss_fn = losses.FastrcnnCuboidLoss(params.frcnn_cuboid_loss)
    self._cuboid_total_loss_weight = (
      params.architecture.cuboid_total_loss_weight)

    self._generate_detections_fn = postprocess_ops.GenericDetectionGenerator(
        params.postprocess)

    self._transpose_input = params.train.transpose_input

  def build_outputs(self, features, labels, mode):
    is_training = mode == mode_keys.TRAIN
    model_outputs = {}

    backbone_features = self._backbone_fn(features, is_training)
    fpn_features = self._fpn_fn(backbone_features, is_training)

    # Print number of parameters and FLOPS in model.
    batch_size, _, _, _ = list(backbone_features.values())[0].get_shape().as_list()  # pylint: disable=line-too-long
    benchmark_utils.compute_model_statistics(
        batch_size, is_training=(mode == mode_keys.TRAIN))

    rpn_score_outputs, rpn_box_outputs = self._rpn_head_fn(
        fpn_features, is_training)
    model_outputs.update({
        'rpn_score_outputs': rpn_score_outputs,
        'rpn_box_outputs': rpn_box_outputs,
    })
    rpn_rois, _ = self._generate_rois_fn(
        rpn_box_outputs,
        rpn_score_outputs,
        labels['anchor_boxes'],
        labels['image_info'][:, 1, :],
        is_training)

    if is_training:
      rpn_rois = tf.stop_gradient(rpn_rois)

      # Sample proposals.
      rpn_rois, matched_gt_boxes, matched_gt_classes, matched_gt_indices = (
          self._sample_rois_fn(
              rpn_rois, labels['gt_boxes'], labels['gt_classes']))

      # Create bounding box training targets.
      box_targets = box_utils.encode_boxes(
          matched_gt_boxes, rpn_rois, weights=[10.0, 10.0, 5.0, 5.0])
      # If the target is background, the box target is set to all 0s.
      box_targets = tf.where(
          tf.tile(
              tf.expand_dims(tf.equal(matched_gt_classes, 0), axis=-1),
              [1, 1, 4]),
          tf.zeros_like(box_targets),
          box_targets)
      model_outputs.update({
          'class_targets': matched_gt_classes,
          'box_targets': box_targets,
      })

    roi_features = spatial_transform_ops.multilevel_crop_and_resize(
        fpn_features, rpn_rois, output_size=7)

    class_outputs, box_outputs = self._frcnn_head_fn(roi_features, is_training)
    model_outputs.update({
        'class_outputs': class_outputs,
        'box_outputs': box_outputs,
    })

    # Core detections
    if not is_training:
      # boxes, scores, classes, valid_detections, cuboids = (
      #   self._generate_detections_fn(
      #     box_outputs, class_outputs, rpn_rois,
      #     labels['image_info'][:, 1:2, :], cuboid_outputs=cuboid_outputs))
      #     # TODO even needed cuboid_outputs ? ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      boxes, scores, classes, valid_detections, _ = (
        self._generate_detections_fn(
          box_outputs, class_outputs, rpn_rois,
          labels['image_info'][:, 1:2, :]))
      model_outputs.update({
          'num_detections': valid_detections,
          'detection_boxes': boxes,
          'detection_classes': classes,
          'detection_scores': scores,
      })

    # Do mask sampling and detections
    if self._include_mask:
      if is_training:
        mask_rpn_rois, classes, mask_targets = self._sample_masks_fn(
            rpn_rois, matched_gt_boxes, matched_gt_classes, matched_gt_indices,
            labels['gt_masks'])
        mask_targets = tf.stop_gradient(mask_targets)

        classes = tf.cast(classes, dtype=tf.int32)

        model_outputs.update({
            'mask_targets': mask_targets,
            'sampled_class_targets': classes,
        })
      else:
        # From core detections above
        mask_rpn_rois = boxes
        classes = tf.cast(classes, dtype=tf.int32)

      mask_roi_features = spatial_transform_ops.multilevel_crop_and_resize(
          fpn_features, mask_rpn_rois, output_size=14)

      mask_outputs = self._mrcnn_head_fn(
        mask_roi_features, classes, is_training)

      if is_training:
        model_outputs.update({
            'mask_outputs': mask_outputs,
        })
      else:
        model_outputs.update({
            'detection_masks': tf.nn.sigmoid(mask_outputs)
        })
      
    # Do cuboid sampling and detections
    if self._include_cuboids:
      if is_training:
        # Sample proposals
        cu_rpn_rois, cu_classes, fg_cuboids = self._sample_cuboids_fn(
            rpn_rois, matched_gt_boxes, matched_gt_classes, matched_gt_indices,
            labels['cuboid_targets'])
        fg_cuboids = dict(
          (prop_key, tf.stop_gradient(prop_value))
          for prop_key, prop_value in fg_cuboids.items())

        cu_classes = tf.cast(cu_classes, dtype=tf.int32)

        # FIXME ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.add_scalar_summary('cu_classes_frac_nonzero', 
          tf.reduce_sum(
            tf.cast(tf.greater(cu_classes, 0), tf.float32)) / 
              tf.cast(tf.math.reduce_prod(tf.shape(cu_classes)), tf.float32))

        model_outputs.update({
            'cuboid_targets': fg_cuboids,
            'sampled_cuboid_class_targets': cu_classes,
        })
      else:
        # From core detections above
        cu_rpn_rois = boxes
        cu_classes = tf.cast(classes, dtype=tf.int32)

      # cuboid_outputs = self._cuboid_head_fn(roi_features, is_training)
      CUBOID_PROPS = ('center', 'depth', 'yaw', 'size')
      cu_roi_features = dict(
        (prop_key, 
         spatial_transform_ops.multilevel_crop_and_resize(
            fpn_features, cu_rpn_rois, output_size=14))
        for prop_key in CUBOID_PROPS)
          # TODO need hardcode keys? ~~~~~~~~~~~~~~~~~~
      cuboid_outputs = self._cuboid_head_fn(
        cu_roi_features, cu_classes, is_training)
      model_outputs.update({
        'cuboid_outputs': cuboid_outputs,
      })

    return model_outputs

  def _build_losses(self, features, labels, outputs):
    losses = {}

    losses['rpn_score_loss'] = self._rpn_score_loss_fn(
        outputs['rpn_score_outputs'], labels['rpn_score_targets'])
    losses['rpn_box_loss'] = self._rpn_box_loss_fn(
        outputs['rpn_box_outputs'], labels['rpn_box_targets'])

    losses['fast_rcnn_class_loss'] = self._frcnn_class_loss_fn(
        outputs['class_outputs'], outputs['class_targets'])
    losses['fast_rcnn_box_loss'] = self._frcnn_box_loss_fn(
        outputs['box_outputs'],
        outputs['class_targets'],
        outputs['box_targets'])

    if self._include_cuboids:
      cuboid_losses = self._cuboid_loss_fn(
        outputs['cuboid_outputs'],
        outputs['cuboid_targets'],
        outputs['sampled_cuboid_class_targets'])
      for cuboid_prop, loss_value in cuboid_losses.items():
        loss_value = self._cuboid_total_loss_weight * loss_value
        losses[cuboid_prop + '_loss'] = loss_value

    if self._include_mask:
      losses['mask_loss'] = self._mask_loss_fn(
          outputs['mask_outputs'],
          outputs['mask_targets'],
          outputs['sampled_class_targets'])

    return losses

  def train(self, features, labels):
    # If the input image is transposed (from NHWC to HWCN), we need to revert it
    # back to the original shape before it's used in the computation.
    if self._transpose_input:
      features = tf.transpose(features, [3, 0, 1, 2])

    outputs = self.model_outputs(features, labels, mode=mode_keys.TRAIN)
    losses = self._build_losses(features, labels, outputs)

    # rpn_score_loss = self._rpn_score_loss_fn(
    #     outputs['rpn_score_outputs'], labels['rpn_score_targets'])
    # rpn_box_loss = self._rpn_box_loss_fn(
    #     outputs['rpn_box_outputs'], labels['rpn_box_targets'])

    # frcnn_class_loss = self._frcnn_class_loss_fn(
    #     outputs['class_outputs'], outputs['class_targets'])
    # frcnn_box_loss = self._frcnn_box_loss_fn(
    #     outputs['box_outputs'],
    #     outputs['class_targets'],
    #     outputs['box_targets'])

    # cuboid_loss = 0.0
    # if self._include_cuboids:
    #   cuboid_losses = self._cuboid_loss_fn(
    #     outputs['cuboid_outputs'],
    #     outputs['class_targets'], labels['cuboid_targets'])
    #   for cuboid_prop, loss_value in cuboid_losses.items():
    #     cuboid_loss = cuboid_loss + self._cuboid_total_loss_weight * loss_value
    #     self.add_scalar_summary(cuboid_prop + '_loss', loss_value)

    # if self._include_mask:
    #   mask_loss = self._mask_loss_fn(
    #       outputs['mask_outputs'],
    #       outputs['mask_targets'],
    #       outputs['sampled_class_targets'])
    # else:
    #   mask_loss = 0.0

    # model_loss = (rpn_score_loss + rpn_box_loss + frcnn_class_loss
    #               + frcnn_box_loss + mask_loss + cuboid_loss)

    model_loss = 0.0
    for loss_name, loss_value in losses.items():
      self.add_scalar_summary(loss_name, loss_value)
      model_loss = model_loss + loss_value

    # self.add_scalar_summary('rpn_score_loss', rpn_score_loss)
    # self.add_scalar_summary('rpn_box_loss', rpn_box_loss)
    # self.add_scalar_summary('fast_rcnn_class_loss', frcnn_class_loss)
    # self.add_scalar_summary('fast_rcnn_box_loss', frcnn_box_loss)
    # if self._include_mask:
    #   self.add_scalar_summary('mask_loss', mask_loss)
    self.add_scalar_summary('model_loss', model_loss)

    total_loss, train_op = self.optimize(model_loss)
    scaffold_fn = self.restore_from_checkpoint()
    if self._enable_summary:
      host_call_fn = self.summarize()
    else:
      host_call_fn = None

    return tf.estimator.tpu.TPUEstimatorSpec(
        mode=tf.estimator.ModeKeys.TRAIN,
        loss=total_loss,
        train_op=train_op,
        host_call=host_call_fn,
        scaffold_fn=scaffold_fn)

  def evaluate(self, features, labels):
    raise NotImplementedError('The estimator evaluation is not implemented.')

  def predict(self, features):
    images = features['images']
    labels = features['labels']

    outputs = self.model_outputs(
        images, labels=labels, mode=mode_keys.PREDICT)

    predictions = {
        'pred_image_info': labels['image_info'],
        'pred_num_detections': outputs['num_detections'],
        'pred_detection_boxes': outputs['detection_boxes'],
        'pred_detection_classes': outputs['detection_classes'],
        'pred_detection_scores': outputs['detection_scores'],
    }
    for prop, detections in outputs['detection_cuboids'].items():
      predictions['pred_' + prop] = detections
    if self._include_mask:
      predictions.update({
          'pred_detection_masks': outputs['detection_masks'],
      })

    if 'groundtruths' in labels:
      predictions['pred_source_id'] = labels['groundtruths']['source_id']
      predictions['gt_source_id'] = labels['groundtruths']['source_id']
      predictions['pred_filename_utf8s'] = (
          labels['groundtruths']['filename_utf8s'])
      predictions['gt_image_info'] = labels['image_info']
      predictions['gt_num_detections'] = (
          labels['groundtruths']['num_detections'])
      predictions['gt_boxes'] = labels['groundtruths']['boxes']
      predictions['gt_classes'] = labels['groundtruths']['classes']
      predictions['gt_areas'] = labels['groundtruths']['areas']
      predictions['gt_is_crowds'] = labels['groundtruths']['is_crowds']

      # Log model losses
      losses = self._build_losses(features, labels, outputs)

      model_loss = 0.0
      for loss_name, loss_value in losses.items():
        model_loss = model_loss + loss_value

        # Tiles the loss from [1] to [batch_size] since Estimator requires all
        # predictions have the same batch dimension.
        batch_size = tf.shape(images)[0]
        predictions['loss_' + loss_name] = tf.tile(
          tf.reshape(loss_value, [1]), [batch_size])

      predictions['loss_model_loss'] = tf.tile(
          tf.reshape(model_loss, [1]), [batch_size])

    return tf.estimator.tpu.TPUEstimatorSpec(
        mode=tf.estimator.ModeKeys.PREDICT, predictions=predictions)
