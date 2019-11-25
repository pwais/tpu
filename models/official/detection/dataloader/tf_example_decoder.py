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

"""Tensorflow Example proto decoder for object detection.

A decoder to decode string tensors containing serialized tensorflow.Example
protos for object detection.
"""
import tensorflow.compat.v1 as tf


def _get_source_id_from_encoded_image(parsed_tensors):
  return tf.strings.as_string(
      tf.strings.to_hash_bucket_fast(parsed_tensors['image/encoded'],
                                     2**63 - 1))


class TfExampleDecoder(object):
  """Tensorflow Example proto decoder."""

  def __init__(self,
               include_mask=False,
               regenerate_source_id=False,
               include_cuboids=False,
               include_rv=None):
    self._include_mask = include_mask
    self._include_cuboids = include_cuboids
    self._include_rv = include_rv or []
    self._regenerate_source_id = regenerate_source_id
    self._keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string),
        'image/source_id': tf.FixedLenFeature((), tf.string, ''),
        'image/filename': tf.FixedLenFeature((), tf.string, ''),
        'image/filename_utf8s': tf.VarLenFeature(tf.int64),
        'image/height': tf.FixedLenFeature((), tf.int64, -1),
        'image/width': tf.FixedLenFeature((), tf.int64, -1),
        'image/object/bbox/xmin': tf.VarLenFeature(tf.float32),
        'image/object/bbox/xmax': tf.VarLenFeature(tf.float32),
        'image/object/bbox/ymin': tf.VarLenFeature(tf.float32),
        'image/object/bbox/ymax': tf.VarLenFeature(tf.float32),
        'image/object/class/label': tf.VarLenFeature(tf.int64),
        'image/object/area': tf.VarLenFeature(tf.float32),
        'image/object/is_crowd': tf.VarLenFeature(tf.int64),
    }
    if include_mask:
      self._keys_to_features.update({
          'image/object/mask':
              tf.VarLenFeature(tf.string),
      })
    if include_cuboids:
      self._keys_to_features.update({
          'image/object/cuboid/cam/center_x': tf.VarLenFeature(tf.float32),
          'image/object/cuboid/cam/center_y': tf.VarLenFeature(tf.float32),
          'image/object/cuboid/cam/center_z': tf.VarLenFeature(tf.float32),
          'image/object/cuboid/cam/yaw': tf.VarLenFeature(tf.float32),

          'image/object/cuboid/length': tf.VarLenFeature(tf.float32),
          'image/object/cuboid/width': tf.VarLenFeature(tf.float32),
          'image/object/cuboid/height': tf.VarLenFeature(tf.float32),

          'image/object/cuboid/box/center_x': tf.VarLenFeature(tf.float32),
          'image/object/cuboid/box/center_y': tf.VarLenFeature(tf.float32),
          'image/object/cuboid/box/center_depth': tf.VarLenFeature(tf.float32),
          'image/object/cuboid/box/perspective_yaw':
              tf.VarLenFeature(tf.float32),
      })
    for img_type in self._include_rv:
      self._keys_to_features.update({
        'rv_image/%s/height' % img_type: tf.FixedLenFeature((), tf.int64),
        'rv_image/%s/width' % img_type: tf.FixedLenFeature((), tf.int64),
        'rv_image/%s/channels' % img_type: tf.FixedLenFeature((), tf.int64),
        'rv_image/%s/encoded' % img_type: tf.FixedLenFeature((), tf.string),
        'rv_image/%s/format' % img_type: tf.FixedLenFeature((), tf.string),
      })

  def _decode_image(self, parsed_tensors):
    """Decodes the image and set its static shape."""
    image = tf.io.decode_image(parsed_tensors['image/encoded'], channels=3)
    image.set_shape([None, None, 3])
    return image

  def _decode_boxes(self, parsed_tensors):
    """Concat box coordinates in the format of [ymin, xmin, ymax, xmax]."""
    xmin = parsed_tensors['image/object/bbox/xmin']
    xmax = parsed_tensors['image/object/bbox/xmax']
    ymin = parsed_tensors['image/object/bbox/ymin']
    ymax = parsed_tensors['image/object/bbox/ymax']
    return tf.stack([ymin, xmin, ymax, xmax], axis=-1)

  def _decode_masks(self, parsed_tensors):
    """Decode a set of PNG masks to the tf.float32 tensors."""
    def _decode_png_mask(png_bytes):
      mask = tf.squeeze(
          tf.io.decode_png(png_bytes, channels=1, dtype=tf.uint8), axis=-1)
      mask = tf.cast(mask, dtype=tf.float32)
      mask.set_shape([None, None])
      return mask

    height = parsed_tensors['image/height']
    width = parsed_tensors['image/width']
    masks = parsed_tensors['image/object/mask']
    return tf.cond(
        tf.greater(tf.size(masks), 0),
        lambda: tf.map_fn(_decode_png_mask, masks, dtype=tf.float32),
        lambda: tf.zeros([0, height, width], dtype=tf.float32))

  def _decode_areas(self, parsed_tensors):
    xmin = parsed_tensors['image/object/bbox/xmin']
    xmax = parsed_tensors['image/object/bbox/xmax']
    ymin = parsed_tensors['image/object/bbox/ymin']
    ymax = parsed_tensors['image/object/bbox/ymax']
    return tf.cond(
        tf.greater(tf.shape(parsed_tensors['image/object/area'])[0], 0),
        lambda: parsed_tensors['image/object/area'],
        lambda: (xmax - xmin) * (ymax - ymin))

  def decode(self, serialized_example):
    """Decode the serialized example.

    Args:
      serialized_example: a single serialized tf.Example string.

    Returns:
      decoded_tensors: a dictionary of tensors with the following fields:
        - image: a uint8 tensor of shape [None, None, 3].
        - source_id: a string scalar tensor.
        - height: an integer scalar tensor.
        - width: an integer scalar tensor.
        - groundtruth_classes: a int64 tensor of shape [None].
        - groundtruth_is_crowd: a bool tensor of shape [None].
        - groundtruth_area: a float32 tensor of shape [None].
        - groundtruth_boxes: a float32 tensor of shape [None, 4].
        - groundtruth_instance_masks: a float32 tensor of shape
            [None, None, None].
        - groundtruth_instance_masks_png: a string tensor of shape [None].
    """
    parsed_tensors = tf.io.parse_single_example(
        serialized_example, self._keys_to_features)
    for k in parsed_tensors:
      if isinstance(parsed_tensors[k], tf.SparseTensor):
        if parsed_tensors[k].dtype == tf.string:
          parsed_tensors[k] = tf.sparse_tensor_to_dense(
              parsed_tensors[k], default_value='')
        else:
          parsed_tensors[k] = tf.sparse_tensor_to_dense(
              parsed_tensors[k], default_value=0)

    image = self._decode_image(parsed_tensors)
    boxes = self._decode_boxes(parsed_tensors)
    areas = self._decode_areas(parsed_tensors)

    decode_image_shape = tf.logical_or(
        tf.equal(parsed_tensors['image/height'], -1),
        tf.equal(parsed_tensors['image/width'], -1))
    image_shape = tf.cast(tf.shape(image), dtype=tf.int64)

    parsed_tensors['image/height'] = tf.where(decode_image_shape,
                                              image_shape[0],
                                              parsed_tensors['image/height'])
    parsed_tensors['image/width'] = tf.where(decode_image_shape, image_shape[1],
                                             parsed_tensors['image/width'])

    is_crowds = tf.cond(
        tf.greater(tf.shape(parsed_tensors['image/object/is_crowd'])[0], 0),
        lambda: tf.cast(parsed_tensors['image/object/is_crowd'], dtype=tf.bool),
        lambda: tf.zeros_like(parsed_tensors['image/object/class/label'], dtype=tf.bool))  # pylint: disable=line-too-long
    if self._regenerate_source_id:
      source_id = _get_source_id_from_encoded_image(parsed_tensors)
    else:
      source_id = tf.cond(
          tf.greater(tf.strings.length(parsed_tensors['image/source_id']),
                     0), lambda: parsed_tensors['image/source_id'],
          lambda: _get_source_id_from_encoded_image(parsed_tensors))
    filename = parsed_tensors['image/filename']
    filename_utf8s = parsed_tensors['image/filename_utf8s']
    # filename_utf8s.set_shape([10000])
    if self._include_mask:
      masks = self._decode_masks(parsed_tensors)

    decoded_tensors = {
        'image': image,
        'filename': filename,
        'filename_utf8s': filename_utf8s,
        'source_id': source_id,
        'height': parsed_tensors['image/height'],
        'width': parsed_tensors['image/width'],
        'groundtruth_classes': parsed_tensors['image/object/class/label'],
        'groundtruth_is_crowd': is_crowds,
        'groundtruth_area': areas,
        'groundtruth_boxes': boxes,
    }
    if self._include_mask:
      decoded_tensors.update({
          'groundtruth_instance_masks': masks,
          'groundtruth_instance_masks_png': parsed_tensors['image/object/mask'],
      })
    if self._include_cuboids:
      keys = (
        'image/object/cuboid/cam/center_x',
        'image/object/cuboid/cam/center_y',
        'image/object/cuboid/cam/center_z',
        'image/object/cuboid/cam/yaw',

        'image/object/cuboid/length',
        'image/object/cuboid/width',
        'image/object/cuboid/height',

        'image/object/cuboid/box/center_x',
        'image/object/cuboid/box/center_y',
        'image/object/cuboid/box/center_depth',
        'image/object/cuboid/box/perspective_yaw',
      )
      for key in keys:
        # E.g. image/object/cuboid/length -> cuboid/length
        PREFIX = 'image/object/'
        tensor_key = key[len(PREFIX):]
        decoded_tensors[tensor_key] = parsed_tensors[key]
    if self._include_rv:
      for img_type in self._include_rv:
        # Grab the image, which might be a JPEG or PNG
        # (which might have only one useful channel)
        rv_image = tf.io.decode_image(
                      parsed_tensors['rv_image/%s/encoded' % img_type],
                      channels=3)
        rv_image.set_shape([None, None, 3])
        decoded_tensors.update({
          'rv_image/%s/image' % img_type: rv_image,
        })

        # Unlike above, we *require* rv image dimensions to be set, so we
        # can just copy them.
        COPY_KEYS = (
          'rv_image/%s/height' % img_type,
          'rv_image/%s/width' % img_type,
          'rv_image/%s/channels' % img_type,
        )
        for key in COPY_KEYS:
          decoded_tensors[key] = parsed_tensors[key]

    return decoded_tensors
