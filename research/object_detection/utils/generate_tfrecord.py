"""

Implementation taken from:
    object_detection/g3doc/using_your_own_dataset.md

Usage:
  # From models/research/
  # Create train data:
  python object_detection/utils/generate_tfrecord.py --dir=dataset_krones --csv_name=train_labels.csv

  # Create test data:
  python object_detection/utils/generate_tfrecord.py --dir=dataset_krones --csv_name=test_labels.csv
"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import io
import pandas as pd
import tensorflow as tf

from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple

flags = tf.app.flags
flags.DEFINE_string('dir', '', 'Path data/.csv and images/.jpg root')
flags.DEFINE_string('csv_name', '', 'Name of CSV file')
FLAGS = flags.FLAGS


# TO-DO replace this with label map
def class_text_to_int(row_label):
    if row_label == 'Fallen':
        return 1
    elif row_label == 'Standing':
        return 2


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path):
    _path_ = os.path.join(path, '{}'.format(group.filename))

    with tf.gfile.GFile(_path_, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)

    # features
    image = Image.open(encoded_jpg_io)

    # dimensions
    width, height = image.size

    filename = group.filename.encode('utf8')

    image_format = b'jpg'
    xmins, xmaxs, ymins, ymaxs = [], [], [], []
    classes_text, classes = [], []

    # normalized bounding boxes
    # class for each bounding box
    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))

    # generate TFRecord
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def main(_):
    cwd = os.getcwd() + '/'
    output_path = cwd + FLAGS.dir + '/tf.record'
    csv_input = cwd + FLAGS.dir + '/data/' + FLAGS.csv_name
    image_dir = cwd + FLAGS.dir + '/images'
    writer = tf.python_io.TFRecordWriter(output_path)

    examples = pd.read_csv(csv_input)
    grouped = split(examples, 'filename')

    for group in grouped:
        tf_example = create_tf_example(group, image_dir)
        writer.write(tf_example.SerializeToString())

    writer.close()
    print('Successfully created the TFRecords: {}'.format(output_path))


if __name__ == '__main__':
    tf.app.run()
