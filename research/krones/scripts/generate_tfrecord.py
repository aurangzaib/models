from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import os
from collections import namedtuple

import cv2
import pandas as pd
from PIL import Image

import tensorflow as tf
from krones.scripts.img_aug import AugmentDataset
from object_detection.utils import dataset_util
from random import randint
import matplotlib.pyplot as plt

flags = tf.app.flags
flags.DEFINE_string('dir', '', 'Path data/.csv and images/.jpg root')
flags.DEFINE_string('csv', '', 'Name of CSV file')
flags.DEFINE_string('output', '', 'Name of TFRecord file')
FLAGS = flags.FLAGS

# control flags
SAVE_RECORD = False
SAVE_AUGMENTED_RECORD = False
SAVE_AUGMENTED_IMAGES = False
SHOW_VISUALIZATION = False

"""
Implementation taken from:
    object_detection/g3doc/using_your_own_dataset.md

Explanation:
    Read name of images from CSV file
    Get dimension and class of each bounding box in each image
    Save result as TFRecord
    
Usage:
  # From models/research/
  # Create train data:
  python dataset_krones_2/scripts/generate_tfrecord.py \
                                --dir=${krones_train_dataset} \
                                --csv=${krones_train_csv} \
                                --output=${krones_train_record} \

  # Create test data:
  python dataset_krones_2/scripts/generate_tfrecord.py --dir=${krones_eval_dataset} --csv=${krones_eval_csv} --output=${krones_eval_record}
"""


class GenerateTFRecord:
    @staticmethod
    def class_text_to_int(row_label):
        if row_label == 'fallen':
            return 1
        elif row_label == 'standing':
            return 2
        else:
            None

    @staticmethod
    def split(df, group):
        data = namedtuple('data', ['filename', 'object'])
        gb = df.groupby(group)
        return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]

    @staticmethod
    def dataset_overview(grouped):
        fallen, standing, examples = 0, 0, range(len(grouped))
        fallen_list, standing_list = [], []
        for group in grouped:
            for index, row in group.object.iterrows():
                class_name = row['class'].encode('utf8')
                if class_name == 'fallen':
                    fallen += 1
                elif class_name == 'standing':
                    standing += 1

            fallen_list.append(fallen), standing_list.append(standing)
            fallen, standing = 0, 0

        # stack-bar graph
        plt.bar(examples, fallen_list, 0.35, color='r')
        plt.bar(examples, standing_list, 0.35, bottom=fallen, color='g')
        plt.show()

    @staticmethod
    def create_tf_record(group, path):
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
            classes.append(GenerateTFRecord.class_text_to_int(row['class']))

        # generate TFRecord
        bounds = [xmins, xmaxs, ymins, ymaxs]
        tf_example = GenerateTFRecord.create_features(height, width, filename,
                                                      image_format, encoded_jpg,
                                                      bounds, classes_text, classes)
        return tf_example

    @staticmethod
    def create_tf_record_augmented(image, image_dir, bbox, filename, cls):
        # full path of the images
        path = os.path.join(image_dir, '{}'.format("tmp.jpg"))
        # save image so that can be read later using GFile
        # using GFile reduces the TFRecord size significantly
        # TODO: find a way to avoid saving image using opencv and re-reading it again using GFile
        cv2.imwrite(path, image)

        # read encoded jpg files
        with tf.gfile.GFile(path, 'rb') as fid:
            encoded_jpg = fid.read()
        encoded_jpg_io = io.BytesIO(encoded_jpg)

        # features
        image = Image.open(encoded_jpg_io)

        # dimensions
        width, height = image.size

        # format
        image_format = b'jpg'

        # bounding boxes
        xmins, xmaxs, ymins, ymaxs = [], [], [], []
        classes_text, classes = [], []
        # normalized bounding boxes
        # class for each bounding box
        for index, row in enumerate(bbox):
            xmins.append(row.x1 / width)
            xmaxs.append(row.x2 / width)
            ymins.append(row.y1 / height)
            ymaxs.append(row.y2 / height)
            classes_text.append(cls[index].encode('utf8'))
            classes.append(GenerateTFRecord.class_text_to_int(cls[index]))

        # generate TFRecord
        bounds = [xmins, xmaxs, ymins, ymaxs]
        tf_example = GenerateTFRecord.create_features(height, width, filename,
                                                      image_format, encoded_jpg,
                                                      bounds, classes_text, classes)
        return tf_example

    @staticmethod
    def create_features(height, width, filename, image_format, encoded_jpg, bounds, classes_text, classes):
        return tf.train.Example(features=tf.train.Features(feature={
            'image/height': dataset_util.int64_feature(height),
            'image/width': dataset_util.int64_feature(width),
            'image/filename': dataset_util.bytes_feature(filename),
            'image/source_id': dataset_util.bytes_feature(filename),
            'image/encoded': dataset_util.bytes_feature(encoded_jpg),
            'image/format': dataset_util.bytes_feature(image_format),
            'image/object/bbox/xmin': dataset_util.float_list_feature(bounds[0]),
            'image/object/bbox/xmax': dataset_util.float_list_feature(bounds[1]),
            'image/object/bbox/ymin': dataset_util.float_list_feature(bounds[2]),
            'image/object/bbox/ymax': dataset_util.float_list_feature(bounds[3]),
            'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
            'image/object/class/label': dataset_util.int64_list_feature(classes),
        }))

    @staticmethod
    def write_tf_record(output_path, image_dir, grouped, full_labels):
        writer = tf.python_io.TFRecordWriter(output_path)
        for group in grouped:
            filename = group.filename.encode('utf8')
            # classes for the image
            classes = AugmentDataset.get_classes(filename, full_labels)
            # create original tf_record
            tf_example = GenerateTFRecord.create_tf_record(group, image_dir)
            # save tf_record
            writer.write(tf_example.SerializeToString())
            if SAVE_AUGMENTED_RECORD:
                # augment dataset
                ag_img, ag_bbx, w_lb = AugmentDataset.create_augmentations(filename, image_dir, full_labels)
                # visualize augmented dataset
                AugmentDataset.viz_augmentations(ag_img, ag_bbx, classes) if SHOW_VISUALIZATION else None
                # show augmented dataset
                AugmentDataset.save_augmentations(ag_img, ag_bbx, classes) if SAVE_AUGMENTED_IMAGES else None
                # generate augmentations of the image
                for img, bbx, lb in zip(ag_img, ag_bbx, w_lb):
                    # create augmented tf_record
                    full_filename = "{}-{}-{}".format(filename, lb, randint(0, 100000))
                    ag_tf_example = GenerateTFRecord.create_tf_record_augmented(img,
                                                                                image_dir,
                                                                                bbx.bounding_boxes,
                                                                                full_filename,
                                                                                classes)
                    # save tf_record
                    writer.write(ag_tf_example.SerializeToString())
            # close file
        writer.close()
        print('Successfully created the TFRecords: {}'.format(output_path))


def main(_):
    # directories
    output_path = FLAGS.output
    csv_input = FLAGS.csv
    image_dir = FLAGS.dir
    # labels for dataset
    full_labels = pd.read_csv(csv_input)
    full_labels.head()

    examples = pd.read_csv(csv_input)
    grouped = GenerateTFRecord.split(examples, 'filename')

    GenerateTFRecord.dataset_overview(grouped)
    # augment, generate and write tf_record files as train.record
    GenerateTFRecord.write_tf_record(output_path, image_dir, grouped, full_labels) if SAVE_RECORD else None


if __name__ == '__main__':
    tf.app.run()
