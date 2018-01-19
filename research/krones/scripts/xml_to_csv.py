import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET
import tensorflow as tf

"""
Usage (from models/research/):

  # Create train data:
  python krones/scripts/xml_to_csv.py \
                        --xml_dir=${krones_train_annotations} \
                        --csv=${krones_train_csv}

  # Create test data:
  python krones/scripts/xml_to_csv.py --xml_dir=${krones_eval_annotations} --csv=${krones_eval_csv}
"""

flags = tf.app.flags
flags.DEFINE_string('xml_dir', '', 'XML files directory')
flags.DEFINE_string('csv', '', 'Filename for generated CSV')
FLAGS = flags.FLAGS


def xml_to_csv(path):
    xml_list = []
    xml_files = glob.glob(path + '/*.xml')
    for xml_file in xml_files:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            is_bndbox = str(member[4]).strip().find('bndbox') != -1
            if is_bndbox:
                value = (root.find('filename').text,
                         int(root.find('size')[0].text),
                         int(root.find('size')[1].text),
                         member[0].text,
                         int(member[4][0].text),
                         int(member[4][1].text),
                         int(member[4][2].text),
                         int(member[4][3].text)
                         )
                xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def main():
    xml_dir = FLAGS.xml_dir
    csv = FLAGS.csv
    xml_df = xml_to_csv(xml_dir)
    xml_df.to_csv(csv, index=None)
    print('Successfully converted xml to csv.')


main()
