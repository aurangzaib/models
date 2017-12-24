import os
import sys
import time

import cv2
import imageio
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from multiprocessing import Queue, Pool

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

from utils import label_map_util
from utils import visualization_utils as vis_util


def optimize_image(img, x_start_stop, y_start_stop):
    # reduce dimensions
    img = cv2.resize(img, None,
                     fx=0.3, fy=0.3,
                     interpolation=cv2.INTER_LINEAR)
    # crop ROI
    img = cv2.rectangle(img,
                        (x_start_stop[0], y_start_stop[0]),
                        (x_start_stop[1], y_start_stop[1]),
                        color=(0, 0, 255),
                        thickness=3)
    height = abs(y_start_stop[1] - y_start_stop[0])
    width = abs(x_start_stop[1] - x_start_stop[0])

    img = img[
          int(y_start_stop[0]):int(y_start_stop[0] + height),
          int(x_start_stop[0]):int(x_start_stop[0] + width)
          ]

    is_debug_enabled = False
    if is_debug_enabled:
        cv2.imshow("image", img)
        cv2.waitKey()

    return img


def detect_object(image_np, sess, detection_graph):
    # Definite input and output Tensors for detection_graph
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np = optimize_image(image_np, xy_start[0], xy_start[1])
    image_np_expanded = np.expand_dims(image_np, axis=0)

    # Each box represents a part of the image where a particular object was detected.
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    # Actual detection.
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})

    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8)

    return image_np


def worker(input_q, output_q):
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.Session(graph=detection_graph)

    while True:
        frame = input_q.get()
        output_q.put(detect_object(frame, sess, detection_graph))

    sess.close()


# Path to frozen detection graph. This is the actual model that is used for the object detection.
MODEL_NAME = 'ssd_mobilenet_weight'
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90
xy_start = (0, 630), (120, 280)

PATH_TO_TEST_IMAGES_DIR = 'test_images/'
TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image3.jpg'.format(i)) for i in range(0, 1)]

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map,
                                                            max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)

queue_size = 1
input_q = Queue(maxsize=queue_size)
output_q = Queue(maxsize=queue_size)
pool = Pool(2, worker, (input_q, output_q))

# capture video
video_cap = imageio.get_reader(PATH_TO_TEST_IMAGES_DIR + "project_video.mp4")

for frame in video_cap:
    input_q.put(frame)
    output = output_q.get()
    plt.imshow(output)
    plt.pause(0.00001)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
plt.show()
pool.terminate()
video_cap.stop()
cv2.destroyAllWindows()
