import glob
import os
import sys
import time as t
from multiprocessing import Queue, Pool

import cv2
import imageio
import numpy as np

import tensorflow as tf

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

from utils import label_map_util
from utils import visualization_utils as vis_util

# global variable
should_optimize = False


def optimize_image(img, x_boundary, y_boundary, scale_factor):
    """
    define ROI in the image
    crop ROI
    """
    is_debug_enabled = False

    # original image dimensions
    if is_debug_enabled:
        print("original dim: ", img.shape)

    # reduce dimensions
    img = cv2.resize(img, None,
                     fx=scale_factor, fy=scale_factor,
                     interpolation=cv2.INTER_LINEAR)

    if is_debug_enabled:
        img = cv2.rectangle(img,
                            (x_boundary[0], y_boundary[0]),
                            (x_boundary[1], y_boundary[1]),
                            color=(0, 0, 255),
                            thickness=3)

    # crop ROI
    height = abs(y_boundary[1] - y_boundary[0])
    width = abs(x_boundary[1] - x_boundary[0])
    img = img[
          int(y_boundary[0]):int(y_boundary[0] + height),
          int(x_boundary[0]):int(x_boundary[0] + width)
          ]

    # cropped image dimensions
    if is_debug_enabled:
        print("dimension: {}".format(img.shape))
        cv2.imshow("image", img)
        cv2.waitKey()

    return img


def detect_object(image_np, sess, detection_graph, xy_boundary, category_index, scale_factor):
    """
    get optimized image
    detect bounding boxes with scores
    draw bounding boxes on image
    """

    # Definite input and output Tensors for detection_graph
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    if should_optimize:
        image_np = optimize_image(image_np, xy_boundary[0], xy_boundary[1], scale_factor)
    image_np_expanded = np.expand_dims(image_np, axis=0)

    # Each box represents a part of the image where a particular object was detected.
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    # Actual detection
    (boxes, scores, classes, num) = sess.run([detection_boxes, detection_scores, detection_classes, num_detections],
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


def worker(input_q, output_q, path_to_check_point, xy_boundary, category_index, scale_factor):
    detection_graph = tf.Graph()
    # runs only once
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(path_to_check_point, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
        sess = tf.Session(graph=detection_graph)

    while True:
        frame = input_q.get()

        # takes around 190 ms for image with dimension (1280, 720, 3)
        output_q.put(detect_object(frame, sess, detection_graph, xy_boundary, category_index, scale_factor))


def handle_queues(input_q, output_q, input_frame):
    # start time
    t_start = t.time()
    # put frame in queue`
    input_q.put(input_frame)
    # get frame with bounding boxes
    output_frame = output_q.get()
    # start time
    t_end = t.time()
    print("time (ms): {:0.2f}".format((t_end - t_start) * 1000))
    # visualization (matplotlib is very slow)
    cv2.imshow("detection", output_frame)
    cv2.waitKey(1)


def __main__():
    """
    Feature Extractor: MobileNets
    Meta-architecture: SSD
    Dataset: MS-COCO
    """

    # flags
    use_image, use_video = True, False

    # directory of images/videos
    path_to_test = {
        "root": "object_detection/test_images/",
        "pet": 'object_detection/test_images/pet-bottles/',
        "krones": "object_detection/test_images/dt3/"
    }

    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    models = {
        "ssd_mobilenet": "object_detection/weights/ssd_mobilenet_v1_coco_2017_11_17",
        "ssd_inception": "object_detection/weights/ssd_inception_v2_coco_2017_11_17",
        "rcnn_inception": "object_detection/weights/faster_rcnn_inception_v2_coco_2017_11_08",
        "rcnn_resnet": "object_detection/weights/faster_rcnn_resnet50_coco_2017_11_08",
        "raccoon": "object_detection/weights/ssd_inception_v2_racoon",
        "krones": "krones/models/ssd_mobilenet_v1_coco/frozen_graph"
    }

    # label maps for different datasets
    labels = {
        "mscoco": "object_detection/data/mscoco_label_map.pbtxt",
        "kitti": "object_detection/data/kitti_label_map.pbtxt",
        "pet": "object_detection/data/pet_label_map.pbtxt",
        "raccoon": "object_detection/data/racoon_label_map.pbtxt",
        "krones": "krones/data/label_map.pbtxt"
    }

    checkpoint_path = models["krones"] + '/frozen_inference_graph.pb'

    # List of the strings that is used to add correct label for each box.
    path_to_labels = labels["krones"]

    num_classes = 90

    # resolution -- has impact on inference time
    scale_factor = 0.5

    # ROI -- has impact on inference time
    x_boundary = (1 * scale_factor, 1270 * scale_factor)
    y_boundary = (1 * scale_factor, 700 * scale_factor)
    xy_boundary = x_boundary, y_boundary

    # test images
    image_paths = [name for name in glob.glob(path_to_test["krones"] + '*')]

    # image_paths = [os.path.join(path_to_test["root"], 'raccoon-{}.jpg'.format(i)) for i in range(184, 187)]
    label_map = label_map_util.load_labelmap(path_to_labels)
    categories = label_map_util.convert_label_map_to_categories(label_map,
                                                                max_num_classes=num_classes,
                                                                use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    queue_size = 5
    input_q, output_q = Queue(maxsize=queue_size), Queue(maxsize=queue_size)
    pool = Pool(
        # number of queues
        2,
        # worker
        worker,
        # arguments of worker
        (input_q, output_q, checkpoint_path, xy_boundary, category_index, scale_factor)
    )

    # for images
    if use_image:
        for image_path in image_paths:
            print("image: ", image_path)
            if os.path.isfile(image_path):
                handle_queues(input_q, output_q, cv2.imread(image_path))
                cv2.waitKey(0)

    # for video
    if use_video:
        video_cap = imageio.get_reader(path_to_test["root"] + "project_video.mp4")

        # for each frame, get bounding boxes
        for index, frame in enumerate(video_cap):
            # drop every 2nd frame to improve FPS
            if index % 2 == 0:
                handle_queues(input_q, output_q, frame)
        video_cap.stop()

    # cleaning tasks
    pool.terminate()
    cv2.destroyAllWindows()


__main__()
