import glob

import cv2
import imgaug as ia
import numpy as np
import pandas as pd
from imgaug import augmenters as iaa
from random import randint

# possible augmentations
orientations = [90, 180, 270]
brightnesses = [0.30, 0.60, 0.90, 1.20, 1.50, 1.80]
scales = [0.4, 0.6, 0.8]
# window labels for visualization
window_labels = [
    "angle 90", "angle 180", "angle 270",
    "brightness 0.30", "brightness 0.60", "brightness 0.90",
    "brightness 1.20", "brightness 1.50", "brightness 1.80",
    "scale 0.4", "scale 0.6", "scale 0.8",
]


class AugmentDataset:
    # image increment variable
    multiplier = 1

    @staticmethod
    def pad(image, by):
        """Pad image with a 1px white and (BY-1)px black border"""
        if by <= 0:
            return image
        image_border1 = np.pad(
            image, ((1, 1), (1, 1), (0, 0)),
            mode="constant", constant_values=255
        )
        image_border2 = np.pad(
            image_border1, ((by - 1, by - 1), (by - 1, by - 1), (0, 0)),
            mode="constant", constant_values=0
        )
        return image_border2

    @staticmethod
    def get_classes(image_name, full_labels):
        selected_value = full_labels[full_labels.filename == image_name]
        ss = selected_value.iterrows()
        return [row['class'] for index, row in ss]

    @staticmethod
    def get_boxes(image_name, full_labels):
        selected_value = full_labels[full_labels.filename == image_name]
        ss = selected_value.iterrows()
        return [ia.BoundingBox(row['xmin'], row['ymin'], row['xmax'], row['ymax']) for index, row in ss]

    @staticmethod
    def draw_boxes(img, bboxes, classes):
        _ = np.copy(img)
        for bbox, cls in zip(bboxes, classes):
            color = (0, 0, 255) if cls == 'fallen' else (0, 255, 0)
            cv2.rectangle(_, (bbox.x1, bbox.y1), (bbox.x2, bbox.y2), color, 3)
        return _

    @staticmethod
    def augment_orientation(image, bounding_boxes, angle=90):
        # rotation values
        seq = iaa.Sequential([iaa.Affine(rotate=angle)])
        seq_det = seq.to_deterministic()
        # apply rotation on the image
        img_aug = seq_det.augment_images([image])[0]
        # update the bounding boxes
        bbs_aug = seq_det.augment_bounding_boxes([bounding_boxes])[0].remove_out_of_image().cut_out_of_image()
        # return image and bounding boxes
        return img_aug, bbs_aug

    @staticmethod
    def recursive_augment_orientation(image, bbs, augmented_images, augmented_boxes):
        # orientation augmentations
        for _ in orientations:
            # first stage augmentation
            ag, bbx = AugmentDataset.augment_orientation(image, bbs, _)
            augmented_images.append(ag), augmented_boxes.append(bbx)
            # second stage augmentation
            for __ in brightnesses:
                ag2, bbx2 = AugmentDataset.augment_brightness(ag, bbx, __)
                augmented_images.append(ag2), augmented_boxes.append(bbx2)
            for __ in scales:
                ag2, bbx2 = AugmentDataset.augment_brightness(ag, bbx, __)
                augmented_images.append(ag2), augmented_boxes.append(bbx2)
        # return appended results
        return augmented_images, augmented_boxes

    @staticmethod
    def augment_brightness(image, bounding_boxes, brightness=0.50):
        # rotation values
        seq = iaa.Sequential([iaa.Multiply(brightness)])
        seq_det = seq.to_deterministic()
        # apply rotation on the image
        img_aug = seq_det.augment_images([image])[0]
        # return image and bounding boxes
        return img_aug, bounding_boxes

    @staticmethod
    def recursive_augment_brightness(image, bbs, augmented_images, augmented_boxes):
        # first stage augmentation
        for _ in brightnesses:
            ag, bbx = AugmentDataset.augment_brightness(image, bbs, _)
            augmented_images.append(ag), augmented_boxes.append(bbx)
            # second stage augmentation
            for __ in orientations:
                ag2, bbx2 = AugmentDataset.augment_orientation(ag, bbx, __)
                augmented_images.append(ag2), augmented_boxes.append(bbx2)
            for __ in scales:
                ag2, bbx2 = AugmentDataset.augment_scale(ag, bbx, __)
                augmented_images.append(ag2), augmented_boxes.append(bbx2)
        # return image and bounding boxes
        return augmented_images, augmented_boxes

    @staticmethod
    def augment_scale(image, bounding_boxes, scale=0.50):
        # scaling values
        seq = iaa.Sequential([iaa.Affine(scale={"x": scale, "y": scale})])
        seq_det = seq.to_deterministic()
        # apply rotation on the image
        img_aug = seq_det.augment_images([image])[0]
        bbs_aug = seq_det.augment_bounding_boxes([bounding_boxes])[0].remove_out_of_image().cut_out_of_image()
        # return image and bounding boxes
        return img_aug, bbs_aug

    @staticmethod
    def recursive_augment_scale(image, bbs, augmented_images, augmented_boxes):
        # first stage augmentation
        for _ in scales:
            ag, bbx = AugmentDataset.augment_scale(image, bbs, _)
            augmented_images.append(ag), augmented_boxes.append(bbx)
            # second stage augmentation
            for __ in orientations:
                ag2, bbx2 = AugmentDataset.augment_orientation(ag, bbx, __)
                augmented_images.append(ag2), augmented_boxes.append(bbx2)
            for __ in brightnesses:
                ag2, bbx2 = AugmentDataset.augment_brightness(ag, bbx, __)
                augmented_images.append(ag2), augmented_boxes.append(bbx2)
        # return image and bounding boxes
        return augmented_images, augmented_boxes

    @staticmethod
    def create_augmentations(filename, image_dir, full_labels):
        """
        generate augmentations the given image
        augmentations are:
            3 orientations
            2 brightness
            2 scales
        """
        image = cv2.imread(image_dir + "/" + filename)
        # bounding box for the original image
        bbs = ia.BoundingBoxesOnImage(AugmentDataset.get_boxes(filename, full_labels), shape=image.shape)
        # augmentation containers
        augmented_images, augmented_boxes = [], []
        # recursive orientation augmentations
        augmented_images, augmented_boxes = AugmentDataset.recursive_augment_orientation(image,
                                                                                         bbs,
                                                                                         augmented_images,
                                                                                         augmented_boxes)
        # recursive brightness augmentations
        augmented_images, augmented_boxes = AugmentDataset.recursive_augment_brightness(image,
                                                                                        bbs,
                                                                                        augmented_images,
                                                                                        augmented_boxes)
        # recursive scale augmentations
        augmented_images, augmented_boxes = AugmentDataset.recursive_augment_scale(image,
                                                                                   bbs,
                                                                                   augmented_images,
                                                                                   augmented_boxes)
        # augmentations and their bounding boxes for the original image
        return augmented_images, augmented_boxes, window_labels

    @staticmethod
    def viz_augmentations(augmented_images, augmented_boxes, classes):
        for img, bbx in zip(augmented_images, augmented_boxes):
            title = "{}".format(randint(0, 10000))
            cv2.imshow(title, AugmentDataset.draw_boxes(img, bbx.bounding_boxes, classes))
            cv2.waitKey()

    @staticmethod
    def viz_boxes(filenames, full_labels):
        for filename in filenames:
            # read image
            image = cv2.imread(filename)
            # read bounding box
            bbs = ia.BoundingBoxesOnImage(AugmentDataset.get_boxes(filename.rsplit('/', 1)[-1], full_labels),
                                          shape=image.shape)
            # read classes
            classes = AugmentDataset.get_classes(filename.rsplit('/', 1)[-1], full_labels)
            # show image with boxes
            cv2.imshow("image", AugmentDataset.draw_boxes(image, bbs.bounding_boxes, classes))
            cv2.waitKey()

    @staticmethod
    def save_augmentations(augmented_images, augmented_boxes, classes):
        count = len(augmented_images)
        for index in range(count):
            _dir_ = "/Users/siddiqui/Documents/Projects/tensorflow/models/research/krones/dataset/train/images_debug/"
            window_title = "{}{}.{}".format(_dir_, str(AugmentDataset.multiplier * index), "jpg")
            _img_ = AugmentDataset.draw_boxes(augmented_images[index], augmented_boxes[index].bounding_boxes, classes)
            cv2.imwrite(window_title, _img_)
        AugmentDataset.multiplier += 1


def __main__():
    image_dir = "/Users/siddiqui/Documents/Projects/tensorflow/models/research/krones/dataset/train/images"
    img_names = glob.glob(image_dir + "/*.jpg")
    full_labels = pd.read_csv('../data/train.csv')
    full_labels.head()
    ia.seed(1)
    AugmentDataset.viz_boxes(img_names, full_labels)


# __main__()
