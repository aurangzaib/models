import glob

import cv2
import imgaug as ia
import numpy as np
import pandas as pd
from imgaug import augmenters as iaa


class AugmentDataset:

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
    def augment_brightness(image, bounding_boxes, brightness=0.50):
        # rotation values
        seq = iaa.Sequential([iaa.Multiply(brightness)])
        seq_det = seq.to_deterministic()
        # apply rotation on the image
        img_aug = seq_det.augment_images([image])[0]
        # return image and bounding boxes
        return img_aug, bounding_boxes

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
    def perform_augmentation(img_name, full_labels):
        img = cv2.imread('../dataset/train/images/' + img_name)
        bbs = ia.BoundingBoxesOnImage(AugmentDataset.get_boxes(img_name, full_labels), shape=img.shape)
        classes = AugmentDataset.get_classes(img_name, full_labels)
        orientations = [90, 180, 270]
        brightnesses = [0.40, 1.60]
        scales = [0.5, 0.75]

        for _ in orientations:
            ag, bboxes = AugmentDataset.augment_orientation(img, bbs, _)
            ag = AugmentDataset.draw_boxes(ag, bboxes.bounding_boxes, classes)
            cv2.imshow("orientation: " + str(_), ag)

        for _ in brightnesses:
            ag, bboxes = AugmentDataset.augment_brightness(img, bbs, _)
            ag = AugmentDataset.draw_boxes(ag, bboxes.bounding_boxes, classes)
            cv2.imshow("brightness: " + str(_), ag)

        for _ in scales:
            ag, bboxes = AugmentDataset.augment_scale(img, bbs, _)
            ag = AugmentDataset.draw_boxes(ag, bboxes.bounding_boxes, classes)
            cv2.imshow("scale: " + str(_), ag)

        cv2.waitKey()

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
        # possible augmentations
        # orientations, brightnesses, scales = [90, 180, 270], [0.40, 1.60], [0.5, 0.75]
        orientations, brightnesses, scales = [90, 180, 270], [0.40, 1.60], [0.5, 0.75]
        # augmentation containers
        augmented_images, augmented_boxes = [], []
        # orientation augmentations
        for _ in orientations:
            ag, bbx = AugmentDataset.augment_orientation(image, bbs, _)
            augmented_images.append(ag), augmented_boxes.append(bbx)
        # brightness augmentations
        for _ in brightnesses:
            ag, bbx = AugmentDataset.augment_brightness(image, bbs, _)
            augmented_images.append(ag), augmented_boxes.append(bbx)
        # scale augmentations
        for _ in scales:
            ag, bbx = AugmentDataset.augment_scale(image, bbs, _)
            augmented_images.append(ag), augmented_boxes.append(bbx)
        # window labels for visualization
        window_labels = [
            "angle 90", "angle 180", "angle 270",
            "brightness 0.40", "brightness 1.60",
            "scale 0.5", "scale 0.75"
        ]
        # at this point we have all augmentations and their bounding boxes for the original image
        return augmented_images, augmented_boxes, window_labels

    @staticmethod
    def viz_augmentations(augmented_images, augmented_boxes, classes, window_labels):
        for img, bbx, title in zip(augmented_images, augmented_boxes, window_labels):
            cv2.imshow(title, AugmentDataset.draw_boxes(img, bbx.bounding_boxes, classes))
        cv2.waitKey()


def __main__():
    img_names = glob.glob("../dataset/train/images/*.jpg")
    full_labels = pd.read_csv('../data/train.csv')
    full_labels.head()

    ia.seed(1)

    for img_name in img_names:
        img_name = img_name.rsplit('/', 1)[-1]
        classes = AugmentDataset.get_classes(img_name, full_labels)
        ag_images, ag_boxes, windows_labels = AugmentDataset.create_augmentations(img_name, full_labels)
        for img, bbx, cls in zip(ag_images, ag_boxes, classes):
            cv2.imshow(cls, AugmentDataset.draw_boxes(img, bbx.bounding_boxes, cls))
            cv2.waitKey()

# __main__()
