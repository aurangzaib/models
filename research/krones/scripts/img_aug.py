from random import uniform

import cv2
import imgaug as ia
import numpy as np
from imgaug import augmenters as iaa


class AugmentDataset:
    # possible augmentations
    orientations = [90, 180, 270]
    scales = np.arange(0.60, 2.0, 0.20)
    brightnesses = np.arange(0.20, 1.80, 0.20)
    window_labels = [
        "angle 90", "angle 180", "angle 270",
        "brightness 0.30", "brightness 0.60", "brightness 0.90",
        "brightness 1.20", "brightness 1.50", "brightness 1.80",
        "scale 0.4", "scale 0.6", "scale 0.8",
    ]

    # augmentation hyper-parameters
    orientations_range = np.arange(0, 365, 5)
    brightnesses_range = (0.20, 1.80)
    scales_range = (0.60, 1.0)
    labels_range = ["orientation-" + str(_) for _ in orientations_range]

    # image increment variable
    multiplier = 1

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
    def get_random_scale():
        return float("{:0.5f}".format(uniform(AugmentDataset.scales_range[0],
                                              AugmentDataset.scales_range[1])))

    @staticmethod
    def get_random_brightness():
        return float("{:0.5f}".format(uniform(AugmentDataset.brightnesses_range[0],
                                              AugmentDataset.brightnesses_range[1])))

    @staticmethod
    def draw_boxes(img, bboxes, classes):
        _ = np.copy(img)
        for bbox, cls in zip(bboxes, classes):
            color = (0, 0, 255) if cls == 'fallen' else (0, 255, 0)
            cv2.rectangle(_, (bbox.x1, bbox.y1), (bbox.x2, bbox.y2), color, 3)
        return _

    @staticmethod
    def augment_orientation(image, bounding_boxes, angle=90):
        """
        perform orientation augmentation
        """
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
        """
        perform orientation augmentation recursively
        recursively means for each orientation augmentation, perform scale and brightness augmentation too
        """
        # orientation augmentations
        for _ in AugmentDataset.orientations:
            # first stage augmentation
            ag, bbx = AugmentDataset.augment_orientation(image, bbs, _)
            augmented_images.append(ag), augmented_boxes.append(bbx)
            # second stage augmentation
            for __ in AugmentDataset.brightnesses:
                ag2, bbx2 = AugmentDataset.augment_brightness(ag, bbx, __)
                augmented_images.append(ag2), augmented_boxes.append(bbx2)
            for __ in AugmentDataset.scales:
                ag2, bbx2 = AugmentDataset.augment_brightness(ag, bbx, __)
                augmented_images.append(ag2), augmented_boxes.append(bbx2)
        # return appended results
        return augmented_images, augmented_boxes

    @staticmethod
    def augment_brightness(image, bounding_boxes, brightness=0.50):
        """
        perform brightness augmentation
        """
        # rotation values
        seq = iaa.Sequential([iaa.Multiply(brightness)])
        seq_det = seq.to_deterministic()
        # apply rotation on the image
        img_aug = seq_det.augment_images([image])[0]
        # return image and bounding boxes
        return img_aug, bounding_boxes

    @staticmethod
    def recursive_augment_brightness(image, bbs, augmented_images, augmented_boxes):
        """
        perform brightness recursively
        recursively means for each brightness augmentation, perform scale and orientation augmentation too
        """
        # first stage augmentation
        for _ in AugmentDataset.brightnesses:
            ag, bbx = AugmentDataset.augment_brightness(image, bbs, _)
            augmented_images.append(ag), augmented_boxes.append(bbx)
            # second stage augmentation
            for __ in AugmentDataset.orientations:
                ag2, bbx2 = AugmentDataset.augment_orientation(ag, bbx, __)
                augmented_images.append(ag2), augmented_boxes.append(bbx2)
            for __ in AugmentDataset.scales:
                ag2, bbx2 = AugmentDataset.augment_scale(ag, bbx, __)
                augmented_images.append(ag2), augmented_boxes.append(bbx2)
        # return image and bounding boxes
        return augmented_images, augmented_boxes

    @staticmethod
    def augment_scale(image, bounding_boxes, scale=0.50):
        """
        perform scale augmentation
        """
        # scaling values
        seq = iaa.Sequential([iaa.Affine(scale={"x": scale, "y": scale})])
        seq_det = seq.to_deterministic()
        # apply scale on the image
        img_aug = seq_det.augment_images([image])[0]
        bbs_aug = seq_det.augment_bounding_boxes([bounding_boxes])[0].remove_out_of_image().cut_out_of_image()
        # return image and bounding boxes
        return img_aug, bbs_aug

    @staticmethod
    def recursive_augment_scale(image, bbs, augmented_images, augmented_boxes):
        """
        perform scale augmentation recursively
        recursively means for each scale augmentation, perform brightness and orientation augmentation too
        """
        # first stage augmentation
        for _ in AugmentDataset.scales:
            ag, bbx = AugmentDataset.augment_scale(image, bbs, _)
            augmented_images.append(ag), augmented_boxes.append(bbx)
            # second stage augmentation
            for __ in AugmentDataset.orientations:
                ag2, bbx2 = AugmentDataset.augment_orientation(ag, bbx, __)
                augmented_images.append(ag2), augmented_boxes.append(bbx2)
            for __ in AugmentDataset.brightnesses:
                ag2, bbx2 = AugmentDataset.augment_brightness(ag, bbx, __)
                augmented_images.append(ag2), augmented_boxes.append(bbx2)
        # return image and bounding boxes
        return augmented_images, augmented_boxes

    @staticmethod
    def create_recursive_augmentations(filename, image_dir, full_labels):
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
        return augmented_images, augmented_boxes, AugmentDataset.window_labels

    @staticmethod
    def create_augmentations(filename, image_dir, full_labels):
        """
        perform angles from 15 to 345 orientations augmentation
        this is to cover all directions of fallen bottles
        this is not recursive
        intended to be used to generate TFRecord
        """
        f_name = "{}/{}".format(image_dir, filename)
        image = cv2.imread(f_name)
        print(filename)
        # bounding box for the original image
        bbs = ia.BoundingBoxesOnImage(AugmentDataset.get_boxes(filename, full_labels),
                                      shape=image.shape)
        # augmentation containers
        augmented_images, augmented_boxes = [], []
        # orientation augmentation
        for _ in AugmentDataset.orientations_range:
            # random scale augmentation
            scale = AugmentDataset.get_random_scale()
            ag2, bbs2 = AugmentDataset.augment_scale(image, bbs, scale)
            # random brightness augmentation
            brightness = AugmentDataset.get_random_brightness()
            ag2, bbs2 = AugmentDataset.augment_brightness(ag2, bbs2, brightness)
            # orientation augmentation
            ag2, bbs2 = AugmentDataset.augment_orientation(ag2, bbs2, _)
            # append images and bounding boxes
            augmented_images.append(ag2), augmented_boxes.append(bbs2)
        return augmented_images, augmented_boxes, AugmentDataset.labels_range

    @staticmethod
    def save_augmentations(filenames, full_labels, image_dir):
        """
        perform angles from 15 to 345 orientations augmentation
        this is to cover all directions of fallen bottles
        this is not recursive
        intended to be used to save all orientations for debugging
        """
        # directory to save images
        img_save_dir = "{}_{}".format(image_dir, "annotated/")
        for filename in filenames:
            image = cv2.imread(filename)
            image_name = filename.rsplit('/', 1)[-1]
            # skip temporary debugging files
            if ("tmp.jpg" in image_name) is False:
                print("\n-> {}".format(image_name))
                # bounding box for the original image
                bbs = ia.BoundingBoxesOnImage(AugmentDataset.get_boxes(image_name, full_labels), shape=image.shape)
                image_name = image_name.rsplit('.', 1)[0]
                # classes
                classes = AugmentDataset.get_classes(filename.rsplit('/', 1)[-1], full_labels)
                for _ in AugmentDataset.orientations_range:
                    # orientation augmentation
                    ag2, bbs2 = AugmentDataset.augment_orientation(image, bbs, _)
                    # random scale augmentation
                    scale = AugmentDataset.get_random_scale()
                    print(scale),
                    ag2, bbs2 = AugmentDataset.augment_scale(ag2, bbs2, scale)
                    # random brightness augmentation
                    brightness = AugmentDataset.get_random_brightness()
                    ag2, bbs2 = AugmentDataset.augment_brightness(ag2, bbs2, brightness)
                    # draw boxes for debugging
                    ag2 = AugmentDataset.draw_boxes(ag2, bbs2.bounding_boxes, classes)
                    # new file name
                    aug_image_name = "{}{}-{}.{}".format(img_save_dir, image_name, str(_), "jpg")
                    # save augmentation
                    cv2.imwrite(aug_image_name, ag2)

    @staticmethod
    def assert_augmentations(filenames, full_labels):
        """
        assert the scale augmentation
        """
        for index in range(0, 10):
            for filename in filenames:
                image = cv2.imread(filename)
                image_name = filename.rsplit('/', 1)[-1]
                # skip temporary debugging files
                if ("tmp.jpg" in image_name) is True:
                    continue
                print("-> {}".format(image_name))
                # bounding box for the original image
                bbs = ia.BoundingBoxesOnImage(AugmentDataset.get_boxes(image_name, full_labels), shape=image.shape)
                for _ in AugmentDataset.orientations_range:
                    # random scale augmentation
                    scale = AugmentDataset.get_random_scale()
                    AugmentDataset.augment_scale(image, bbs, scale)

    @staticmethod
    def viz_augmentations(augmented_images, augmented_boxes, classes, titles):
        for img, bbx, ttl in zip(augmented_images, augmented_boxes, titles):
            title = "aug {}".format(ttl)
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
            image_name = filename.rsplit('/', 1)[-1]
            classes = AugmentDataset.get_classes(image_name, full_labels)
            # show image with boxes
            cv2.imshow("image", AugmentDataset.draw_boxes(image, bbs.bounding_boxes, classes))
            cv2.waitKey()

    @staticmethod
    def save_annotations(augmented_images, augmented_boxes, filename, classes, img_dir):
        count = len(augmented_images)
        save_dir = "{}_annotated".format(img_dir)
        # remove extension
        filename = filename.rsplit('.', 1)[0]
        for index in range(count):
            save_dir = "{}/{}-{}.{}".format(save_dir, filename, str(AugmentDataset.multiplier * index), "jpg")
            _img_ = AugmentDataset.draw_boxes(augmented_images[index], augmented_boxes[index].bounding_boxes, classes)
            cv2.imwrite(save_dir, _img_)
        AugmentDataset.multiplier += 1
