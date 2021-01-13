# Custom data augmentation for traffic signal detection

# imports
import numpy as np

import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage


def data_augmentation_0(img, g_blur=3):
    # simplest data augmentation, try library
    aug = iaa.GaussianBlur(sigma=(0, g_blur))
    return aug(image=img)


def data_augmentation_1(img, labels, pad):
    # data augmentation of padding --> check bounding box deformation
    aug = iaa.CropAndPad(percent=(-pad, pad))
    bboxes = labels2bboxes(labels, img.shape)
    img_aug, bboxes_aug = aug(image=img, bounding_boxes=bboxes)
    # clip and remove boxes outside the image
    bboxes_aug = bboxes_aug.clip_out_of_image()
    bboxes_aug = bboxes_aug.remove_out_of_image()
    labels_aug = bboxes2labels(bboxes_aug)
    return img_aug, labels_aug


def labels2bboxes(labels, img_shape):
    # create BoundingBoxesOnImage object for ImgAug augmenters
    bbs = []
    for lb in labels:
        bbs.append(BoundingBox(lb[1], lb[2], lb[3], lb[4], label=lb[0]))
    return BoundingBoxesOnImage(bbs, shape=img_shape)


def bboxes2labels(bboxes):
    labels = []
    for bb in bboxes:
        lb = [int(bb.label), bb.x1_int, bb.y1_int, bb.x2_int, bb.y2_int]
        labels.append(lb)
    return np.array(labels)
