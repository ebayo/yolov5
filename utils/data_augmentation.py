# Custom data augmentation for traffic signal detection

# imports
import numpy as np

import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

# from datasets.py --> letterbox
# use for data augmentations that change the image shape and keep the same color as the padding from rectangle to square
FILL_COLOR = 114


class DataAugmenter:
    FILL_COLOR = 114

    def __init__(self, da):
        #TODO: check the existence of the parameters??

        # Create aug_list with parameters in daug
        aug_geometric = [iaa.Affine(scale={'x': (da['scalem'], da['scaleM']), 'y': (da['scalem'], da['scaleM'])},
                                    translate_percent={'x': (-da['trans'], da['trans']),
                                                       'y': (-da['trans'], da['trans'])},
                                    rotate=(-da['rot'], da['rot']),
                                    shear=(-da['shear'], da['shear']),
                                    cval=FILL_COLOR),
                         iaa.PerspectiveTransform(scale=(0, da['pers']),
                                                  cval=FILL_COLOR)]

        aug_camera = [iaa.GaussianBlur(sigma=(0, da['sigma'])),
                      iaa.MotionBlur(k=(da['mot_km'], da['mot_kM']),
                                     angle=(-da['mot_an'], da['mot_an']),
                                     direction=(da['mot_dm'], da['mot_dM'])),
                      iaa.JpegCompression(compression=(da['jpegm'], da['jpegM'])),
                      iaa.LinearContrast(alpha=(da['con_alpham'], da['con_alphaM']),
                                         per_channel=da['con_chan']),
                      iaa.MultiplyHueAndSaturation(mul=(da['col_mulm'], da['col_mulM']),
                                                   per_channel=da['col_chan'],
                                                   from_colorspace='BGR')
                      ]

        # TODO: should we check for occlusions??
        cutout = iaa.Cutout(nb_iterations=(0, da['co_num']),
                            size=(da['co_sm'], da['co_sM']),
                            squared=False,
                            cval=FILL_COLOR)

        # Create a mix of all others
        self.augmenter = iaa.Sequential([iaa.SomeOf((0, 1), aug_geometric),  # none or 1
                                         iaa.SomeOf((0, None), aug_camera),  # from none to all
                                         cutout], random_order=True)  # mix the apply order

    def augment(self, img, labels):
        # create bboxes from labels
        bboxes = labels2bboxes(labels, img.shape)

        # Apply augmenters
        img_aug, bboxes_aug = self.augmenter(image=img, bounding_boxes=bboxes)

        # Clip bounding boxes --> inside aug (iaa.Clip....
        bboxes_aug = bboxes_aug.clip_out_of_image()
        bboxes_aug = bboxes_aug.remove_out_of_image()

        # Create labels from bboxes
        labels_aug = bboxes2labels(bboxes_aug)

        return img_aug, labels_aug


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


def data_augmentation_2(img, dir0, dir1):
    # data augmentation of MotionBlur, see the effect of the direction parameter (we will always go forward)
    aug = iaa.MotionBlur(k=15, direction=(dir0, dir1));
    img_aug = aug(image=img)
    return img_aug


def data_augmentation(img, labels, da):
    # Create aug_list with parameters in daug
    aug_POV = [iaa.Affine(scale={'x': (da['scalem'], da['scaleM']), 'y': (da['scalem'], da['scaleM'])},
                          translate_percent={'x': (-da['trans'], da['trans']), 'y': (-da['trans'], da['trans'])},
                          rotate=(-da['rot'], da['rot']),
                          shear=(-da['shear'], da['shear']),
                          cval=FILL_COLOR),
               iaa.PerspectiveTransform(scale=(0, da['pers']),
                                        cval=FILL_COLOR)]

    aug_camera = [iaa.GaussianBlur(sigma=(0, da['sigma'])),
                  iaa.MotionBlur(k=(da['mot_km'], da['mot_kM']),
                                 angle=(-da['mot_an'], da['mot_an']),
                                 direction=(da['mot_dm'], da['mot_dM'])),
                  iaa.JpegCompression(compression=(da['jpegm'], da['jpegM']))]

    # Create either Sequential or SomeOf
    aug = iaa.Sequential([iaa.SomeOf((0, 1), aug_POV),
                          iaa.SomeOf((0, None), aug_camera),
                          iaa.Cutout(nb_iterations=(0, da['co_num']),
                                     size=(da['co_sm'], da['co_sM']),
                                     squared=False,
                                     cval=FILL_COLOR)], random_order=True)

    # create bboxes from labels
    bboxes = labels2bboxes(labels, img.shape)

    # Apply augmenters
    img_aug, bboxes_aug = aug(image=img, bounding_boxes=bboxes)

    # Clip bounding boxes --> inside aug (iaa.Clip....
    bboxes_aug = bboxes_aug.clip_out_of_image()
    bboxes_aug = bboxes_aug.remove_out_of_image()

    # Create labels from bboxes
    labels_aug = bboxes2labels(bboxes_aug)

    return img_aug, labels_aug

    # TODO:
    # - random parameters  --> by default (in general) when tuple --> hyperparameters are their min and max
    # - random order --> in iaa.Sequential, random_order=True
    # - random number of augmenters --> use iaa.SomeOf(...)
    #       - n=(min, None) --> from min to all augmenters
    #       - can use random_order=True
    #       - iaa.ClipCBAsToImagePlanes() --> appl


# Auxiliary functions


def labels2bboxes(labels, img_shape):
    # create BoundingBoxesOnImage object for ImgAug augmenters
    bbs = []
    for lb in labels:
        bbs.append(BoundingBox(lb[1], lb[2], lb[3], lb[4], label=lb[0]))
    return BoundingBoxesOnImage(bbs, shape=img_shape)


# TODO: change directly to YOLO format --> remove the change from datasets.py ??
# https://imgaug.readthedocs.io/en/latest/source/api_augmentables_bbs.html
def bboxes2labels(bboxes):
    labels = []
    for bb in bboxes:
        lb = [int(bb.label), bb.x1_int, bb.y1_int, bb.x2_int, bb.y2_int]
        labels.append(lb)
    return np.array(labels, dtype=np.float32)

# def check_data_aug(hyp):
