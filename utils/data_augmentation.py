# Custom data augmentation for traffic signal detection

import numpy as np

import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

FILL_COLOR = 114

keys_data_aug = ['scalem', 'scaleM', 'trans', 'rot', 'shear', 'pers', 'sigma',
                 'mot_km', 'mot_kM', 'mot_an', 'mot_dm', 'mot_dM',
                 'jpegm', 'jpegM', 'con_alpham', 'con_alphaM', 'con_chan',
                 'col_mulm', 'col_mulM', 'col_chan', 'col_addm', 'col_addM',
                 'co_num', 'co_sm', 'co_sM']


class DataAugmenter:
    def __init__(self, da):
        assert check_data_aug(da), 'Parameters for custom data augmentation missing. Should have: {}'.format(
                                    keys_data_aug)

        aug_geometric = [iaa.Affine(scale=(da['scalem'], da['scaleM']),
                                    translate_percent={'x': (-da['trans'], da['trans']),
                                                       'y': (-da['trans'], da['trans'])},
                                    rotate=(-da['rot'], da['rot']),
                                    shear=(-da['shear'], da['shear']),
                                    cval=FILL_COLOR),
                         iaa.PerspectiveTransform(scale=(0, da['pers']),
                                                  cval=FILL_COLOR, keep_size=True)]

        aug_camera = [iaa.GaussianBlur(sigma=(0, da['sigma'])),
                      iaa.MotionBlur(k=(da['mot_km'], da['mot_kM']),
                                     angle=(-da['mot_an'], da['mot_an']),
                                     direction=(da['mot_dm'], da['mot_dM'])),
                      iaa.JpegCompression(compression=(da['jpegm'], da['jpegM'])),
                      iaa.LinearContrast(alpha=(da['con_alpham'], da['con_alphaM']),
                                         per_channel=da['con_chan']),
                      iaa.MultiplyHueAndSaturation(mul=(da['col_mulm'], da['col_mulM']),
                                                   per_channel=da['col_chan'],
                                                   from_colorspace='BGR'),
                      iaa.AddToHueAndSaturation((da['col_addm'], da['col_addM']),
                                                per_channel=da['col_chan'])
                      ]

        cutout = iaa.Cutout(nb_iterations=(0, da['co_num']),
                            size=(da['co_sm'], da['co_sM']),
                            squared=False,
                            cval=FILL_COLOR)

        # Create a mix of all others
        self.augmenter = iaa.Sequential([iaa.SomeOf((0, 1), aug_geometric),  # none or 1
                                         iaa.SomeOf((0, len(aug_camera) - 2), aug_camera),  # from none to all-2
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
    return np.array(labels, dtype=np.float32)


def check_data_aug(da):
    for k in keys_data_aug:
        if k not in da:
            return False
    return True
