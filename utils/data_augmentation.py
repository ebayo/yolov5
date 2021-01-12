# Custom data augmentation for traffic signal detection

# imports

import imgaug as ia
import imgaug.augmenters as iaa

def data_augmentation_0 (img, g_blur = 3):
    # simplest data augmentation, try library

    aug = iaa.GaussianBlur(sigma=(0, g_blur))
    return aug(img)