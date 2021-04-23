import cv2
from albumentations import (
    HorizontalFlip, VerticalFlip, ShiftScaleRotate,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, RandomContrast, RandomGamma,
    RandomBrightness, ShiftScaleRotate, ElasticTransform, Resize, IAAPerspective, RGBShift, Rotate, Normalize
)

def strong_aug(args, p=0.5):
    return Compose([
        OneOf([
            #RandomBrightnessContrast(),
            #RGBShift(),
            #HueSaturationValue(),
            Blur(),
            ], p=p),
        OneOf([
            IAAAdditiveGaussianNoise(),
            IAASharpen(),
            IAAEmboss(),
            ], p=p),
        OneOf([
            HorizontalFlip(),
            #VerticalFlip(),
        ], p=p),
        OneOf([
            IAAPiecewiseAffine(),
            IAAPerspective(),
            ElasticTransform(border_mode=cv2.BORDER_CONSTANT),
            GridDistortion(border_mode=cv2.BORDER_CONSTANT),
            ], p=p),
        OneOf([
            ShiftScaleRotate(border_mode=cv2.BORDER_CONSTANT),
            Rotate(limit=45, border_mode=cv2.BORDER_CONSTANT),
            ], p=p),
        Resize(args.input_img_size_y,args.input_img_size_x,always_apply=True),
        #Normalize(always_apply=True),
        ], p=p,
        additional_targets={'image': 'image',
                            'mask': 'mask'
                            })

def resize(args):
    return Compose([
                    Resize(args.input_img_size_y,args.input_img_size_x,always_apply=True),
                    #Normalize(always_apply=True),
                    ],
                    additional_targets={'image': 'image',
                                        'mask': 'mask'
                                       })

def strong_aug2(args, p=0.5):
    return Compose([
        HorizontalFlip(p=1),
        RandomContrast(p=1),
        RandomGamma(p=1),
        RandomBrightness(p=1),
        ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=1),
        GridDistortion(p=1),
        OpticalDistortion(distort_limit=2, shift_limit=0.5, p=1),
        ShiftScaleRotate(p=1),
        Resize(args.input_img_size,args.input_img_size,always_apply=True, p=1),
        ])

def augment(transform, args, data):
    if transform:
        augmentation = strong_aug(args, p=0.7)
        augmented = augmentation(**data)
    else:
        augmentation = resize(args)
        augmented = augmentation(**data)

    return augmented
