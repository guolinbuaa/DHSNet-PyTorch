import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image
import os
import os.path
import cv2
import numpy as np


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def find_classes(dir):
    classes = [
            d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))
            ]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir):
    images = []
    dir = os.path.expanduser(dir)
    d = dir

    for root, _, fnames in sorted(os.walk(d)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class ImageFolderS(data.Dataset):
    """
    A simplified version of torchvision.ImageFolder.
    Args:
    root (string):
        Root directory path.
    transform (callable, optional):
        A function that takes in an PIL imageand returns a transformed version.
    loader (callable, optional):
        A function to load an image given its path.
    """

    def __init__(
            self,
            root,
            root_gt,
            transform=None,
            transform_gt=None,
            loader=default_loader,
            loader_gt=default_loader
            ):
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise(
                RuntimeError(
                    "Found 0 images in subfolders of: "+root+"\n"
                    "Supported image extensions are: "+",".join(IMG_EXTENSIONS)
                    )
                )
        gt = make_dataset(root_gt)
        if len(gt) == 0:
            raise(
                RuntimeError(
                    "Found 0 images in subfolders of: "+root_gt+"\n"
                    "Supported image extensions are: "+",".join(IMG_EXTENSIONS)
                    )
                )
        if len(gt) != len(imgs):
            raise(
                RuntimeError(
                    "Size of image data and ground truth doesn't match\n"
                    "Found {0:n} images and \
                            {1:n} depth maps".format(len(imgs), len(gt))
                    )
                )

        if not root.endswith('/'):
            root += '/'
        if not root_gt.endswith('/'):
            root_gt += '/'

        self.root = root
        self.root_gt = root_gt
        self.imgs = imgs

        self.loader = loader
        self.loader_gt = loader_gt
        self.transform = transform
        self.transform_gt = transform_gt

    def __getitem__(self, index=0):
        # load rgbd images
        path = self.imgs[index]
        img = self.transform(self.loader(path))

        # load ground truth
        path = path.split('/')
        file_name = path[len(path)-1]
        file_name = file_name[:len(file_name)-4] + '.png'
        path_gt = self.root_gt + file_name
        gt = self.transform_gt(self.loader_gt(path_gt))
        gt = gt[0, :, :]
        gt = gt.unsqueeze(0)

        return img, gt

    def __len__(self):
        return len(self.imgs)
