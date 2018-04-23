import os.path
from PIL import Image
import numpy as np
import argparse

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images

parser = argparse.ArgumentParser('create numpy data from image folders')
parser.add_argument('--root', help='data directory', type=str, default='./edges2shoes')
args = parser.parse_args()

root = args.root

for subset in ['val', 'train']:
    dir_A = os.path.join(root, '%sA' % subset)
    dir_B = os.path.join(root, '%sB' % subset)
    A_paths = sorted(make_dataset(dir_A))
    B_paths = sorted(make_dataset(dir_B))
    mem_A_np = []
    mem_B_np = []
    for i, (A, B) in enumerate(zip(A_paths, B_paths)):
        mem_A_np.append(np.asarray(Image.open(A).convert('RGB').resize((64,64), Image.BICUBIC)))
        mem_B_np.append(np.asarray(Image.open(B).convert('RGB').resize((64,64), Image.BICUBIC)))
        if i % 1000 == 0:
            print i
    full_A = np.stack(mem_A_np)
    full_B = np.stack(mem_B_np)

    A_size = len(mem_A_np)
    B_size = len(mem_B_np)
    print "%sA size=%d" % (subset, A_size)
    print "%sB size=%d" % (subset, B_size)
    np.save(os.path.join(root, "%sA" % subset), full_A)
    np.save(os.path.join(root, "%sB" % subset), full_B)
