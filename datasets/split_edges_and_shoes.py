from pdb import set_trace as st
import os
import numpy as np
import cv2
import argparse

parser = argparse.ArgumentParser('separate image pairs')
parser.add_argument('--fold_AB', dest='fold_AB', help='output directory', type=str, default='edges2shoes')
parser.add_argument('--split', dest='split', help='train / val split', type=str, default='val')
args = parser.parse_args()

for arg in vars(args):
    print('[%s] = ' % arg,  getattr(args, arg))

sp = args.split
img_fold_AB = os.path.join(args.fold_AB, sp)
img_list = os.listdir(img_fold_AB)
num_imgs = len(img_list)
print('split = %s, use %d/%d images' % (sp, num_imgs, len(img_list)))
img_fold_A = os.path.join(args.fold_AB, sp+"A")
img_fold_B = os.path.join(args.fold_AB, sp+"B")
if not os.path.isdir(img_fold_A):
    os.makedirs(img_fold_A)
if not os.path.isdir(img_fold_B):
    os.makedirs(img_fold_B)
print('split = %s, number of images = %d' % (sp, num_imgs))
for n in range(num_imgs):
    name_AB = img_list[n]
    path_AB = os.path.join(img_fold_AB, name_AB)
    assert os.path.isfile(path_AB)
    name_A = name_AB.replace('_AB.', '_A.')
    name_B = name_AB.replace('_AB.', '_B.')
    path_A = os.path.join(img_fold_A, name_A)
    path_B = os.path.join(img_fold_B, name_B)

    im_AB = cv2.imread(path_AB, cv2.IMREAD_COLOR)
    im_A = im_AB[:, :256, :]
    im_B = im_AB[:, 256:, :]
    cv2.imwrite(path_A, im_A)
    cv2.imwrite(path_B, im_B)
