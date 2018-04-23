import torch
import torch.utils.data
import os.path
from PIL import Image
import random
import io
import torchvision.transforms as transforms
import numpy as np

DEV_SIZE = 200

def load_edges2shoes(root):
    """loads in memory numpy data files"""
    def _load(fname):
        arr = np.load(os.path.join(root, fname))
        # convert data from b,0,1,c to b,c,0,1
        arr = np.transpose(arr, (0,3,1,2))
        # scale and shift to [-1,1]
        arr = arr / 127.5 - 1.
        return arr.astype('float32')

    print "loading data numpy files..."
    trainA = _load("trainA.npy")
    trainB = _load("trainB.npy")
    testA  = _load("valA.npy")
    testB  = _load("valB.npy")
    print "done."

    # shuffle train data
    rand_state = random.getstate()
    random.seed(123)
    indx = range(len(trainA))
    random.shuffle(indx)
    trainA = trainA[indx]
    trainB = trainB[indx]
    random.setstate(rand_state)

    devA = trainA[:DEV_SIZE]
    devB = trainB[:DEV_SIZE]

    trainA = trainA[DEV_SIZE:]
    trainB = trainB[DEV_SIZE:]

    return trainA, trainB, devA, devB, testA, testB

class AlignedIterator(object):
    """Iterate multiple ndarrays (e.g. images and labels) IN THE SAME ORDER
    and return tuples of minibatches"""

    def __init__(self, data_A, data_B, **kwargs):
        super(AlignedIterator, self).__init__()

        assert data_A.shape[0] == data_B.shape[0], 'passed data differ in number!'
        self.data_A = data_A
        self.data_B = data_B

        self.num_samples = data_A.shape[0]

        batch_size = kwargs.get('batch_size', 100)
        shuffle = kwargs.get('shuffle', False)

        self.n_batches = self.num_samples / batch_size
        if self.num_samples % batch_size != 0:
            self.n_batches += 1

        self.batch_size = batch_size

        self.shuffle = shuffle

        self.reset()

    def __iter__(self):
        return self

    def reset(self):
        if self.shuffle:
            self.data_indices = np.random.permutation(self.num_samples)
        else:
            self.data_indices = np.arange(self.num_samples)
        self.batch_idx = 0

    def next(self):
        if self.batch_idx == self.n_batches:
            self.reset()
            raise StopIteration

        idx = self.batch_idx * self.batch_size
        chosen_indices = self.data_indices[idx:idx+self.batch_size]
        self.batch_idx += 1

        return {'A': torch.from_numpy(self.data_A[chosen_indices]),
                'B': torch.from_numpy(self.data_B[chosen_indices])}

    def __len__(self):
        return self.num_samples

class UnalignedIterator(object):
    """Iterate multiple ndarrays (e.g. several images) IN DIFFERENT ORDER
    and return tuples of minibatches"""
    def __init__(self, data_A, data_B, **kwargs):
        super(UnalignedIterator, self).__init__()

        assert data_A.shape[0] == data_B.shape[0], 'passed data differ in number!'
        self.data_A = data_A
        self.data_B = data_B

        self.num_samples = data_A.shape[0]

        self.batch_size = kwargs.get('batch_size', 100)
        self.n_batches = self.num_samples / self.batch_size
        if self.num_samples % self.batch_size != 0:
            self.n_batches += 1

        self.reset()

    def __iter__(self):
        return self

    def reset(self):
        self.data_indices = [np.random.permutation(self.num_samples) for _ in range(2)]
        self.batch_idx = 0

    def next(self):
        if self.batch_idx == self.n_batches:
            self.reset()
            raise StopIteration

        idx = self.batch_idx * self.batch_size
        chosen_indices_A = self.data_indices[0][idx:idx+self.batch_size]
        chosen_indices_B = self.data_indices[1][idx:idx+self.batch_size]

        self.batch_idx += 1

        return {'A': torch.from_numpy(self.data_A[chosen_indices_A]),
                'B': torch.from_numpy(self.data_B[chosen_indices_B])}

    def __len__(self):
        return self.num_samples


class Edges2Shoes(object):
    def __init__(self, opt, subset, unaligned, fraction, load_in_mem):
        self.root = opt.dataroot
        self.subset = subset
        self.unaligned = unaligned
        self.fraction = fraction
        self.load_in_mem = load_in_mem
        assert fraction > 0. and fraction <= 1.
        if subset in ['dev', 'train']:
            self.dir_A = os.path.join(self.root, 'trainA')
            self.dir_B = os.path.join(self.root, 'trainB')
        elif subset == 'val':  # test set
            self.dir_A = os.path.join(self.root, 'valA')
            self.dir_B = os.path.join(self.root, 'valB')
        else:
            raise NotImplementedError('subset %s no supported' % subset)

        self.A_paths = sorted(make_dataset(self.dir_A))
        self.B_paths = sorted(make_dataset(self.dir_B))

        # shuffle data
        rand_state = random.getstate()
        random.seed(123)
        indx = range(len(self.A_paths))
        random.shuffle(indx)
        self.A_paths = [self.A_paths[i] for i in indx]
        self.B_paths = [self.B_paths[i] for i in indx]
        random.setstate(rand_state)
        if subset == "dev":
            self.A_paths = self.A_paths[:DEV_SIZE]
            self.B_paths = self.B_paths[:DEV_SIZE]
        elif subset == 'train':
            self.A_paths = self.A_paths[DEV_SIZE:]
            self.B_paths = self.B_paths[DEV_SIZE:]

        # return only fraction of the subset
        subset_size = int(len(self.A_paths) * fraction)
        self.A_paths = self.A_paths[:subset_size]
        self.B_paths = self.B_paths[:subset_size]

        if load_in_mem:
            mem_A_paths = []
            mem_B_paths = []
            for A, B in zip(self.A_paths, self.B_paths):
                with open(A, 'rb') as fa:
                    mem_A_paths.append(io.BytesIO(fa.read()))
                with open(B, 'rb') as fb:
                    mem_B_paths.append(io.BytesIO(fb.read()))
            self.A_paths = mem_A_paths
            self.B_paths = mem_B_paths

        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        self.transform = get_transform(opt)

    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]
        if self.unaligned:
            index_B = random.randint(0, self.B_size - 1)
        else:
            index_B = index % self.A_size
        B_path = self.B_paths[index_B]
        # print('(A, B) = (%d, %d)' % (index_A, index_B))
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')

        A_img = self.transform(A_img)
        B_img = self.transform(B_img)

        return {'A': A_img, 'B': B_img}

    def __len__(self):
        return max(self.A_size, self.B_size)


class DataLoader(object):
    def __init__(self, opt, subset, unaligned, batchSize,
                 shuffle=False, fraction=1., load_in_mem=True, drop_last=False):
        self.opt = opt
        self.dataset = Edges2Shoes(opt, subset, unaligned, fraction, load_in_mem)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=batchSize,
            shuffle=shuffle,
            num_workers=int(opt.nThreads),
            drop_last=drop_last)

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return len(self.dataset)

def get_transform(opt):
    transform_list = [transforms.Scale([64, 64], Image.BICUBIC)]
    transform_list += [transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

def __scale_width(img, target_width):
    ow, oh = img.size
    if (ow == target_width):
        return img
    w = target_width
    h = int(target_width * oh / ow)
    return img.resize((w, h), Image.BICUBIC)

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
