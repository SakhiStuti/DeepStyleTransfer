import os
import torch.utils.data as data
import numpy as np
from PIL import Image
from torchvision import transforms

class Dataset(data.Dataset):
    def __init__(self, root, tf,size = None):
        super(Dataset, self).__init__()
        self.root = root
        self.paths = os.listdir(self.root)
        self.transform = tf
        if size:
            self.paths = self.paths[:size]
            
    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        img = self.transform(img)
        return (img)

    def __len__(self):
        return len(self.paths)
    

class InfiniteSampler(data.sampler.Sampler):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __iter__(self):
        return iter(self.loop())

    def __len__(self):
        return 2 ** 31

    def loop(self):
        i = 0
        order = np.random.permutation(self.num_samples)
        while True:
            yield order[i]
            i += 1
            if i >= self.num_samples:
                np.random.seed()
                order = np.random.permutation(self.num_samples)
                i = 0

def img_transform():
    transform_list = [
        transforms.Resize(size=(512, 512)),
        transforms.RandomCrop(256),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)