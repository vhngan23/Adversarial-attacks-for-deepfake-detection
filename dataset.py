from os import listdir
from os.path import isfile, join

from torch.utils.data import Dataset
from skimage import io 
import torch
from torchvision import datasets

class SingleDirDataset(Dataset):
    def __init__(self, root_dir, transform = None) :
        super().__init__()
        self.root_dir = root_dir
        self.images_paths = [f for f in listdir(root_dir) if isfile(join(root_dir, f))]
        self.transform = transform 
    
    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, index):
        img_path = join(self.root_dir, self.images_paths[index])
        image = io.imread(img_path)

        if self.transform:
            image = self.transform(image)
        
        return image

def fakeNrealDataset(path, transform = None ):
    return datasets.ImageFolder(path, transform)