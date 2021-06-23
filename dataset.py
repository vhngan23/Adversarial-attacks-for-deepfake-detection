from os import listdir
from os.path import isfile, join

from torch.utils.data import Dataset
from skimage import io 
import torch

class SingleDirDataset(Dataset):
    def __init__(self, root_dir,label, transform = None) :
        super().__init__()
        self.root_dir = root_dir
        self.images_paths = [f for f in listdir(root_dir) if isfile(join(root_dir, f))]
        self.transform = transform 
        #0 for fake 1 for real 
        self.label = torch.tensor(int(label)) 
    
    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, index):
        img_path = join(self.root_dir, self.images_paths[index])
        image = io.imread(img_path)
        print(img_path)
        if self.transform:
            image = self.transform(image)
        
        return [image,self.label]

