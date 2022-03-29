from PIL import Image, ImageFile

from torch.utils.data import Dataset
import os

class OfficeHomeDataset(Dataset):
    def __init__(self, dataset_path, domain_list, transform=None) -> None:
        self.path = dataset_path
        self.domain_list = domain_list
        self.transform = transform
        self.domain_dict = {'Art':0,'Clipart':1,'Product':2,'Real_World':3}

        dataset = []
        img_idx = 0
        for domain in self.domain_list:
            domain_idx = self.domain_dict[domain]
            domain_sample_ann = os.path.join(self.path,domain+'.txt')
            with open(domain_sample_ann,'r') as txt:
                lines = txt.readlines()
            for img_info in lines:
                img_path, cls = img_info.split(' ')
                img_path = os.path.join(self.path,img_path)
                cls = int(cls)
                dataset.append((img_path, cls, domain_idx, img_idx))
                img_idx = img_idx + 1
        
        self.dataset = dataset

    def get_num_classes(self):
        _,cls,_,_ = zip(*self.dataset)
        return int(max(cls)) + 1


    def read_image(self,img_path):
        success = False

        if not os.path.exists(img_path):
            raise IOError("{} does not exists".format(img_path))
        
        while not success:
            try:
                img = Image.open(img_path).convert('RGB')
                success = True
            except IOError:
                print("IOError occured while reading {}".format(img_path))

        return img

    def __len__(self):
        return(len(self.dataset))

    def __getitem__(self, index):
        img_path, cls, domain, idx = self.dataset[index]
        if isinstance(img_path,tuple):
            all_imgs = []
            for path in img_path:
                img = self.read_image(path)
                if self.transform is not None:
                    img = self.transform(img)
                all_imgs.append(img)
            all_imgs = tuple(all_imgs)

            return all_imgs + cls + domain + idx

        else:
            img = self.read_image(img_path)
            if self.transform is not None:
                img = self.transform(img)

            return img, cls, domain, idx


