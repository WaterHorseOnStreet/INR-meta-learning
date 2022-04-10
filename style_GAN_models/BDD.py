import glob
import os

from torch.utils import data
from torchvision.datasets.folder import pil_loader
import tqdm
import json

def load_json(f):
    with open(f, 'r') as fp:
        return json.load(fp)


def save_json(obj, f, *args, **kwargs):
    with open(f, 'w') as fp:
        json.dump(obj, fp, *args, **kwargs)

def split_json(path, dataset_type):
    dirname = os.path.join(os.path.dirname(path), dataset_type)
    os.makedirs(dirname, exist_ok=True)

    labels = load_json(path)
    for label in tqdm(labels):
        name = label['name']
        f = os.path.join(dirname, name.replace('.jpg', '.json'))
        save_json(label, f, indent=4)

class BDDDataset(data.Dataset):

    def __init__(self, root, train=True, transform=None):
        self.root = root
        self.train = train
        self.transform = transform
        self.samples = None

        self.prepare()

    def prepare(self):
        self.samples = []

        if self.train:
            label_paths = glob.glob(
                os.path.join(self.root, 'labels/bdd100k_labels_images_train.json'))
            image_dir = os.path.join(self.root, 'images/10k/train')
        else:
            label_paths = glob.glob(
                os.path.join(self.root, 'labels/bdd100k_labels_images_val.json'))
            image_dir = os.path.join(self.root, 'images/100k/val')

        for label_path in label_paths:
            image_path = os.path.join(
                image_dir,
                os.path.basename(label_path).replace('.json', '.jpg'))

            if os.path.exists(image_path):
                self.samples.append((image_path, label_path))
            else:
                raise FileNotFoundError

    def __getitem__(self, index):
        # TODO: handle label dict

        image_path, label_path = self.samples[index]

        image = pil_loader(image_path)
        label = load_json(label_path)

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.samples)