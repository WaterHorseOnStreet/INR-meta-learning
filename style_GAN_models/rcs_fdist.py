import os
import torchvision
from pathlib import Path
import json
from cityscapesscripts.preparation.json2labelImg import json2labelImg
from PIL import Image
import numpy as np
import torch

here = os.getcwd()
cityscapes_path = os.path.join(here,'../../../dataset/cityscapes')
aachen_gt = Path(os.path.join(cityscapes_path,'./gtFine/train/aachen'))
SUFFIX = '.json'
polygons = [path for path in aachen_gt.glob(f'*{SUFFIX}')]

def convert_json_to_label(json_file):
    label_file = json_file.replace('_polygons.json', '_labelTrainIds.png')
    json2labelImg(json_file, label_file, 'trainIds')

    if 'train/' in json_file:
        pil_label = Image.open(label_file)
        label = np.asarray(pil_label)
        sample_class_stats = {}
        for c in range(19):
            n = int(np.sum(label == c))
            if n > 0:
                sample_class_stats[int(c)] = n
        sample_class_stats['file'] = label_file
        return sample_class_stats
    else:
        return None

def get_rcs_class_probs(data_root, temperature):
    with open(os.path.join(data_root, 'sample_class_stats.json'), 'r') as of:
        sample_class_stats = json.load(of)
    overall_class_stats = {}
    for s in sample_class_stats:
        s.pop('file')
        for c, n in s.items():
            c = int(c)
            if c not in overall_class_stats:
                overall_class_stats[c] = n
            else:
                overall_class_stats[c] += n
    overall_class_stats = {
        k: v
        for k, v in sorted(
            overall_class_stats.items(), key=lambda item: item[1])
    }
    freq = torch.tensor(list(overall_class_stats.values()))
    freq = freq / torch.sum(freq)
    freq = 1 - freq
    freq = torch.softmax(freq / temperature, dim=-1)

    return list(overall_class_stats.keys()), freq.numpy()

def save_class_stats(out_dir, sample_class_stats):
    sample_class_stats = [e for e in sample_class_stats if e is not None]
    with open(os.path.join(out_dir, 'sample_class_stats.json'), 'w') as of:
        json.dump(sample_class_stats, of, indent=2)

    sample_class_stats_dict = {}
    for stats in sample_class_stats:
        f = stats.pop('file')
        sample_class_stats_dict[f] = stats
    with open(os.path.join(out_dir, 'sample_class_stats_dict.json'), 'w') as of:
        json.dump(sample_class_stats_dict, of, indent=2)

    samples_with_class = {}
    for file, stats in sample_class_stats_dict.items():
        for c, n in stats.items():
            if c not in samples_with_class:
                samples_with_class[c] = [(file, n)]
            else:
                samples_with_class[c].append((file, n))
    with open(os.path.join(out_dir, 'samples_with_class.json'), 'w') as of:
        json.dump(samples_with_class, of, indent=2)

def get_rcs_class_probs(data_root, temperature):
    with open(os.path.join(data_root, 'sample_class_stats.json'), 'r') as of:
        sample_class_stats = json.load(of)
    overall_class_stats = {}
    for s in sample_class_stats:
        s.pop('file')
        for c, n in s.items():
            c = int(c)
            if c not in overall_class_stats:
                overall_class_stats[c] = n
            else:
                overall_class_stats[c] += n
    overall_class_stats = {
        k: v
        for k, v in sorted(
            overall_class_stats.items(), key=lambda item: item[1])
    }
    freq = torch.tensor(list(overall_class_stats.values()))
    freq = freq / torch.sum(freq)
    freq = 1 - freq
    freq = torch.softmax(freq / temperature, dim=-1)

    return list(overall_class_stats.keys()), freq.numpy()

out_dir = os.path.join(here,'./rcs_output')
os.makedirs(out_dir,exist_ok=True)


# sample_rcs = []
# for json_file in polygons:
#     json_file = str(json_file)
#     sample_rcs.append(convert_json_to_label(json_file))

# save_class_stats(out_dir,sample_rcs)


class_list, freq_list = get_rcs_class_probs(out_dir,0.1)
with open(os.path.join(out_dir,'samples_with_class.json'), 'r') as of:
    samples_with_class_and_n = json.load(of)
samples_with_class_and_n = {
    int(k): v
    for k, v in samples_with_class_and_n.items()
    if int(k) in class_list
}
samples_with_class = {}
rcs_min_pixels = 3000

for c in class_list:
    samples_with_class[c] = []
    for file, pixels in samples_with_class_and_n[c]:
        if pixels > rcs_min_pixels:
            samples_with_class[c].append(file.split('/')[-1])
    assert len(samples_with_class[c]) > 0

ann_path = aachen_gt
sample_path = Path(os.path.join(cityscapes_path,'./leftImg8bit/train/aachen'))
SUFFIX_ANN = '_labelTrainIds.png'
SUFFIX_SAMPLE = '.png'
samples_img = [path for path in sample_path.glob(f'*{SUFFIX_SAMPLE}')]
samples_ann = []
for img in samples_img:
    img = str(img)
    name = img.split('/')[-1]
    name = name.split('.')[0]
    name = name.split('_')
    name = '_'.join([name[1],name[2]])
    samples_ann.append(list(ann_path.glob(f'*{name}*'))[0])
    
img_infos = []
for i,(img,seg_map) in enumerate(zip(samples_img,samples_ann)):
    img = str(img)
    seg_map = str(seg_map)
    img_info = dict(filename=img)
    img_info['ann'] = dict(seg_map=seg_map)
    img_infos.append(img_info)


file_to_idx = {}
for i, dic in enumerate(img_infos):
    file = dic['ann']['seg_map']
    file = file.split('/')[-1]
    file_to_idx[file] = i

c = np.random.choice(class_list, p=freq_list)
print(c)
f1 = np.random.choice(samples_with_class[c])
print(f1)
i1 = file_to_idx[f1]
s1 = samples_img[i1]
print(s1)
# rcs_min_crop_ratio = 0.5
# if rcs_min_crop_ratio > 0:
#     for j in range(10):
#         n_class = torch.sum(s1['gt_semantic_seg'].data == c)
#         # mmcv.print_log(f'{j}: {n_class}', 'mmseg')
#         if n_class > self.rcs_min_pixels * self.rcs_min_crop_ratio:
#             break
#         # Sample a new random crop from source image i1.
#         # Please note, that self.source.__getitem__(idx) applies the
#         # preprocessing pipeline to the loaded image, which includes
#         # RandomCrop, and results in a new crop of the image.
#         s1 = self.source[i1]
# i2 = np.random.choice(range(len(self.target)))
# s2 = self.target[i2]

# return {
#     **s1, 'target_img_metas': s2['img_metas'],
#     'target_img': s2['img']
# }