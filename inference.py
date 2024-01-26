import numpy as np
from PIL import Image
import argparse, os
from torchvision import transforms
import torch
from torch.utils.data import DataLoader
from datasets import BalancedBatchSampler, DeepFashionDataset
from networks import ResNet50basedNet
import torch.nn as nn
from torch.utils.data import DataLoader
import cv2

def find_most_similar(query, gallery_loader, model, cuda, k=10):
    query_img = query[0].float().cuda() if cuda else query[0].float()
    query_vec = model(torch.unsqueeze(query_img, 0))
    top_k_idx = np.zeros(k)
    top_k_correct = np.zeros(k)
    top_k_dist = np.full((k), 1e5)

    for batch_idx, (data, target_label, _, idx) in enumerate(gallery_loader):

        if not type(data) in (tuple, list):
            data = (data,)
        if cuda:
            data = tuple(d.cuda() for d in data)

        gallery_vecs = model(*data)[idx != query[3]]
        correct = 1 * (target_label[idx != query[3]] == query[1])  # retrieve or not
        idx = idx[idx != query[3]]
        dist = (query_vec - gallery_vecs).pow(2).sum(dim=1)
        tmp_dist = np.concatenate((top_k_dist, dist.detach().cpu().numpy()))
        tmp_idx = np.concatenate((top_k_idx, idx.numpy()))
        tmp_correct = np.concatenate((top_k_correct, correct))
        top_k_idx = tmp_idx[np.argsort(tmp_dist)[:k]]
        top_k_correct = tmp_correct[np.argsort(tmp_dist)[:k]]
        top_k_dist = np.sort(tmp_dist)[:k]

    return top_k_idx, top_k_correct, top_k_dist

def get_topK_acc(gallery_dataset, item_dict, model, cuda, save_file=None, k=100):
    labels = np.asarray([d[1] for d in gallery_dataset], dtype=int)
    item_to_indices = {label: np.where(labels == label)[0] for label in np.arange(len(item_dict))}
    gallery_loader = DataLoader(gallery_dataset, batch_size=32, shuffle=False, num_workers=4)
    acc = np.zeros(k)
    query_cnt = 0
    for item_idx in item_to_indices.keys():
        if len(item_to_indices[item_idx]) <= 1:
            continue
        if query_cnt % 1 == 0:
            print('finding for {}th query...'.format(query_cnt + 1))
        np.random.shuffle(item_to_indices[item_idx])
        query = gallery_dataset[item_to_indices[item_idx][0]]
        top_k_idx, correct, top_k_dist = find_most_similar(query, gallery_loader, model, cuda, k)
        if save_file is not None:
            save_file.write('{} {} {}'.format(query[3], top_k_idx[:10], top_k_dist[:10]))
        acc += [1 * (np.sum(correct[:k_+1]) > 0) for k_ in range(k)]
        query_cnt += 1
        print(acc / query_cnt)

    acc /= query_cnt

    return acc, query_cnt


def get_img(img_path):
    img_dict = {}
    query_img = cv2.imread(img_path)
    img_dict['image'] = torch.from_numpy(query_img).permute(2, 0, 1)
    return [img_dict]

def main(args):
    print(args())
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Image detection')
    parser.add_argument('-db', dest = 'image_data', required=True, help='Directory of Image')
    parser.add_argument('-q', dest = 'img_path', required=True, help='Query Image Path')
    parser.add_argument('-dm', dest = 'detection_model_file', required=True, help='Detection Model Path')
    parser.add_argument('-sm', dest = 'sim_model_file', required=True, help = 'Sim Model Path')
    parser.add_argument('-r', dest='result_dir', required=True, help='Directory to save the result')
    args = parser.parse_args()
    
    main(args)