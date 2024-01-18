import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import BalancedBatchSampler, DeepFashionDataset
from torch.optim import lr_scheduler
import torch.optim as optim
# from torch.utils.tensorboard import SummaryWriter
import torchvision

from networks import EmbeddingNet, ResNet50basedNet, Resnet101basedNet
from losses import OnlineTripletLoss
from utils import AllTripletSelector, HardestNegativeTripletSelector, RandomNegativeTripletSelector, SemihardNegativeTripletSelector # Strategies for selecting triplets within a minibatch
from utils import get_data
from metrics import AverageNonzeroTripletsMetric
from trainer import fit
import numpy as np
import os
import sys
import argparse

model_save_path = "./result/"
img_list, base_path, item_dict = get_data('C:/Users/y2657/sungil/data/image')
model = Resnet101basedNet()
cuda = torch.cuda.is_available()
device = "cuda" if torch.cuda.is_available() else "cpu"
if not os.path.exists(model_save_path):
    os.mkdir(model_save_path)
else:
  model_name = sorted(os.listdir(model_save_path))[-1]
  model.load_state_dict(torch.load(os.path.join(model_save_path,model_name)))
if device == "cuda":
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

kwargs = {'num_workers': 8, 'pin_memory': True} if device == 'cuda' else {}
train_dataset = DeepFashionDataset(img_list['train'])
train_batch_sampler = BalancedBatchSampler(train_dataset.labels, train_dataset.source, n_classes=32, n_samples=4)
online_train_loader = DataLoader(train_dataset, batch_sampler=train_batch_sampler, **kwargs)

test_dataset = DeepFashionDataset(img_list['validation'])
test_batch_sampler = BalancedBatchSampler(test_dataset.labels, test_dataset.source, n_classes=32, n_samples=4)
online_test_loader = DataLoader(test_dataset, batch_sampler=test_batch_sampler, **kwargs)

margin = 1.
loss_fn = OnlineTripletLoss(margin, RandomNegativeTripletSelector(margin))
lr = 1e-3
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
scheduler = lr_scheduler.StepLR(optimizer, 5, gamma=0.1, last_epoch=-1)
n_epochs = 20
log_interval = 200

fit(online_train_loader, online_test_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, model_save_path, metrics=[AverageNonzeroTripletsMetric()])