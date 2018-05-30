import sys
import argparse
from tqdm import tqdm

import torch
import torch.utils.data
from torchvision import datasets, transforms
from torchvision.utils import save_image

from models import M1


def get_data(batch_size):
    # kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    kwargs = {}
    train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data/', train=True, download=True,
                transform=transforms.ToTensor()),
            batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data/', train=False, transform=transforms.ToTensor()),
            batch_size=batch_size, shuffle=True, **kwargs)
    return train_loader, test_loader

batch_size = 128
train_loader, test_loader = get_data(batch_size)
network_arch = {}
M1_model = M1(network_arch)
M1_model.model.train()

for epoch in range(10):
    train_loss = 0
    for i, (x, y) in enumerate(tqdm(train_loader)):
        output = M1_model.fit(x)

        train_loss += output['loss'].item()
    print('epoch loss:', train_loss / len(train_loader.dataset))
    M1_model.model.eval()
    M1_model.model.gen_samples()
    M1_model.model.train()
