import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import sys

"""
_r = os.getcwd().split('/')
_p = '/'.join(_r[:_r.index('transtailor-vgg')+1])
print('Change dir from %s to %s' % (os.getcwd(), _p))
os.chdir(_p)
sys.path.append(_p)
"""
from config import parse_from_dict

parse_from_dict({
    "base": {
        "task_name": "vgg16_stanford",
        "cuda": True,
        "seed": 1,
        "checkpoint_path": "",
        "epoch": 0,
        "multi_gpus": False,
        "fp16": False
    },
    "model": {
        "name": "stanford.vgg16",
        "num_class": 120,
        "pretrained": True
    },
    "train": {
        "trainer": "normal",
        "max_epoch": 160,
        "optim": "sgd",
        "steplr": [
            [80, 0.1],
            [120, 0.01],
            [160, 0.001]
        ],
        "weight_decay": 5e-4,
        "momentum": 0.9,
        "nesterov": False
    },
    "data": {
        "type": "stanford",
        "shuffle": True,
        "batch_size": 32,
        "test_batch_size": 32,
        "num_workers": 4
    },
    "loss": {
        "criterion": "softmax"
    },
    "gbn": {
        "sparse_lambda": 1e-3,
        "flops_eta": 0,
        "lr_min": 5e-4,
        "lr_max": 1e-2,
        "tock_epoch": 5,
        "T": 10,
        "p": 0.002
    }
})
from config import cfg

import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim

torch.cuda.set_device(0)

from logger import logger
from main import set_seeds, recover_pack, adjust_learning_rate, _step_lr, _sgdr

from utils import dotdict

from prune.universal import Meltable, GatedBatchNorm2d, Conv2dObserver, IterRecoverFramework, FinalLinearObserver
from prune.utils import analyse_model, finetune
import torchvision
import torch.nn.functional as F

import torchvision.models as models


class ModifiedVGG16Model(torch.nn.Module):
    def __init__(self):
        super(ModifiedVGG16Model, self).__init__()

        model = models.vgg16_bn(pretrained=True)
        self.features = model.features

        for param in self.features.parameters():
            param.requires_grad = True

        self.avgpool = nn.AvgPool2d(7, stride=1)
        classifier = nn.Sequential(
            nn.Linear(512, 120),
        )
        self.fc = classifier

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def get_pack():
    set_seeds()
    pack = recover_pack()

    pack.net = torch.load('ckps/vgg16_stanford_baseline.ckp', map_location='cpu' if not cfg.base.cuda else 'cuda')

    torch.save(pack.net, 'logs/temp.ckp')

    GBNs = GatedBatchNorm2d.transform(pack.net)
    for gbn in GBNs:
        gbn.extract_from_bn()

    ignored_params = list(map(id, pack.net.fc.parameters()))
    base_params = filter(lambda p: id(p) not in ignored_params, pack.net.parameters())
    pack.optimizer = torch.optim.SGD([{'params': base_params}, {'params': pack.net.fc.parameters(), 'lr': 0.005}],
                                     lr=0.0005, momentum=0.9, weight_decay=0.0005)

    return pack, GBNs


def clone_model(net):
    model = torch.load('ckps/vgg16_stanford_baseline.ckp', map_location='cpu' if not cfg.base.cuda else 'cuda')
    model = model.cuda()
    gbns = GatedBatchNorm2d.transform(model)
    model.load_state_dict(net.state_dict())
    return model, gbns


def eval_prune(pack):
    cloned, _ = clone_model(pack.net)
    _ = Conv2dObserver.transform(cloned)
    cloned.fc[0] = FinalLinearObserver(cloned.fc[0])
    cloned_pack = dotdict(pack.copy())
    cloned_pack.net = cloned
    Meltable.observe(cloned_pack, 0.001)
    Meltable.melt_all(cloned_pack.net)
    flops, params = analyse_model(cloned_pack.net, torch.randn(1, 3, 224, 224).cuda())
    del cloned
    del cloned_pack

    return flops, params


def prune(pack, GBNs, BASE_FLOPS, BASE_PARAM):
    LOGS = []
    flops_save_points = set([90, 80, 70, 60, 50])
    iter_idx = 0

    pack.tick_trainset = pack.train_loader
    prune_agent = IterRecoverFramework(pack, GBNs, sparse_lambda=cfg.gbn.sparse_lambda, flops_eta=cfg.gbn.flops_eta,
                                       minium_filter=3)
    while True:

        left_filter = prune_agent.total_filters - prune_agent.pruned_filters
        num_to_prune = int(left_filter * cfg.gbn.p)
        info = prune_agent.prune(num_to_prune, tick=True, lr=cfg.gbn.lr_min)
        flops, params = eval_prune(pack)
        info.update({
            'flops': '[%.2f%%] %.3f MFLOPS' % (flops / BASE_FLOPS * 100, flops / 1e6),
            'param': '[%.2f%%] %.3f M' % (params / BASE_PARAM * 100, params / 1e6)
        })
        LOGS.append(info)
        print(
            'Step 1: ter: %d,\t FLOPS: %s,\t Param: %s,\t Left: %d,\t Pruned Ratio: %.2f %%,\t Train Loss: %.4f,\t Test Acc: %.2f' %
            (iter_idx, info['flops'], info['param'], info['left'], info['total_pruned_ratio'] * 100, info['train_loss'],
             info['after_prune_test_acc']))

        iter_idx += 1
        if iter_idx % cfg.gbn.T == 0:
            print('Step 2:')
            prune_agent.tock(lr_min=cfg.gbn.lr_min, lr_max=cfg.gbn.lr_max, tock_epoch=cfg.gbn.tock_epoch)

        flops_ratio = flops / BASE_FLOPS * 100
        for point in [i for i in list(flops_save_points)]:
            if flops_ratio <= point:
                torch.save(pack.net, './logs/vgg16_stanford/%s.ckp' % str(point))
                torch.save(pack.net, 'logs/temp.ckp')
                flops_save_points.remove(point)

        if len(flops_save_points) == 0:
            break


def run():
    pack, GBNs = get_pack()

    cloned, _ = clone_model(pack.net)
    BASE_FLOPS, BASE_PARAM = analyse_model(cloned, torch.randn(1, 3, 224, 224).cuda())
    print('%.3f MFLOPS' % (BASE_FLOPS / 1e6))
    print('%.3f M' % (BASE_PARAM / 1e6))
    del cloned
    # print(pack.net)
    ignored_params = list(map(id, pack.net.fc.parameters()))
    base_params = filter(lambda p: id(p) not in ignored_params, pack.net.parameters())
    pack.optimizer = torch.optim.SGD([{'params': base_params}, {'params': pack.net.fc.parameters(), 'lr': 0.005}],
                                     lr=0.0005, momentum=0.9, weight_decay=0.0005)

    prune(pack, GBNs, BASE_FLOPS, BASE_PARAM)


run()



