

import torch
import torch.nn as nn

import os, contextlib
from thop import profile

def analyse_model(net, inputs):
    # silence
    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stdout(devnull):
            flops, params = profile(net, (inputs, ))
    return flops, params


def finetune(pack, lr_min, lr_max, T, mute=False):
    logs = []
    epoch = 0

    def iter_hook(curr_iter, total_iter):

        ignored_params = list(map(id, pack.net.fc.parameters()))
        base_params = filter(lambda p: id(p) not in ignored_params, pack.net.parameters())
        pack.optimizer = torch.optim.SGD([{'params': base_params},{'params': pack.net.fc.parameters(), 'lr': 0.005}], lr=0.0005, momentum=0.9, weight_decay=0.0005)
        
    best_acc = 0.0
    for i in range(T):
        info = pack.trainer.train(pack, iter_hook = iter_hook)
        info.update(pack.trainer.test(pack))
        if info['acc@1'] > best_acc:
            best_acc = info['acc@1']
            torch.save(pack.net,'logs/temp.ckp')
        info.update({'LR': pack.optimizer.param_groups[0]['lr']})
        epoch += 1
        if not mute:
            print(i, ': ', info)
        logs.append(info)
    print('*****fine-tuning best acc:', best_acc)
    return logs
