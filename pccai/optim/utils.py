# Copyright (c) 2010-2022, InterDigital
# All rights reserved. 

# See LICENSE under the root folder.


# Utilities related to network optimization

import torch
import torch.optim as optim

# Import all the loss classes to be used
from pccai.optim.cd_canonical import ChamferDistCanonical
from pccai.optim.cd_sparse import ChamferDistSparse


# List the all the loss classes in the following dictionary 
loss_classes = {
    'cd_canonical': ChamferDistCanonical,
    'cd_sparse': ChamferDistSparse
}

def get_loss_class(loss_name):
    loss = loss_classes.get(loss_name.lower(), None)
    assert loss is not None, f'loss class "{loss_name}" not found, valid loss classes are: {list(loss_classes.keys())}'
    return loss


def configure_optimization(pccnet, optim_config):
    """Configure the optimizers and the schedulers for training."""

    # Separate parameters for the main optimizer and the auxiliary optimizer
    parameters = set(
        n
        for n, p in pccnet.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    )
    aux_parameters = set(
        n
        for n, p in pccnet.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    )

    # Make sure we don't have an intersection of parameters
    params_dict = dict(pccnet.named_parameters())
    inter_params = parameters & aux_parameters
    union_params = parameters | aux_parameters
    assert len(inter_params) == 0
    assert len(union_params) - len(params_dict.keys()) == 0

    # We only support the Adam optimizer to make things less complicated
    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(list(parameters))),
        lr=optim_config['main_args']['lr'],
        betas=(optim_config['main_args']['opt_args'][0], optim_config['main_args']['opt_args'][1]),
        weight_decay=optim_config['main_args']['opt_args'][2]
    )
    sche_args = optim_config['main_args']['schedule_args']
    if sche_args[0].lower() == 'exp':
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=sche_args[1])
    elif sche_args[0].lower() == 'step':
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=sche_args[1], gamma=sche_args[2])
    elif sche_args[0].lower() == 'multistep':
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=sche_args[1:-1], gamma=sche_args[-1])
    else: # 'fix' scheme
        scheduler = None

    # For the auxiliary parameters
    if len(aux_parameters) > 0:
        aux_optimizer = optim.Adam(
            (params_dict[n] for n in sorted(list(aux_parameters))),
            lr=optim_config['aux_args']['lr'],
            betas=(optim_config['aux_args']['opt_args'][0], optim_config['aux_args']['opt_args'][1]),
            weight_decay=optim_config['aux_args']['opt_args'][2]
        )
        aux_sche_args = optim_config['aux_args']['schedule_args']
        if aux_sche_args[0].lower() == 'exp':
                aux_scheduler = optim.lr_scheduler.ExponentialLR(aux_optimizer, gamma=aux_sche_args[1])
        elif aux_sche_args[0].lower() == 'step':
                aux_scheduler = optim.lr_scheduler.StepLR(aux_optimizer, step_size=aux_sche_args[1], gamma=aux_sche_args[2])
        elif aux_sche_args[0].lower() == 'multistep':
                aux_scheduler = optim.lr_scheduler.MultiStepLR(aux_optimizer, milestones=aux_sche_args[1:-1], gamma=aux_sche_args[-1])
        else: # 'fix' scheme
            aux_scheduler = None
    else:
        aux_optimizer = aux_scheduler = None

    return optimizer, scheduler, aux_optimizer, aux_scheduler