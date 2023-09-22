import numpy as np
import torch
import torch.nn as nn
import torchvision

from networks import small_nets


def create_backbone(args):

    '''
    Backbone for the target task.
    '''

    if args.backbone == 'mlp_mnist':

        #net = small_nets.MLP(input_dim=784, output_dim=10, width=1200, depth=2)
        net = small_nets.MLP(input_dim=784, output_dim=10, width=1000, depth=3)

        net.readout_name = 'classifier'

    elif args.backbone == 'resnet101':

        # randomly initialized network
        net = torchvision.models.resnet101()

        # replace the final readout layer
        net.fc = nn.Linear(2048, args.num_classes)
        
        # randomly initialize readout layer
        for p in net.fc.parameters():
            if len(p.shape) > 1:
                nn.init.kaiming_normal_(p, nonlinearity='relu')
            else:
                nn.init.zeros_(p)
        
        net.readout_name = 'fc'

    elif args.backbone == 'vit_l_32':

        # randomly initialized network
        net = torchvision.models.vit_l_32()

        # replace the final readout layer
        net.heads.head = nn.Linear(1024, args.num_classes)
        
        # randomly initialize readout layer
        for p in net.heads.head.parameters():
            if len(p.shape) > 1:
                nn.init.kaiming_normal_(p, nonlinearity='relu')
            else:
                nn.init.zeros_(p)

        net.readout_name = 'heads.head'

    else:

        raise NotImplementedError

    net.get_nb_parameters = lambda: np.sum(p.numel() for p in net.parameters())
    net.get_module_names = lambda: ''.join([f'{pn} -- shape = {list(p.shape)}, #params = {p.numel()}\n' for pn, p in net.named_parameters()])

    return net


def load_pretrained_backbone(args, zero_head=True):

    '''
    Load pretrained backbone from url or path specified in "args.pretrained".
    
    Args:
        zero_head = if True, we zero out the final prediction layer (aka head or readout)
                    if False, head is random initialised
    '''

    if args.backbone == 'resnet101':

        # create a pretrained network
        if args.pretrained == 'IMAGENET1K_V1':
            weights = torchvision.models.ResNet101_Weights.IMAGENET1K_V1
        elif args.pretrained == 'IMAGENET1K_V2':
            weights = torchvision.models.ResNet101_Weights.IMAGENET1K_V2
        else:
            raise NotImplementedError
        net = torchvision.models.resnet101(weights=weights)

        # replace the final readout layer
        net.fc = nn.Linear(2048, args.num_classes)
        
        if zero_head:  # zero-initialize readout layer
            for p in net.fc.parameters():
                nn.init.zeros_(p)
        else:  # randomly initialize readout layer
            for p in net.fc.parameters():
                if len(p.shape) > 1:
                    nn.init.kaiming_normal_(p, nonlinearity='relu')
                else:
                    nn.init.zeros_(p)

        net.readout_name = 'fc'

    elif args.backbone == 'vit_l_32':

        # create a pretrained network
        if args.pretrained == 'IMAGENET1K_V1':
            weights = torchvision.models.ViT_L_32_Weights.IMAGENET1K_V1
        else:
            raise NotImplementedError
        net = torchvision.models.vit_l_32(weights=weights)

        # replace the final readout layer
        net.heads.head = nn.Linear(1024, args.num_classes)
        
        if zero_head:  # zero-initialize readout layer
            for p in net.heads.head.parameters():
                nn.init.zeros_(p)
        else:  # randomly initialize readout layer
            for p in net.heads.head.parameters():
                if len(p.shape) > 1:
                    nn.init.kaiming_normal_(p, nonlinearity='relu')
                else:
                    nn.init.zeros_(p)

        net.readout_name = 'heads.head'

    else:

        raise NotImplementedError
    
    return net
