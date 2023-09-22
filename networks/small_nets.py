import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn


class MLP(nn.Module):

    '''
    Vanilla MLP.
    '''

    def __init__(self, input_dim, output_dim, width=1200, depth=2, activation='relu'):

        '''
        Args:
            depth = number of hidden layers (depth=0 means pure linear MLP)
        '''
        
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.width = width
        self.depth = depth
        
        layers = []
        hin = input_dim
        for i in range(depth):
            layers.append(nn.Linear(hin, width))
            if activation == 'tanh':
                nonlin = nn.Tanh()
            else:
                nonlin = nn.ReLU()
            layers.append(nonlin)
            hin = width
        
        self.layers = nn.Sequential(*layers)

        self.classifier = nn.Linear(width, output_dim)

    def forward(self, x):        
        x = x.view(-1, self.input_dim)
        out = self.classifier(self.layers(x))
        return out

    def get_nb_parameters(self):
        return np.sum(p.numel() for p in self.parameters())

    def get_module_names(self):
        out = ''
        for pn, p in self.named_parameters():
            out += f'{pn} -- shape = {list(p.shape)}, #params = {p.numel()}\n'
        return out

