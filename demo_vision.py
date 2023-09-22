import os, sys
import logging
import time
from datetime import datetime
import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F

import networks
import datasets
import utils


parser = argparse.ArgumentParser()

# method and hparams
parser.add_argument('--method', type=str, default='mc_dropout', help='approximate posterior inference method')
parser.add_argument('--hparams', type=str, default='', help='all hparams specific to the method (comma-separated, =-assigned forms)')

# finetuning of pretrained model or training from the scratch (None)
parser.add_argument('--pretrained', type=str, default=None, help='path or url to the pretrained model')

# dataset and backbone network
parser.add_argument('--dataset', type=str, default='mnist', help='dataset name')
parser.add_argument('--backbone', type=str, default='mlp', help='backbone name')
parser.add_argument('--val_heldout', type=float, default=0.1, help='validation set heldout proportion')

# error calibration
parser.add_argument('--ece_num_bins', type=int, default=15, help='number of bins for error calibration')

# other optim hparams
parser.add_argument('--epochs', type=int, default=100, help='number of training epochs')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--lr_head', type=float, default=None, help='learning rate for head')
parser.add_argument('--momentum', type=float, default=0.5, help='momentum')

parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--log_dir', type=str, default='results', help='root folder for saving logs')
parser.add_argument('--test_eval_freq', type=int, default=1, help='do test evaluation (every this epochs)')

args = parser.parse_args()


args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
args.use_cuda = torch.cuda.is_available()

# random seed
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
np.random.seed(args.seed)

# parse hparams
hparams = args.hparams
hparams = hparams.replace('"', '')
hpstr = hparams.replace(',', '_')
opts = hparams.split(',')
hparams = {}
for opt in opts:
    if '=' in opt:
        key, val = opt.split('=')
        hparams[key] = val  # note: string valued
args.hparams = hparams

if args.lr_head is None:
    args.lr_head = args.lr

# set directory for saving results
pretr = 1 if args.pretrained is not None else 0
main_dir = f'{args.dataset}_val_heldout{args.val_heldout}/'
main_dir += f'{args.backbone}/{args.method}_{hpstr}_pretr{pretr}/' 
main_dir += f'ep{args.epochs}_bs{args.batch_size}_lr{args.lr}_lrh{args.lr_head}_mo{args.momentum}/'
main_dir += f'seed{args.seed}_' + datetime.now().strftime('%Y_%m%d_%H%M%S')
args.log_dir = os.path.join(args.log_dir, main_dir)
utils.mkdir(args.log_dir)

# create logger
logging.basicConfig(
    handlers=[
        logging.FileHandler(os.path.join(args.log_dir, 'logs.txt')), 
        logging.StreamHandler()
    ], 
    format='[%(asctime)s,%(msecs)03d %(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger()
cmd = " ".join(sys.argv)
logger.info(f"Command :: {cmd}\n")

# prepare data
logger.info('Preparing data...')
train_loader, val_loader, test_loader, args.ND = datasets.prepare(args)  # ND = train set size

# create backbone (skeleton)
logger.info('Creating an underlying backbone network (skeleton)...')
net = networks.create_backbone(args)
logger.info('Total params in the backbone: %.2fM' % (net.get_nb_parameters() / 1000000.0))
logger.info('Backbone modules:\n%s' % (net.get_module_names()))

# load pretrained backbone (with zero'ed final prediction module)
if args.pretrained is not None:
    logger.info('Load pretrained backbone network...')
    net0 = networks.load_pretrained_backbone(args)  # feat-ext params = pretrained, head = zero
    net = networks.load_pretrained_backbone(args, zero_head=False)  # feat-ext params = pretrained, head = random
else:
    logger.info('No pretrained backbone network provided.')
    net0 = None

if args.method == 'vanilla':

    from methods.vanilla import Runner

    runner = Runner(net, net0, args, logger)
    runner.train(train_loader, val_loader, test_loader)

elif args.method == 'vi':

    from methods.vi import Runner

    runner = Runner(net, net0, args, logger)
    runner.train(train_loader, val_loader, test_loader)

elif args.method == 'mc_dropout':

    from methods.mc_dropout import Runner

    runner = Runner(net, net0, args, logger)
    runner.train(train_loader, val_loader, test_loader)

elif args.method == 'sgld':

    from methods.sgld import Runner

    runner = Runner(net, net0, args, logger)
    runner.train(train_loader, val_loader, test_loader)

elif args.method == 'la':

    from methods.la import Runner

    runner = Runner(net, net0, args, logger)
    runner.train(train_loader, val_loader, test_loader)

else:
    raise NotImplementedError

