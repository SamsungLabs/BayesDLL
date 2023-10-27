import os
import copy
import time
from tqdm import tqdm
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import bayesdll.calibration as calibration


class Runner:

    def __init__(self, net, net0, args, logger):

        '''
        Args:
            net = randomly initialized backbone (on cpu)
            net0 = pretrained backbone or None (on cpu)
        '''
        
        self.args = args
        self.logger = logger
        
        # prepare prior backbone (either 0 or pretrained)
        if args.pretrained is None:  # if no pretrained backbone provided, zero out all params
            self.net0 = copy.deepcopy(net)
            with torch.no_grad():
                for pn, p in self.net0.named_parameters():
                    p.copy_(torch.zeros_like(p))
        else:  # pretrained backbone available
            self.net0 = net0
        self.net0 = self.net0.to(args.device)

        # workhorse network (its params will be filled in on the fly)
        self.net = net.to(args.device)

        # create variational inference model (nn.Module)
        hparams = args.hparams
        self.model = Model(
            self.net if args.pretrained is None else self.net0,  # used as init for m (random init if not pretrained, or pretrained otherwise)
            ND=args.ND, prior_sig=float(hparams['prior_sig']), bias=str(hparams['bias'])
        ).to(args.device)

        # create optimizer
        # self.optimizer = torch.optim.SGD(
        #     self.model.parameters(), 
        #     lr = args.lr, momentum = args.momentum, weight_decay = 0
        # )
        self.optimizer = torch.optim.SGD(
            [{'params': [p for pn, p in self.model.m.named_parameters() if self.net.readout_name not in pn], 'lr': args.lr},
             {'params': [p for pn, p in self.model.m.named_parameters() if self.net.readout_name in pn], 'lr': args.lr_head},
             {'params': [p for pn, p in self.model.s_.named_parameters() if self.net.readout_name not in pn], 'lr': args.lr},
             {'params': [p for pn, p in self.model.s_.named_parameters() if self.net.readout_name in pn], 'lr': args.lr_head}], 
            momentum = args.momentum, weight_decay = 0
        )

        # TODO: create scheduler?

        self.criterion = torch.nn.CrossEntropyLoss()

        self.kld = float(hparams['kld'])  # kl discount factor (factor in data augmentation)
        self.nst = int(hparams['nst'])  # number of samples at test time


    def train(self, train_loader, val_loader, test_loader):

        args = self.args
        logger = self.logger

        logger.info('Start training...')

        epoch = 0

        losses_train = np.zeros(args.epochs)
        losses_nll_train = np.zeros(args.epochs)
        losses_kl_train = np.zeros(args.epochs)
        errors_train = np.zeros(args.epochs)
        if val_loader is not None:
            losses_val = np.zeros(args.epochs)
            errors_val = np.zeros(args.epochs)
        losses_test = np.zeros(args.epochs)
        errors_test = np.zeros(args.epochs)

        best_loss = np.inf

        svec = torch.nn.utils.parameters_to_vector(self.model._retrieve_s().parameters())
        logger.info('[Epoch -1/%d] s (= std(q)) mean = %.8f, min = %.8f, max = %.8f' % 
            (args.epochs, svec.mean().item(), svec.min().item(), svec.max().item()))

        tic0 = time.time()
        for ep in range(epoch, args.epochs):

            # TODO: Do some optimizer lr scheduling...

            tic = time.time()
            losses_train[ep], errors_train[ep], losses_nll_train[ep], losses_kl_train[ep] = \
                self.train_one_epoch(train_loader)
            toc = time.time()

            prn_str = '[Epoch %d/%d] Training summary: ' % (ep, args.epochs)
            prn_str += 'loss = %.4f (nll = %.4f, kl = %.4f), prediction error = %.4f ' % (
                losses_train[ep], losses_nll_train[ep], losses_kl_train[ep], errors_train[ep])
            prn_str += '(time: %.4f seconds)' % (toc-tic,)
            logger.info(prn_str)

            svec = torch.nn.utils.parameters_to_vector(self.model._retrieve_s().parameters())
            logger.info('[Epoch %d/%d] s (= std(q)) mean = %.8f, min = %.8f, max = %.8f' % 
                (ep, args.epochs, svec.mean().item(), svec.min().item(), svec.max().item()))

            # test evaluation
            if ep % args.test_eval_freq == 0:

                # test on validation set (if available)
                if val_loader is not None:
                    tic = time.time()
                    losses_val[ep], errors_val[ep], targets_val, logits_val, logits_all_val = \
                        self.evaluate(val_loader)
                    toc = time.time()
                    prn_str = f'(Epoch {ep}) Validation summary: '
                    prn_str += 'loss = %.4f, prediction error = %.4f ' % (losses_val[ep], errors_val[ep])
                    prn_str += '(time: %.4f seconds)' % (toc-tic,)
                    logger.info(prn_str)

                # test on test set
                tic = time.time()
                losses_test[ep], errors_test[ep], targets_test, logits_test, logits_all_test = \
                    self.evaluate(test_loader)
                toc = time.time()
                prn_str = f'(Epoch {ep}) Test summary: '
                prn_str += 'loss = %.4f, prediction error = %.4f ' % (losses_test[ep], errors_test[ep])
                prn_str += '(time: %.4f seconds)' % (toc-tic,)
                logger.info(prn_str)

                loss_now = losses_val[ep] if val_loader is not None else losses_test[ep]
                if loss_now < best_loss:

                    best_loss = loss_now
                    logger.info('Best evaluation loss so far! @epoch %s: loss = %s' % (ep, loss_now))
                    
                    # save logits and labels
                    if val_loader is not None:
                        fname = self.save_logits(
                            targets_val, logits_val, logits_all_val, suffix='val'
                        )  # save prediction logits on validation set
                        logger.info(f'Logits on val set saved at {fname}')
                    fname = self.save_logits(
                        targets_test, logits_test, logits_all_test, suffix='test'
                    )  # save prediction logits on test set
                    logger.info(f'Logits on test set saved at {fname}')
                    
                    # save checkpoint
                    fname = self.save_ckpt(ep)  # save checkpoint
                    logger.info(f'Checkpoint saved at {fname}')

                    # perform error calibration (ECE, MCE, reliability plot, etc.)
                    ece_no_ts, mce_no_ts, nll_no_ts = calibration.analyze(
                        targets_test, logits_test, num_bins = args.ece_num_bins, 
                        plot_save_path = os.path.join(args.log_dir, 'reliability_T1.png'), 
                        temperature = 1
                    )  # calibration with default temperature (T=1)
                    logger.info(
                        '[Calibration - Default T=1] ECE = %.4f, MCE = %.4f, NLL = %.4f' %
                        (ece_no_ts, mce_no_ts, nll_no_ts)
                    )
                    if val_loader is not None:  # perform temperature scaling
                        Topt, success = calibration.find_optimal_temperature(
                            targets_val, logits_val, 
                            plot_save_path = os.path.join(args.log_dir, 'temp_scale_optim_curve.png'), 
                        )  # find optimal temperature on val set
                        if success:
                            ece_ts, mce_ts, nll_ts = calibration.analyze(
                                targets_test, logits_test, num_bins = args.ece_num_bins, 
                                plot_save_path = os.path.join(args.log_dir, 'reliability_Topt.png'), 
                                temperature = Topt
                            )  # calibration with optimal temperature
                            logger.info(
                                '[Calibration - Temp-scaled Topt=%.4f] ECE = %.4f, MCE = %.4f, NLL = %.4f' %
                                (Topt, ece_ts, mce_ts, nll_ts)
                            )
                        else:
                            logger.info('!! Temperature scalaing optimization failed !!')

        toc0 = time.time()
        logger.info('Training done! Total time = %f (average per epoch = %f) seconds' % 
            (toc0-tic0, (toc0-tic0)/args.epochs))


    def train_one_epoch(self, train_loader):

        args = self.args

        self.model.train()
        self.net.train()
        
        loss, loss_nll, loss_kl, error, nb_samples = 0, 0, 0, 0, 0
        with tqdm(train_loader, unit="batch") as tepoch:
            for x, y in tepoch:

                x, y = x.to(args.device), y.to(args.device)

                # evaluate minibatch -ELBO loss for a given a batch
                loss_, out, loss_nll_, loss_kl_ = self.model(
                    x, y, self.net, self.net0, self.criterion, self.kld, eval_grad=1
                )

                self.optimizer.step()  # update self.model.{m,s_}

                # prediction on training
                pred = out.data.max(dim=1)[1]
                err = pred.ne(y.data).sum()

                loss += loss_ * len(y)
                loss_nll += loss_nll_ * len(y)
                loss_kl += loss_kl_ * len(y)
                error += err.item()
                nb_samples += len(y)

                tepoch.set_postfix(loss=loss/nb_samples, error=error/nb_samples)

        return loss/nb_samples, error/nb_samples, loss_nll/nb_samples, loss_kl/nb_samples

    
    def evaluate(self, test_loader):

        '''
        Prediction by sample-averaged predictive distibution,
            (1/S) * \sum_{i=1}^S p(y|x,theta^i) where theta^i ~ q(theta).

        Returns:
            loss = averaged test CE loss
            err = averaged test error
            targets = all groundtruth labels
            logits = all prediction logits (after sample average)
            logits_all = all prediction logits (before sample average)
        '''

        args = self.args

        self.model.eval()
        self.net.eval()
        
        loss, error, nb_samples, targets, logits, logits_all = 0, 0, 0, [], [], []
        with tqdm(test_loader, unit="batch") as tepoch:
            for x, y in tepoch:

                x, y = x.to(args.device), y.to(args.device)

                logits_all_ = []
                if self.nst == 0:  # use just posterior mean
                    with torch.no_grad():
                        for p, p_m in zip(self.net.parameters(), self.model.m.parameters()):
                            p.copy_(p_m)
                        out = self.net(x)
                    logits_all_.append(out)
                    logits_all_ = torch.stack(logits_all_, 2)
                    logits_ = F.log_softmax(logits_all_, 1).logsumexp(-1)
                else:  # use posterior samples
                    for ii in range(self.nst):  # for each sample theta ~ q(theta)
                        with torch.no_grad():
                            _, out, _, _ = self.model(
                                x, y, self.net, self.net0, self.criterion, self.kld, eval_grad=0
                            )
                        logits_all_.append(out)
                    logits_all_ = torch.stack(logits_all_, 2)
                    logits_ = F.log_softmax(logits_all_, 1).logsumexp(-1) - np.log(self.nst)

                loss_ = self.criterion(logits_, y)

                # prediction on test
                pred = logits_.data.max(dim=1)[1]
                err = pred.ne(y.data).sum()

                targets.append(y.cpu().detach().numpy())
                logits.append(logits_.cpu().detach().numpy())  # sampled-averaged logits
                logits_all.append(logits_all_.cpu().detach().numpy())  # sample-wise logits
                loss += loss_.item() * len(y)
                error += err.item()
                nb_samples += len(y)

                tepoch.set_postfix(loss=loss/nb_samples, error=error/nb_samples)

        targets = np.concatenate(targets, axis=0)
        logits = np.concatenate(logits, axis=0)
        logits_all = np.concatenate(logits_all, axis=0)

        return loss/nb_samples, error/nb_samples, targets, logits, logits_all

    
    def save_logits(self, targets, logits, logits_all, suffix=None):

        suffix = '' if suffix is None else f'_{suffix}'
        fname = os.path.join(self.args.log_dir, f'logits{suffix}.pkl')
        
        with open(fname, 'wb') as ff:
            pickle.dump(
                {'targets': targets, 'logits': logits, 'logits_all': logits_all}, 
                ff, protocol=pickle.HIGHEST_PROTOCOL
            )

        return fname


    def save_ckpt(self, epoch):

        fname = os.path.join(self.args.log_dir, f"ckpt.pt")

        torch.save(
            {
                'model': self.model.state_dict(), 
                'prior_sig': self.model.prior_sig, 
                'optimizer': self.optimizer.state_dict(),
                'epoch': epoch, 
            },
            fname
        )

        return fname


    def load_ckpt(self, ckpt_path):
        
        ckpt = torch.load(ckpt_path, map_location=self.args.device)
        
        self.model.load_state_dict(ckpt['model'])
        self.model.prior_sig = ckpt['prior_sig']
        self.optimizer.load_state_dict(ckpt['optimizer'])

        return ckpt['epoch']


class Model(nn.Module):

    '''
    Variational inference model.

    Represents q(theta) = N(theta; m, Diag(v)) where v = s^2 (s = clamp(s_,min=1e-8)).
    '''

    def __init__(self, net, ND, prior_sig=1.0, bias='informative'):

        '''
        Args:
            net = either pretrained or random init backbone (init for m)
            ND = training data size
            prior_sig = prior Gaussian sigma
            bias = how to treat bias parameters:
                "informative": -- the same treatment as weights
                "uninformative": uninformative bias prior
        '''

        super().__init__()

        # create networks for vi params "m" and "s_"
        self.m = copy.deepcopy(net)
        self.s_ = copy.deepcopy(net)

        # initialize s_
        with torch.no_grad():
            for param in self.s_.parameters():
                param.copy_((1e-6)*torch.ones_like(param))  # small value

        self.ND = ND
        self.prior_sig = prior_sig
        self.bias = bias

    
    def _retrieve_s(self):
        
        s = copy.deepcopy(self.s_)
        with torch.no_grad():
            for psrc, ptgt in zip(self.s_.parameters(), s.parameters()):
                ptgt.copy_(psrc.clamp(min=1e-8))

        return s

        
    def forward(self, x, y, net, net0, criterion, kld=1.0, eval_grad=0):

        '''
        Evaluate minibatch -ELBO loss for a given a batch.

        Args:
            x, y = batch input, output
            net = workhorse network (its parameters will be filled in)
            net0 = prior mean parameters
            criterion = loss function
            kld = kl discount factor (to factor in data augmentation)

        Returns:
            loss = -ELBO on the batch; scalar
            out = class prediction on the batch
            loss_nll = nll part of -ELBO
            loss_kl = kl part of -ELBO
        '''

        bias = self.bias

        # sample theta ~ q(theta), ie, theta = m + eps*s, eps~N(0,I)
        with torch.no_grad():
            for p, p_m, p_s_ in zip(net.parameters(), self.m.parameters(), self.s_.parameters()):
                eps = torch.randn_like(p)
                p.copy_(p_m + p_s_.clamp(min=1e-8)*eps)

        # fwd pass with theta
        if eval_grad==0:
            with torch.no_grad():
                out = net(x)
        else:
            out = net(x)

        # evaluate nll loss
        loss_nll = criterion(out, y)

        # gradient d{loss_nll}/d{theta}
        if eval_grad:
            net.zero_grad()
            loss_nll.backward()

        # evaluate kl loss (before scaled by 1/ND), and also compute the total loss gradient
        loss_kl = 0.
        with torch.no_grad():
            for (pname, p), p0, p_m, p_s_ in zip(net.named_parameters(), net0.parameters(), self.m.parameters(), self.s_.parameters()):
                
                # kl
                if not ('bias' in pname and bias == 'uninformative'):
                    d = p.numel()
                    sig2 = self.prior_sig**2
                    s = p_s_.clamp(min=1e-8)
                    v = s**2
                    loss_kl += 0.5 * ( (((p_m-p0)**2+v)/sig2).sum() - (v/sig2).log().sum() - d )

                # gradients
                if eval_grad and p.grad is not None:
                    if not ('bias' in pname and bias == 'uninformative'):
                        p_m.grad = p.grad + kld*(p_m-p0)/sig2/self.ND
                        p_s_.grad = p.grad*((p-p_m)/s) + kld*(s/sig2-1/s)/self.ND

        loss = loss_nll + kld*loss_kl/self.ND

        return loss.item(), out.detach(), loss_nll.item(), loss_kl.item()

