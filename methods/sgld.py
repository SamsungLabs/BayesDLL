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

import calibration


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

        # workhorse network (current LD sample is maintained in here)
        self.net = net.to(args.device)

        # create langevin mcmc model (nn.Module with actually no parameters)
        hparams = args.hparams
        self.model = Model(
            ND=args.ND, prior_sig=float(hparams['prior_sig']), bias=str(hparams['bias'])
        ).to(args.device)

        # create optimizer (for workhorse network -- to update LD sample)
        # self.optimizer = torch.optim.SGD(
        #     self.net.parameters(), 
        #     lr = args.lr, momentum = args.momentum, weight_decay = 0
        # )
        self.optimizer = torch.optim.SGD(
            [{'params': [p for pn, p in self.net.named_parameters() if self.net.readout_name not in pn], 'lr': args.lr},
             {'params': [p for pn, p in self.net.named_parameters() if self.net.readout_name in pn], 'lr': args.lr_head}],
            momentum = args.momentum, weight_decay = 0
        )

        # TODO: create scheduler?

        self.criterion = torch.nn.CrossEntropyLoss()

        self.Ninflate = float(hparams['Ninflate'])  # N inflation factor (factor in data augmentation)
        self.nd = float(hparams['nd'])  # noise discount
        self.burnin = int(hparams['burnin'])  # burnin periord (in epochs)
        self.thin = int(hparams['thin'])  # thinning steps (in batch iters)
        self.nst = int(hparams['nst'])  # number of samples at test time


    def train(self, train_loader, val_loader, test_loader):

        args = self.args
        logger = self.logger

        logger.info('Start training...')

        epoch = 0

        losses_train = np.zeros(args.epochs)
        errors_train = np.zeros(args.epochs)
        if val_loader is not None:
            losses_val = np.zeros(args.epochs)
            errors_val = np.zeros(args.epochs)
        losses_test = np.zeros(args.epochs)
        errors_test = np.zeros(args.epochs)

        best_loss = np.inf

        tic0 = time.time()
        bi = 0  # batch iteration count
        for ep in range(epoch, args.epochs):

            # TODO: Do some optimizer lr scheduling...

            # check if the burnin period is just finished; if so, prepare collecting posterior samples
            if ep == self.burnin:
                logger.info('(leaving burnin period) start collecting posterior samples')
                with torch.no_grad():
                    theta_vec = nn.utils.parameters_to_vector(self.net.parameters())
                    self.post_theta_mom1 = theta_vec*1.0  # 1st moment
                    if self.nst > 0:  # need to maintain sample variances as well
                        self.post_theta_mom2 = theta_vec**2  # 2nd moment
                self.post_theta_cnt = 1

            tic = time.time()
            losses_train[ep], errors_train[ep], bi = self.train_one_epoch(
                train_loader, collect=(ep>=self.burnin), bi=bi
            )
            toc = time.time()

            prn_str = '[Epoch %d/%d] Training summary: ' % (ep, args.epochs)
            prn_str += 'loss = %.4f, prediction error = %.4f ' % (losses_train[ep], errors_train[ep])
            prn_str += '(time: %.4f seconds)' % (toc-tic,)
            logger.info(prn_str)

            # test evaluation (only after burnin period)
            if ep % args.test_eval_freq == 0 and ep >= self.burnin:

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


    def train_one_epoch(self, train_loader, collect, bi):

        '''
        Run SGLD steps for one epoch. May collect posterior samples if post-burnin.

        Args:
            collect = whether to collect posterior samples or not
            bi = batch iteration# (used for thinning steps)

        Returns:
            loss = averaged NLL loss
            err = averaged classification error
            bi = updated batch iteration#
        '''

        args = self.args
        logger = self.logger

        self.net.train()
        
        loss, error, nb_samples = 0, 0, 0
        with tqdm(train_loader, unit="batch") as tepoch:
            for x, y in tepoch:

                x, y = x.to(args.device), y.to(args.device)

                # evaluate SGLD updates for a given a batch
                loss_, out = self.model(
                    x, y, self.net, self.net0, self.criterion, 
                    [pg['lr'] for pg in self.optimizer.param_groups],
                    self.Ninflate, self.nd
                )

                self.optimizer.step()  # update self.net (ie, theta)

                # prediction on training
                pred = out.data.max(dim=1)[1]
                err = pred.ne(y.data).sum()

                loss += loss_ * len(y)
                error += err.item()
                nb_samples += len(y)

                bi += 1

                # if post-burnin, collect posterior samples every thinning steps
                if collect and bi % self.thin == 0:
                    logger.info('(post-burnin) accumulate posterior samples')
                    with torch.no_grad():
                        theta_vec = nn.utils.parameters_to_vector(self.net.parameters())
                        self.post_theta_mom1 = (theta_vec + self.post_theta_cnt*self.post_theta_mom1) / (self.post_theta_cnt+1)
                        if self.nst > 0:  # need to maintain sample variances as well
                            self.post_theta_mom2 = (theta_vec**2 + self.post_theta_cnt*self.post_theta_mom2) / (self.post_theta_cnt+1)
                    self.post_theta_cnt += 1

                tepoch.set_postfix(loss=loss/nb_samples, error=error/nb_samples)

        return loss/nb_samples, error/nb_samples, bi


    def evaluate(self, test_loader):

        '''
        Prediction by sample-averaged predictive distibution, 
            (1/S) * \sum_{i=1}^S p(y|x,theta^i) where theta^i ~ p(theta|D) from SGLD.

        Returns:
            loss = averaged test CE loss
            err = averaged test error
            targets = all groundtruth labels
            logits = all prediction logits (after sample average)
            logits_all = all prediction logits (before sample average)
        '''

        args = self.args

        # get current posterior mean, vars
        post_theta_mean, post_theta_vars = self.get_mean_vars_from_moments()
        
        net = copy.deepcopy(self.net)  # workhorse network for this evaluation

        net.eval()
        
        loss, error, nb_samples, targets, logits, logits_all = 0, 0, 0, [], [], []
        with tqdm(test_loader, unit="batch") as tepoch:
            for x, y in tepoch:

                x, y = x.to(args.device), y.to(args.device)

                logits_all_ = []
                if self.nst == 0:  # use just posterior mean
                    with torch.no_grad():
                        for p, p_m in zip(net.parameters(), post_theta_mean.parameters()):
                            p.copy_(p_m)
                        out = net(x)
                    logits_all_.append(out)
                    logits_all_ = torch.stack(logits_all_, 2)
                    logits_ = F.log_softmax(logits_all_, 1).logsumexp(-1)
                else:  # use posterior samples
                    for ii in range(self.nst):  # for each sample theta ~ p(theta|D)
                        with torch.no_grad():
                            for p, p_m, p_v in zip(net.parameters(), post_theta_mean.parameters(), post_theta_vars.parameters()):
                                eps = torch.randn_like(p)
                                p.copy_(p_m + p_v.sqrt()*eps)
                            out = net(x)
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


    def get_mean_vars_from_moments(self):

        '''
        Compute posterior mean and variances from the momnet statistics.

        Returns:
            post_theta_mean = net-like nn.Module with posterior mean values
            post_theta_vars = net-like nn.Module with posterior variance values
        '''

        with torch.no_grad():
            post_theta_mean = copy.deepcopy(self.net)
            nn.utils.vector_to_parameters(self.post_theta_mom1, post_theta_mean.parameters())

        post_theta_vars = None
        if self.nst > 0:
            if self.post_theta_cnt > 1:
                ratio = self.post_theta_cnt / (self.post_theta_cnt - 1)
            else:
                ratio = 1.0
            with torch.no_grad():
                post_vars_vec = ratio * (self.post_theta_mom2 - self.post_theta_mom1**2)  # unbiased estimate
                post_vars_vec.clamp_(min=1e-12)  # to avoid numerical error
                post_theta_vars = copy.deepcopy(self.net)
                nn.utils.vector_to_parameters(post_vars_vec, post_theta_vars.parameters())

        return post_theta_mean, post_theta_vars


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
                'post_theta_mom1': self.post_theta_mom1,  
                'post_theta_mom2': self.post_theta_mom2 if self.nst>0 else None, 
                'post_theta_cnt': self.post_theta_cnt, 
                'prior_sig': self.model.prior_sig, 
                'optimizer': self.optimizer.state_dict(),
                'epoch': epoch, 
            },
            fname
        )

        return fname


    def load_ckpt(self, ckpt_path):
        
        ckpt = torch.load(ckpt_path, map_location=self.args.device)

        self.post_theta_mom1 = ckpt['post_theta_mom1']
        if ckpt['post_theta_mom2'] is not None:
            self.post_theta_mom2 = ckpt['post_theta_mom2']
        self.post_theta_cnt = ckpt['epoch']
        self.model.prior_sig = ckpt['prior_sig']
        self.optimizer.load_state_dict(ckpt['optimizer'])

        return ckpt['epoch']

    
class Model(nn.Module):

    '''
    SGLD sampler model.

    Actually no parameters involved.
    '''

    def __init__(self, ND, prior_sig=1.0, bias='informative'):

        '''
        Args:
            ND = training data size
            prior_sig = prior Gaussian sigma
            bias = how to treat bias parameters:
                "informative": -- the same treatment as weights
                "uninformative": uninformative bias prior
        '''

        super().__init__()

        self.ND = ND
        self.prior_sig = prior_sig
        self.bias = bias


    def forward(self, x, y, net, net0, criterion, lrs, Ninflate=1.0, nd=1.0):

        '''
        Evaluate minibatch SGLD updates for a given a batch.

        Args:
            x, y = batch input, output
            net = workhorse network (its parameters will be filled in)
            net0 = prior mean parameters
            criterion = loss function
            lrs = learning rates in list (adjusted to "eta" in SGLD)
            Ninflate = inflate N by this order of magnitude
            nd = noise discount factor

        Returns:
            loss = NLL loss on the batch
            out = class prediction on the batch

        Effects:
            net has .grad fields filled with SGLD updates
        '''

        bias = self.bias

        N = self.ND * Ninflate  # inflated training data size (accounting for data augmentation, etc.)

        if len(lrs) == 1:
            lr_body, lr_head = lrs[0], lrs[0]
        else:
            lr_body, lr_head = lrs[0], lrs[1]

        # fwd pass with theta
        out = net(x)

        # evaluate nll loss
        loss = criterion(out, y)

        # gradient d{loss_nll}/d{theta}
        net.zero_grad()
        loss.backward()

        # compute and set: grad = -(1/N) * d{logp(th)}/d{th} + d{loss}/d{th} + sqrt{2/(N*lr)} * N(0,I)
        with torch.no_grad():
            for (pname, p), p0 in zip(net.named_parameters(), net0.parameters()):
                if p.grad is not None:
                    if net.readout_name not in pname:
                        lr = lr_body
                    else:
                        lr = lr_head
                    if 'bias' in pname and bias == 'uninformative':
                        p.grad = p.grad + (
                            nd * np.sqrt(2/(N*lr)) * torch.randn_like(p)  # sqrt{2/(N*lr)} * N(0,I)
                        )
                    else:
                        p.grad = p.grad + (
                            (p-p0)/(self.prior_sig**2)/N +  # scaled negative log-prior grad = -(1/N) * d{logp(th)}/d{th}
                            nd * np.sqrt(2/(N*lr)) * torch.randn_like(p)  # sqrt{2/(N*lr)} * N(0,I)
                        )

        return loss.item(), out.detach()

