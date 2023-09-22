import os
import copy
import time
from tqdm import tqdm
import pickle
import numpy as np
import torch
import torch.nn as nn
import torchvision

import calibration


class Runner:

    def __init__(self, net, net0, args, logger):

        '''
        Args:
            net = workhorse backbone (on cpu); expected to be initialised as:
                if args.pretrained == None, all-random initialised
                if args.pretrained != None, feat-ext params = pretrained, readout head = random initialised
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

        # workhorse network, expected to be initialised properly
        self.net = net.to(args.device)

        # create vanilla model (no params involved)
        hparams = args.hparams
        self.model = Model(
            wd=float(args.hparams['wd']),  # weight decay coeff (penalize deviation from pretrained or 0)
            bias=str(hparams['bias'])
        )

        # create optimizer
        self.optimizer = torch.optim.SGD(
            [{'params': [p for pn, p in self.net.named_parameters() if self.net.readout_name not in pn], 'lr': args.lr},
             {'params': [p for pn, p in self.net.named_parameters() if self.net.readout_name in pn], 'lr': args.lr_head}],
            momentum = args.momentum, weight_decay = 0
        )

        self.criterion = torch.nn.CrossEntropyLoss()


    def train(self, train_loader, val_loader, test_loader):

        args = self.args
        logger = self.logger

        logger.info('Start training...')

        epoch = 0

        losses_train = np.zeros(args.epochs)
        errors_train = np.zeros(args.epochs)
        losses_ce_train = np.zeros(args.epochs)
        losses_l2_train = np.zeros(args.epochs)
        if val_loader is not None:
            losses_val = np.zeros(args.epochs)
            errors_val = np.zeros(args.epochs)
        losses_test = np.zeros(args.epochs)
        errors_test = np.zeros(args.epochs)

        best_loss = np.inf

        tic0 = time.time()
        for ep in range(epoch, args.epochs):

            # TODO: Do some optimizer lr scheduling...

            tic = time.time()
            losses_train[ep], errors_train[ep], losses_ce_train[ep], losses_l2_train[ep] = \
                self.train_one_epoch(train_loader)
            toc = time.time()

            prn_str = '[Epoch %d/%d] Training summary: ' % (ep, args.epochs)
            prn_str += 'loss = %.4f (ce = %.4f, l2 = %.4f), prediction error = %.4f ' % (
                losses_train[ep], losses_ce_train[ep], losses_l2_train[ep], errors_train[ep])
            prn_str += '(time: %.4f seconds)' % (toc-tic,)
            logger.info(prn_str)

            # test evaluation
            if ep % args.test_eval_freq == 0:

                # test on validation set (if available)
                if val_loader is not None:
                    tic = time.time()
                    losses_val[ep], errors_val[ep], targets_val, logits_val = self.evaluate(val_loader)
                    toc = time.time()
                    prn_str = f'(Epoch {ep}) Validation summary: '
                    prn_str += 'loss = %.4f, prediction error = %.4f ' % (losses_val[ep], errors_val[ep])
                    prn_str += '(time: %.4f seconds)' % (toc-tic,)
                    logger.info(prn_str)

                # test on test set
                tic = time.time()
                losses_test[ep], errors_test[ep], targets_test, logits_test = self.evaluate(test_loader)
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
                            targets_val, logits_val, suffix='val'
                        )  # save prediction logits on validation set
                        logger.info(f'Logits on val set saved at {fname}')
                    fname = self.save_logits(
                        targets_test, logits_test, suffix='test'
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

        self.net.train()

        loss, loss_ce, loss_l2, error, nb_samples = 0, 0, 0, 0, 0
        with tqdm(train_loader, unit="batch") as tepoch:
            for x, y in tepoch:
        
                x, y = x.to(args.device), y.to(args.device)

                # evaluate minibatch loss (+ L2 regularization) for a given a batch
                loss_, out, loss_ce_, loss_l2_ = self.model.forward(
                    x, y, self.net, self.net0, self.criterion, eval_grad=1
                )

                self.optimizer.step()  # update self.net

                # prediction on training
                pred = out.data.max(dim=1)[1]
                err = pred.ne(y.data).sum()

                loss += loss_ * len(y)
                loss_ce += loss_ce_ * len(y)
                loss_l2 += loss_l2_ * len(y)
                error += err.item()
                nb_samples += len(y)

                tepoch.set_postfix(loss=loss/nb_samples, error=error/nb_samples)

        return loss/nb_samples, error/nb_samples, loss_ce/nb_samples, loss_l2/nb_samples


    def evaluate(self, test_loader):

        '''
        Prediction.

        Returns:
            loss = averaged test CE loss
            err = averaged test error
            targets = all groundtruth labels
            logits = all prediction logits
        '''

        args = self.args

        self.net.eval()
        
        loss, error, nb_samples, targets, logits = 0, 0, 0, [], []
        with tqdm(test_loader, unit="batch") as tepoch:
            for x, y in tepoch:

                x, y = x.to(args.device), y.to(args.device)

                _, logits_, _, _ = self.model.forward(
                    x, y, self.net, self.net0, self.criterion, eval_grad=0
                )
                
                loss_ = self.criterion(logits_, y)

                # prediction on test
                pred = logits_.data.max(dim=1)[1]
                err = pred.ne(y.data).sum()

                targets.append(y.cpu().detach().numpy())
                logits.append(logits_.cpu().detach().numpy())
                loss += loss_.item() * len(y)
                error += err.item()
                nb_samples += len(y)

                tepoch.set_postfix(loss=loss/nb_samples, error=error/nb_samples)

        targets = np.concatenate(targets, axis=0)
        logits = np.concatenate(logits, axis=0)
        
        return loss/nb_samples, error/nb_samples, targets, logits


    def save_logits(self, targets, logits, suffix=None):

        suffix = '' if suffix is None else f'_{suffix}'
        fname = os.path.join(self.args.log_dir, f'logits{suffix}.pkl')
        
        with open(fname, 'wb') as ff:
            pickle.dump(
                {'targets': targets, 'logits': logits}, 
                ff, protocol=pickle.HIGHEST_PROTOCOL
            )

        return fname


    def save_ckpt(self, epoch):

        fname = os.path.join(self.args.log_dir, f"ckpt.pt")

        torch.save(
            {
                'model': self.net.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'epoch': epoch, 
            },
            fname
        )

        return fname


    def load_ckpt(self, ckpt_path):
        
        ckpt = torch.load(ckpt_path, map_location=self.args.device)
        
        self.net.load_state_dict(ckpt['model'])
        self.optimizer.load_state_dict(ckpt['optimizer'])

        return ckpt['epoch']


class Model:

    '''
    Vanilla model.
    '''

    def __init__(self, wd=0, bias='penalty'):

        '''
        Args:
            wd = weight decay coeff (penalize deviation from pretrained or 0)
            bias = how to treat bias parameters:
                "penalty": the same treatment as weight params
                "ignore": ignore bias prior (uninformative bias prior)
        '''

        self.wd = wd
        self.bias = bias
        

    def forward(self, x, y, net, net0, criterion, eval_grad=0):

        '''
        Evaluate minibatch loss (+ L2 regularization) for a given a batch.

        Args:
            x, y = batch input, output
            net = workhorse network
            net0 = pretrained network or 0
            criterion = loss function

        Returns:
            loss = l(theta) + 0.5*wd*||theta-theta0||^2 on the batch
            out = class prediction on the batch
            loss_ce = cross-entropy loss
            loss_l2 = L2 loss
        '''

        bias = self.bias

        # fwd pass with theta
        if eval_grad==0:
            with torch.no_grad():
                out = net(x)
        else:
            out = net(x)

        # evaluate ce loss
        loss_ce = criterion(out, y)

        # gradient d{loss_ce}/d{theta}
        if eval_grad:
            net.zero_grad()
            loss_ce.backward()

        # evaluate L2 loss (before scaled by wd), and also compute the total loss gradient
        loss_l2 = 0.
        with torch.no_grad():
            for (pname, p), (_, p0) in zip(net.named_parameters(), net0.named_parameters()):

                # L2 loss
                if not ('bias' in pname and bias == 'ignore'):
                    loss_l2 += ((p-p0)**2).sum()

                # gradients
                if eval_grad and p.grad is not None:
                    if not ('bias' in pname and bias == 'ignore'):
                        p.grad = p.grad + self.wd*(p-p0)

        loss = loss_ce + 0.5*self.wd*loss_l2

        return loss.item(), out.detach(), loss_ce.item(), loss_l2.item()

