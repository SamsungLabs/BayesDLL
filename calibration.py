'''
Sources:
    (blog) https://towardsdatascience.com/neural-network-calibration-using-pytorch-c44b7221a61#:~:text=The%20Expected%20Calibration%20Error%20(ECE,discrepancy%20between%20accuracy%20and%20confidence.
    (codes) https://colab.research.google.com/drive/1H_XlTbNvjxlAXMW5NuBDWhxF3F2Osg1F?usp=sharing#scrollTo=w1SAqFR7wPvs
'''


import os
import pickle
import numpy as np
import scipy
import scipy.special
import scipy.optimize
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models


def calc_bins(labels, logits, num_bins, temperature=1):

    '''
    Do confidence-based binning, compute bin-wise acc & conf means, etc.

    Args:
        labels = true labels
        logits = prediction logits
        num_bins = number of bins (M)
        temperature = scale (T) for logit to prob conversion
    
    Returns:
        bins = right boundary points of bins
        binned = preds-shaped
        bin_accs = bin-wise averaged accuracy
        bin_confs = bin-wise averaged confidence
        bin_sizes = sizees of bins (i.e, # of p(y=j|x^i) cases for each bin)
    '''

    K = logits.shape[1]  # class cardinality

    # convert labels to one-hot form
    labels_oneh = np.eye(K)[labels].flatten()  # one-hot labels, I(y^true(x^i)=j) for all (i,j)
    
    # convert logits to prediction probs
    preds = scipy.special.softmax(
        logits/temperature, axis=1  # temperature scaling applied
    ).flatten()  # prediction probs, p(y=j|x^i) for all (i,j)
   
    # assign each prediction to a bin
    bins = np.linspace(0, 1+1e-8, num_bins+1)[1:]  # right boundary points of bins
    binned = np.digitize(preds, bins)  # prediction prob (confidence) based bin membership

    # bin-wise averaged accuracy and confidence
    bin_accs = np.zeros(num_bins)
    bin_confs = np.zeros(num_bins)
    bin_sizes = np.zeros(num_bins)
    for bin in range(num_bins):
        bin_sizes[bin] = len(preds[binned==bin])
        if bin_sizes[bin] > 0:
            bin_accs[bin] = (labels_oneh[binned==bin]).sum() / bin_sizes[bin]
            bin_confs[bin] = (preds[binned==bin]).sum() / bin_sizes[bin]

    return bins, binned, bin_accs, bin_confs, bin_sizes


def draw_reliability_plot(bins, bin_accs, fig_name, title=None, ece=None, mce=None, nll=None):

    '''
    Draw reliability plot.

    Args:
        bins = right boundary points of bins
        bin_accs = bin-wise averaged accuracy
    '''

    # get bin ceters exactly (note: bins = right boundary points)
    bin_centers = (np.insert(bins, 0, 0)[:-1] + bins) / 2
    width = bin_centers[1] - bin_centers[0]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.gca()
    ax.set_xlim(0, 1+1e-8)
    ax.set_ylim(0, 1)
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    ax.set_axisbelow(True) 
    ax.grid(color='gray', linestyle='dashed')

    plot_groups = []

    p1 = plt.bar(
        bin_centers, bin_centers, width=width, alpha=0.3, edgecolor='black', color='r', hatch='\\', 
    )  # ideal case
    p2 = plt.bar(
        bin_centers, bin_accs, width=width, alpha=0.3, edgecolor='black', color='b', 
    )  # model's
    p3 = plt.plot([0,1],[0,1], '--', color='gray', linewidth=2, label='Y=X')  # y=x line
    plt.gca().set_aspect('equal', adjustable='box')
   
    plot_groups.append([p1, p2, p3])

    if ece is not None and mce is not None and nll is not None:
        e1 = mpatches.Patch(color='green', label='ECE = {:.2f}%'.format(ece*100))  # ece
        e2 = mpatches.Patch(color='red', label='MCE = {:.2f}%'.format(mce*100))  # mce
        e3 = mpatches.Patch(color='blue', label='NLL = {:.4f}'.format(nll))  # nll
        plot_groups.append([e1, e2, e3])
    
    legend1 = plt.legend(plot_groups[0], labels=['Y=X', 'Ideal', 'Model'], loc='upper left')
    plt.legend(handles=plot_groups[1], loc='lower right')
    plt.gca().add_artist(legend1)

    if title is not None:
        plt.title(title)

    plt.savefig(fig_name, bbox_inches='tight')
    plt.close()


def find_optimal_temperature(labels, logits, plot_save_path, max_iter=10000):

    '''
    Args:
        labels = (validation set) true labels
        logits = (validation set) prediction logits
        plot_save_path = path to save optimization curve
        
    Returns:
        Topt = optimal temperature that minimizes the validation nll
    '''

    use_torch = False

    if use_torch:

        labels = torch.from_numpy(labels)
        logits = torch.from_numpy(logits)

        T = nn.Parameter(torch.ones(1))  # init with T=1
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.LBFGS([T,], lr=0.001, max_iter=max_iter, line_search_fn='strong_wolfe')
        #optimizer = optim.LBFGS([T,], lr=0.001, max_iter=max_iter)
        temps, losses = [], []
        
        def _eval():
            loss = criterion(torch.div(logits, T), labels)
            optimizer.zero_grad()
            loss.backward()
            temps.append(T.item())
            losses.append(loss.item())
            return loss
        
        optimizer.step(_eval)
        print('Final T_scaling factor: {:.2f}'.format(T.item()))

        plt.subplot(121)
        plt.plot(list(range(len(temps))), temps)
        plt.gca().set_title('Temperature T')
        plt.gca().set_xlabel('Iterations')
        plt.subplot(122)
        plt.plot(list(range(len(losses))), losses)
        plt.gca().set_title('NLL on validation set')
        plt.gca().set_xlabel('Iterations')
        plt.show()
        plt.savefig(plot_save_path)
        plt.close()

        Topt = T.item()
        success = True
    
    else:

        # objective function
        def fun(T):
            logits_ = logits/T
            nll = np.mean(
                scipy.special.logsumexp(logits_, axis=1) - logits_[np.arange(len(labels)), labels]
            )
            return nll

        temps, losses = [], []

        # callback function (to collect intermediate results)
        def callback(x):
            nonlocal fun
            temps.append(x)
            losses.append(fun(x))

        T = np.ones(1)  # init with T=1
        result = scipy.optimize.minimize(fun, T, options={'maxiter': max_iter}, callback=callback)

        success = result.success
        try:
            Topt = result.x
            plt.subplot(121)
            plt.plot(list(range(len(temps))), temps)
            plt.gca().set_title('Temperature T')
            plt.gca().set_xlabel('Iterations')
            plt.subplot(122)
            plt.plot(list(range(len(losses))), losses)
            plt.gca().set_title('NLL on validation set')
            plt.gca().set_xlabel('Iterations')
            plt.show()
            plt.savefig(plot_save_path)
            plt.close()
        except:
            Topt = 1

    return Topt, success


def analyze(labels, logits, num_bins, plot_save_path, temperature=1):

    '''
    Perform error calibration: ECE, MCE, reliability plot, etc.

    Args:
        labels = true labels
        logits = prediction logits
        num_bins = number of bins (M)
        plot_save_path = path to save reliability plot
        temperature = scale (T) for logit to prob conversion

    Returns:
        ece = expected calibration error
        mce = maximum calibration error
        nll = negative log-likelihood

    Effects:
        reliability plot is generated and saved
    '''
    
    # confidence-binning and compute bin-wise average accuracy and confidence
    bins, binned, bin_accs, bin_confs, bin_sizes = calc_bins(
        labels, logits, num_bins, temperature
    )

    # ece and mce
    ece = ( np.abs(bin_accs-bin_confs) * (bin_sizes/bin_sizes.sum()) ).sum()
    mce = np.abs(bin_accs-bin_confs).max()

    # nll score
    logits_ = logits/temperature
    nll = np.mean(
        scipy.special.logsumexp(logits_, axis=1) - logits_[np.arange(len(labels)), labels]
    )

    # reliability plot
    draw_reliability_plot(
        bins, bin_accs, 
        plot_save_path, 
        title=f'Temperature = {temperature}', 
        ece=ece, mce=mce, nll=nll
    )

    return ece, mce, nll

