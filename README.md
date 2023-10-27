# BayesDLL: Bayesian Deep Learning Library

We release a new Bayesian neural network library for PyTorch for large-scale deep networks. Our library implements mainstream approximate Bayesian inference algorithms: **variational inference**, **MC-dropout**, **stochastic-gradient MCMC**, and **Laplace approximation**. The main differences from other existing Bayesian neural network libraries are as follows: 
**1)** Our library can deal with very large-scale deep networks including Vision Transformers (ViTs). 
**2)** We need virtually zero code modifications for users (e.g., the backbone network definition codes do not neet to be modified at all). 
**3)** Our library also allows the pre-trained model weights to serve as a prior mean, which is very useful for performing Bayesian inference with the large-scale foundation models like ViTs that are hard to optimise from the scratch with the downstream data alone. 

### *The technical report for the details of algorithms and implementation can be found in: [http://arxiv.org/abs/2309.12928](http://arxiv.org/abs/2309.12928)*

---

## Features

* Full implementation (without relying on other libraries) and easy-to-use demo codes for: **variational inference**, **MC-dropout**, **stochastic-gradient Langevin dynamics**, and **Laplace approximation**.
* Codes for evaluating **Uncertainty Quantification** measures provided (eg, **ECE**, **MCE**, **Reliability plots**, **Negative log-likelihood**).
* Fully tested with ResNet-101 and ViT-L-32. But ready to be applicable to other Foundation Models (eg, LLAMA, RoBERTa, Denoising Diffusion generative models) without code modification at all!
* Minimal/acceptable use of extra computational resources (time & GPU memory) -- See our technical report in arXiv for details.
* Technical derivations, algorithms, and implementation details described/summarized in our technical report in arXiv.


## Environment setup

* ```Python``` >= 3.9
* ```PyTorch``` (>=2.0 recommended)
* ```Torchvision``` (>=0.15 recommended)
* ```tqdm```, ```scipy```, ```matplotlib```

 *In case you want to use other deep neural networks, you may need to install related libraries accordingly (eg, ```huggingface```'s ```transformers``` or Ross Wightman's ```timm```)*

### (Optional) Python package setup (New on 27/10/2023)

For those who would like to install the library as an independent python package (titled ```bayesdll```), we have provided  ```setup.py```. First you go to the code root folder and type the following:

```
python setup.py install
```

Then you can use our library in any folder anywhere you like, eg, 

```
# import inference methods
from bayesdll import vi, mc_dropout, sgld, la

# or

# Bayesian deep model inference and test evaluation
from bayesdll.vi import Runner
runner = Runner(net, net0, args, logger)
runner.train(train_loader, val_loader, test_loader)

# or

# uncertainty quantification
import bayesdll.calibration as calibration
calibration.analyze(...)
```



## Usage examples

[Pseudocodes](./figures/pseudocode.html.pdf)

### 1) MNIST (val_heldout = 0.5, network = MLP)

* Vanilla (no Bayesian) -- weight decay = 1e-4, bias treatment = "penalty"
```
python demo_mnist.py --dataset mnist --backbone mlp_mnist --val_heldout 0.5 --method vanilla --hparams wd=1e-4,bias=penalty --epochs 100 --lr 1e-2 --batch_size 128 --momentum 0.5
```

* Variational Inference -- prior sigma = 1.0, kl discount = 1e-3, bias treatment = "informative", nst = 5
```
python demo_mnist.py --dataset mnist --backbone mlp_mnist --val_heldout 0.5 --method vi --hparams prior_sig=1.0,kld=1e-3,bias=informative,nst=5 --epochs 100 --lr 1e-2 --batch_size 128 --momentum 0.5
```

* MC-Dropout -- prior sigma = 1.0, drop prob = 0.1, kl discount = 1e-3, bias treatment = "gaussian", nst = 0
```
python demo_mnist.py --dataset mnist --backbone mlp_mnist --val_heldout 0.5 --method mc_dropout --hparams prior_sig=1.0,p_drop=0.1,kld=1e-3,bias=gaussian,nst=0 --epochs 100 --lr 1e-2 --batch_size 128 --momentum 0.5
```

* SGLD -- prior sigma = 1.0, N-inflation = 1e3, nd = 1.0, burnin = 5 epochs, thin = 10 iters, bias treatment = "informative", nst = 5
```
python demo_mnist.py --dataset mnist --backbone mlp_mnist --val_heldout 0.5 --method sgld --hparams prior_sig=1.0,Ninflate=1e3,nd=1.0,burnin=5,thin=10,bias=informative,nst=5 --epochs 100 --lr 1e-2 --batch_size 128 --momentum 0.5 
```

* Laplace approximation -- prior sigma = 0.01, N-inflation = 1e3, bias treatment = "informative", nst = 5
```
python demo_mnist.py --dataset mnist --backbone mlp_mnist --val_heldout 0.5 --method la --hparams prior_sig=0.01,Ninflate=1e3,bias=informative,nst=5 --epochs 100 --lr 1e-2 --batch_size 128 --momentum 0.5
```


### 2) Pets (val_heldout = 0.5, network = resnet101, pretrained = IMAGENET1K_V1)

* Vanilla (no Bayesian) -- weight decay = 1e-4, bias treatment = "penalty"
```
python demo_vision.py --dataset pets --val_heldout 0.5 --backbone resnet101 --pretrained IMAGENET1K_V1 --method vanilla --hparams wd=1e-4,bias=penalty --epochs 100 --lr 1e-4 --lr_head 1e-2 --batch_size 16 --momentum 0.5 
```

* Variational Inference -- prior sigma = 1.0, kl discount = 1e-6, bias treatment = "informative", nst = 5
```
python demo_vision.py --dataset pets --val_heldout 0.5 --backbone resnet101 --pretrained IMAGENET1K_V1 --method vi --hparams prior_sig=1.0,kld=1e-6,bias=informative,nst=5 --epochs 100 --lr 1e-4 --lr_head 1e-2 --batch_size 16 --momentum 0.5 
```

* MC-Dropout -- prior sigma = 1.0, drop prob = 0.1, kl discount = 1e-3, bias treatment = "gaussian q", nst = 5
```
python demo_vision.py --dataset pets --val_heldout 0.5 --backbone resnet101 --pretrained IMAGENET1K_V1 --method mc_dropout --hparams prior_sig=1.0,p_drop=0.1,kld=1e-3,bias=gaussian,nst=5 --epochs 100 --lr 1e-4 --lr_head 1e-2 --batch_size 16 --momentum 0.5
```

* SGLD -- prior sigma = 1.0, N-inflation = 1e3, nd=0.01, burnin = 50 epochs, thin = 10 iters, bias treatment = "informative", nst = 5
```
python demo_vision.py --dataset pets --val_heldout 0.5 --backbone resnet101 --pretrained IMAGENET1K_V1 --method sgld --hparams prior_sig=1.0,Ninflate=1e3,nd=0.01,burnin=50,thin=10,bias=informative,nst=5 --epochs 100 --lr 1e-4 --lr_head 1e-2 --batch_size 16 --momentum 0.5
```

* Laplace approximation -- prior sigma = 1.0, N-inflation = 1e3, bias treatment = "informative", nst = 0
```
python demo_vision.py --dataset pets --val_heldout 0.5 --backbone resnet101 --pretrained IMAGENET1K_V1 --method la --hparams prior_sig=1.0,Ninflate=1e3,bias=informative,nst=0 --epochs 100 --lr 1e-4 --lr_head 1e-2 --batch_size 16 --momentum 0.5
```


### 3) Pets (val_heldout = 0.5, network = vit_l_32, pretrained = IMAGENET1K_V1)

* Vanilla (no Bayesian) -- weight decay = 1e-4, bias treatment = "penalty"
```
python demo_vision.py --dataset pets --val_heldout 0.5 --backbone vit_l_32 --pretrained IMAGENET1K_V1 --method vanilla --hparams wd=1e-4,bias=penalty --epochs 100 --lr 1e-4 --lr_head 1e-2 --batch_size 16 --momentum 0.5
```

* Variational Inference -- prior sigma = 1.0, kl discount = 1e-6, bias treatment = "informative", nst = 5
```
python demo_vision.py --dataset pets --val_heldout 0.5 --backbone vit_l_32 --pretrained IMAGENET1K_V1 --method vi --hparams prior_sig=1.0,kld=1e-6,bias=informative,nst=5 --epochs 100 --lr 1e-4 --lr_head 1e-2 --batch_size 16 --momentum 0.5
```

* MC-Dropout -- prior sigma = 1.0, drop prob = 0.1, kl discount = 1e-3, bias treatment = "gaussian q", nst = 5
```
python demo_vision.py --dataset pets --val_heldout 0.5 --backbone vit_l_32 --pretrained IMAGENET1K_V1 --method mc_dropout --hparams prior_sig=1.0,p_drop=0.1,kld=1e-3,bias=gaussian,nst=5 --epochs 100 --lr 1e-4 --lr_head 1e-2 --batch_size 16 --momentum 0.5
```

* SGLD -- prior sigma = 1.0, N-inflation = 1e3, nd=0.01, burnin = 50 epochs, thin = 10 iters, bias treatment = "informative", nst = 5
```
python demo_vision.py --dataset pets --val_heldout 0.5 --backbone vit_l_32 --pretrained IMAGENET1K_V1 --method sgld --hparams prior_sig=1.0,Ninflate=1e3,nd=0.01,burnin=50,thin=10,bias=informative,nst=5 --epochs 100 --lr 1e-4 --lr_head 1e-2 --batch_size 16 --momentum 0.5
```

* Laplace approximation -- prior sigma = 1.0, N-inflation = 1e3, bias treatment = "informative", nst = 0
```
python demo_vision.py --dataset pets --val_heldout 0.5 --backbone vit_l_32 --pretrained IMAGENET1K_V1 --method la --hparams prior_sig=1.0,Ninflate=1e3,bias=informative,nst=0 --epochs 100 --lr 1e-4 --lr_head 1e-2 --batch_size 16 --momentum 0.5
```


## Citation
If you found this library useful in your research, please cite:
```
@inproceedings{bayesdll_kim_hospedales_2023,
 title = {{BayesDLL: Bayesian Deep Learning Library}},
 author  = {Kim, Minyoung and Hospedales, Timothy},
 year  = {2023},
 URL = {http://arxiv.org/abs/2309.12928},
 booktitle = {arXiv preprint arXiv:2309.12928}
}
```
