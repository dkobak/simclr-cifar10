# SimCLR on CIFAR-10 with ResNets in PyTorch

This is a barebones PyTorch implementation of SimCLR on CIFAR10 with ResNet18/ResNet34/ResNet50 backbones. The purpose is to provide an optimized single-file implementation, with minimial dependencies (only `torch` and `torchvision`, plus `scikit-learn` for evaluation). To use, simply run `python simclr-cifar10.py`.

The code is optimized for readability and hackability. PRs welcome (see below).

|Backbone|Batch|Sec/epoch|Loss/batch|k=1|k=5|k=10|lin sklearn|lin frozen|lin trained|
|--------|-----|----------|----|--------|--------|---------|-----------|-------|----|
|ResNet18|1024|18.2 s|5.77±.00|89.0±.1|90.3±.2|90.4±.1|90.8±.1|90.9±.1|91.1±.1|
|ResNet34|1024|~22 s||||||||
|ResNet50|512 |||||||||

Standard deviations are over 3 runs. Runtimes are measured on A100 with 16 CPU workers (note that the number of available workers can strongly affect the runtime). `k=...` columns refer to kNN classifiers.

**Training** is done only on the training set --- this is important because doing SimCLR training on training+test sets leads to noticeably higher evaluation results. We use the same set of data augmentations as in the original SimCLR paper. ResNets are modified by replacing the first convolutional layer and removing the first pooling layer, as described in the original SimCLR paper. The projector has output dimensionality 128 and a hidden layer with 1024 neurons. Following the SimSiam paper, we use SGD with momentum 0.9, learning rate 0.03⋅batch_size/256 with cosine annealing, and weight decay 0.0005.

**Evaluation** is done on the test set, using the representation before the projector.
* kNN classifiers (with `k=1`, `k=5`, `k=10`) use cosine distance (better results than Euclidean distance);
* `lin sklearn` runs logistic regression from scikit-learn on frozen representations;
* `lin frozen` trains a linear readout on frozen representations (in principle this should be equivalent);
* `lin trained` trains a linear readout using data augmentations (crops and horizontal flips).

Pull requests that improve any of these results are very welcome. I can run suggested PRs on A100.

## Comparison with the literature

#### ResNet18

* Chen & He (SimSiam paper https://arxiv.org/abs/2011.10566) report **91.1** (Figure D1). They use SGD with momentum, and we borrow their hyperparameters. Here is an unofficial repository implementing their methods: https://github.com/PatrickHua/SimSiam. It uses linear evaluation trained with data augmentations.
* Evgenia Rusak from the Brendel lab got **91.5±.1** following the SimSiam paper hyperparameters (personal communucation). I do not know why her results are better.
* On the other hand, Rusak et al. (https://arxiv.org/abs/2407.00143) report **90.9** following the SimSiam paper hyperparameters (Table 1).
* Ermolov et al. (W-MSE paper https://arxiv.org/abs/2007.06346) report **91.8** linear accuracy (Table 1). They use Adam on frozen representations for evaluation. They also use Adam for SimCLR training, which I haven not see anywhere else. I tried using Adam with their hyperparameters and scheduler, and I got much worse results. I do not know why. They seem to be using a batch norm layer after the projection head; I do not know if it matters. Their repository is here: https://github.com/htdt/self-supervised.
* Here is some repository https://github.com/p3i0t/SimCLR-CIFAR10 that claims 92.9 accuracy. It uses SGD with suboptimal hyperparameters (base learning rate 0.36, momentum 1e-6), and some weird implementation choices (like clipping cosine similarity to be non-negative, or accidentally ending up with base learning rate 0.36=0.6*0.6). I was unable to reproduce their results, and I do not trust them.

#### ResNet34

I am not aware of any reported results.

#### ResNet50

* Evgenia Rusak from the Brendel lab got **93.2±.1** following the SimSiam paper hyperparameters (personal communucation).
* Chen et al. (SimCLR paper https://arxiv.org/abs/2002.05709) report **94.0** (Appendix B.9). They used a TensorFlow implementation and LARS for optimization. AFAIK nobody has been able to reproduce that number.
