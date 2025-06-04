# SimCLR on CIFAR-10 with ResNets in PyTorch

This is a barebones PyTorch implementation of SimCLR on CIFAR10 with ResNet18/ResNet34/ResNet50 backbones. The purpose is to provide an optimized single-file implementation, with minimial dependencies (only `torch` and `torchvision`, plus `scikit-learn` for evaluation). To use, simply run 
```
python simclr-cifar10.py
```
The code is optimized for readability and hackability. PRs welcome (see below).

|Backbone|Batch|Sec/epoch|Loss/batch|kNN k=5|kNN k=10|lin precomp|lin augm|
|--------|-----|----------|----|--------|-----------|-------|----|
|ResNet18|1024 |~15 s|5.77±.00|90.3±.1|90.4±.1|90.7±.1|90.9±.1|
|ResNet34|1024 |22.2 s|5.75±.00|91.1±.1|91.2±.1|91.3±.1|91.8±.1|
|ResNet50|512  |38.5 s|5.05±.00|91.5±.1|91.6±.1|93.1±.1|93.2±.1|

Standard deviations are over 3 runs. Runtimes are measured on A100 with 16 CPU workers (note that the number of available workers can strongly affect the runtime). I used batch size 512 for ResNet50 because batch size 1024 did not fit into memory (note that the loss values cannot be compared across different batch sizes).

**Training** is done only on the training set --- this is important because doing SimCLR training on training+test sets leads to noticeably higher evaluation results. We use the same set of data augmentations as in the original SimCLR paper. ResNets are modified by replacing the first convolutional layer and removing the first pooling layer, as described in the original SimCLR paper. The projector has output dimensionality 128 and a hidden layer with 1024 neurons. Following the SimSiam paper, we use SGD with momentum 0.9, learning rate 0.03⋅batch_size/256 with cosine annealing, and weight decay 0.0005.

**Evaluation** is done on the test set, using the representation before the projector.
* kNN classifiers use cosine distance (Euclidean distance yields worse results by ~3%).
* `lin precomp` trains a linear readout on precomputed representations. We use Adam (learning rate 0.1 for 100 epochs) with cosine annealing. Running logistic regression from scikit-learn is theoretically equivalent, but in practice tends to give worse results especially on higher-dimensional ResNet50 representations, and is also much slower.
* `lin augm` trains a linear readout using data augmentations (crops and horizontal flips). We use Adam (same parameters) with cosine annealing. This is much slower than `lin precomp` because the representations cannot be precomputed, but tends to give slightly better results.

Pull requests that improve any of these results are very welcome. I can run suggested PRs on A100.

Personally, I prefer `KNeighborsClassifier(n_neighbors=10, metric="cosine")` for evaluation because it is quick, deterministic, and avoids all issues with training a linear readout.

### Training for 100 epochs

For smaller-scale experiments people sometimes train for only 100 epochs. I get better results with base learning rate 0.06 instead of 0.03 (this does not hold when training for 1000 epochs):

|Epochs|LR|Time|Loss/batch|kNN k=5|kNN k=10|lin precomp|lin augm|
|--|----|----|----|--------|-----------|-------|----|
|100|0.06|27 min|5.89±.00|81.1±.2|81.8±.2|84.7±.2|84.0±.2|

(It seems my linear training with augmentations is failing here, not sure why.)

### Airbench (unfinished)

In the spirit of https://github.com/KellerJordan/cifar10-airbench, one could have a competition to reach 90.0% kNN accuracy with self-supervised learning as quickly (wall clock on A100) as possible. Any network architecture, any training approach. This implementation allows to get there in XX epochs, taking XX hours XX minutes:

|Backbone|Batch|Epochs|Time|Loss/batch|kNN k=10|
|--------|-----|------|----|----|--------|
|ResNet18|1024 |XX|XX|±.00|±.1|

## Comparison with the literature

#### ResNet18

* Chen & He (SimSiam paper https://arxiv.org/abs/2011.10566) report **91.1** (Figure D1). They use SGD with momentum, and we borrow their hyperparameters. Here is an unofficial repository implementing their methods: https://github.com/PatrickHua/SimSiam. It uses linear evaluation trained with data augmentations. Rusak et al. (https://arxiv.org/abs/2407.00143) report **90.9** following the SimSiam paper hyperparameters (Table 1).
* Ermolov et al. (W-MSE paper https://arxiv.org/abs/2007.06346) report **91.8** linear accuracy (Table 1). They use Adam on frozen representations for evaluation. They also use Adam for SimCLR training, which I haven not see anywhere else. I tried using Adam with their hyperparameters and scheduler, and I got much worse results. I do not know why. They seem to be using a batch norm layer after the projection head; I do not know if it matters. Their repository is here: https://github.com/htdt/self-supervised.
* Here is a repository https://github.com/p3i0t/SimCLR-CIFAR10 that claims 92.9 accuracy. It uses SGD with suboptimal hyperparameters (base learning rate 0.36, momentum 1e-6), employs some weird implementation choices (like clipping cosine similarity to be non-negative, or accidentally ending up with base learning rate 0.36=0.6*0.6), and does not properly freeze the backbone during evaluation. I do not trust the results there.

#### ResNet34

* I am not aware of any reported results.

#### ResNet50

* Chen et al. (SimCLR paper https://arxiv.org/abs/2002.05709) report **94.0** (Appendix B.9). They used a TensorFlow implementation and LARS for optimization. AFAIK nobody has been able to reproduce that number.
