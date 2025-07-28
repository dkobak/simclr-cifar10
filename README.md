# SimCLR on CIFAR-10 with ResNets in PyTorch

This is a barebones PyTorch implementation of [SimCLR](https://arxiv.org/abs/2002.05709) on CIFAR10 with ResNet18/ResNet34/ResNet50 backbones. The purpose is to provide an optimized single-file implementation, with minimial dependencies (only `torch` and `torchvision`, plus `scikit-learn` for evaluation). To use, simply run 
```
python simclr-cifar10.py
```
The code is optimized for readability and hackability. Results are SOTA for SimCLR on CIFAR10. PRs welcome.

|Backbone|Batch|Sec/epoch|Loss/batch|kNN k=10|lin precomp|lin augm|
|--------|-----|----------|----|-----------|-------|----|
|ResNet18|512  |11.9 s|5.07±.00|91.4±.0|91.7±.1|91.9±.1|
|ResNet34|512  |19.6 s|5.05±.00|92.1±.1|92.0±.3|92.7±.0|
|ResNet50|512  |36.9 s|5.05±.00|92.2±.0|93.4±.0|93.7±.1|

Results after 1000 epochs. Standard deviations are over 3 runs. Runtimes are measured on A100 with 16 CPU workers (the number of available workers can strongly affect the runtime). I used batch size 512 because batch size 1024 did not fit into memory for ResNet50 and I wanted to make the loss values comparable across architectures. Larger batch sizes did not improve the performance on ResNet18, so batch size 512 is sufficient.

**Training** is done only on the training set --- this is important because doing SimCLR training on training+test sets leads to noticeably higher evaluation results.
* I use the same set of data augmentations as in the [original SimCLR paper](https://arxiv.org/abs/2002.05709), but set the grayscale probability to 0.1 following [Ermolov et al. 2020](https://arxiv.org/abs/2007.06346) (it improves the results).
* ResNets are modified by replacing the first convolutional layer and removing the first pooling layer, as described in the original SimCLR paper. The projection head has output dimensionality 128 and a hidden layer with 1024 neurons.
* Following [the SimSiam paper](https://arxiv.org/abs/2011.10566), we train for 1000 epochs using SGD with momentum 0.9, learning rate 0.03⋅batch_size/256 with cosine annealing, and weight decay 0.0005. Adam works nearly as well but requires smaller weight decay and learning rate warmup, similar to Ermolov et al. 2020 (Adam code is commented out; SGD gives marginally better results).

**Evaluation** is done on the test set, using the representation before the projection head.
* kNN classifier uses cosine distance (Euclidean distance yields worse results by ~3 percentage points).
* `lin precomp` trains a linear readout on precomputed representations. Here I follow Ermolov et al. and use Adam (learning rate 0.01 for 500 epochs with weight decay 5e-6) with cosine annealing.  For ResNet18 and ResNet34 (512-dim representations) I got almost equally good results using logistic regression from `scikit-learn`; for ResNet50 (2048-dim representations) it was worse.
* `lin augm` trains a linear readout using data augmentations (crops and horizontal flips). This is slower than `lin precomp` because the representations cannot be precomputed, but tends to give slightly better results. I use Adam (learning rate 0.1 for 100 epochs with weight decay 5e-6) with cosine annealing.

Pull requests that improve any of these results are very welcome. I can run suggested PRs on A100.

Personally, I prefer `KNeighborsClassifier(n_neighbors=10, metric="cosine")` for evaluation because it is quick, deterministic, and avoids all issues with training a linear readout.

### Training ResNet18 for 100 epochs

For smaller-scale experiments people sometimes train ResNet18 for only 100 epochs. Here I get better results with base learning rate 0.06 instead of 0.03 (this does not hold when training for 1000 epochs):

|Epochs|LR|Time|Loss/batch|kNN k=10|lin precomp|lin augm|
|--|----|----|----|--------|-------|----|
|100|0.06|20 min|5.18±.01|83.2±.2|85.6±.2|85.0±.3|

I am not sure why `lin precomp` works better than `lin augm` here.

### Training till kNN accuracy 90.0%

In the spirit of https://github.com/KellerJordan/cifar10-airbench, one can have a competition to reach 90.0% kNN accuracy with self-supervised learning as quickly (wall clock on A100) as possible. Any network architecture, any training approach. This implementation allows to get there in 500 epochs, taking 1 hour 39 minutes:

|Backbone|Batch|Epochs|Time|Loss/batch|kNN k=10|
|--------|-----|------|----|----|--------|
|ResNet18|512 |500| 1h 39 min|5.10±.00|90.3±.3|

## Comparison with the literature

TL/DR: This implementation is SOTA.

#### ResNet18

* Chen & He (SimSiam paper https://arxiv.org/abs/2011.10566) report 91.1 linear accuracy (Figure D1). They use SGD with momentum, and we borrow their hyperparameters. Here is an unofficial repository implementing their methods: https://github.com/PatrickHua/SimSiam. It uses linear evaluation trained with data augmentations. Rusak et al. (https://arxiv.org/abs/2407.00143) report 90.9 following the SimSiam paper hyperparameters (Table 1). We get better results (91.8 with `lin augm`) mainly due to grayscaling probability set to 0.1 (with 0.2 our results are similar to Chen & He and Rusak et al.).
* Ermolov et al. (W-MSE paper https://arxiv.org/abs/2007.06346) report 91.8 linear accuracy on frozen representations (Table 1), and I confirmed that using their repo: https://github.com/htdt/self-supervised. This matches my results (91.7 with `lin precomp`). They use Adam on frozen representations for evaluation and also use Adam for SimCLR training. I added kNN evaluation to their code and got 90.5, so their kNN results are worse than mine (91.4) and similar to what I get with Adam. Note that they use two batch norm layers in the projection head (this seems to have only a very minor effect though). They use grayscaling probability 0.1.
* Here is a repository https://github.com/p3i0t/SimCLR-CIFAR10 that claims 92.9 accuracy. It uses SGD with suboptimal hyperparameters (base learning rate 0.36, momentum 1e-6), employs some weird implementation choices (like clipping cosine similarity to be non-negative, or accidentally ending up with base learning rate 0.36=0.6*0.6), and does not properly freeze the backbone during evaluation. So these results are invalid.

#### ResNet34

* I am not aware of any reported results.

#### ResNet50

* Chen et al. (SimCLR paper https://arxiv.org/abs/2002.05709) report 94.0 linear accuracy (Appendix B.9). They used a TensorFlow implementation and LARS for optimization. AFAIK nobody has been able to reproduce that number. I get very close with 93.7 on average (but with grayscale probability set to 0.1, which they did not do).
