import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader

from torchvision.datasets import CIFAR10
from torchvision.models import resnet18, resnet34, resnet50
import torchvision.transforms as transforms

import numpy as np
import time

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

###################### PARAMS ##############################

BACKBONE = "resnet18"
BATCH_SIZE = 1024
N_EPOCHS = 1000
N_CPU_WORKERS = 16
BASE_LR = 0.03 
WEIGHT_DECAY = 5e-4 
MOMENTUM = 0.9
PROJECTOR_HIDDEN_SIZE = 1024
CROP_LOW_SCALE = 0.2
NESTEROV = False
PRINT_EVERY_EPOCHS = 100
MODEL_FILENAME = "simclr-resnet.pt"

###################### DATA LOADER #########################

cifar10_train = CIFAR10(
    root=".", train=True, transform=transforms.ToTensor()
)

cifar10_test = CIFAR10(
    root=".", train=False, transform=transforms.ToTensor()
)

transforms_ssl = transforms.Compose(
    [
        transforms.RandomResizedCrop(size=32, scale=(CROP_LOW_SCALE, 1)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply(
            [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8
        ),
        transforms.RandomGrayscale(p=0.2),
    ]
)


class AugmentedDataset(Dataset):
    def __init__(self, dataset: Dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        item, label = self.dataset[i]

        return self.transform(item), self.transform(item), label


cifar10_loader_ssl = DataLoader(
    AugmentedDataset(cifar10_train, transforms_ssl),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=N_CPU_WORKERS,
)

###################### NETWORK ARCHITECTURE #########################

class ResNetwithProjector(nn.Module):
    def __init__(self, backbone_network):
        super().__init__()

        self.backbone = backbone_network(weights=None)
        self.backbone_output_dim = self.backbone.fc.in_features
        
        self.backbone.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.backbone.maxpool = nn.Identity()
        self.backbone.fc = nn.Identity()

        self.projector = nn.Sequential(
            nn.Linear(self.backbone_output_dim, PROJECTOR_HIDDEN_SIZE), 
            nn.ReLU(), 
            nn.Linear(PROJECTOR_HIDDEN_SIZE, 128),
        )

    def forward(self, x):
        h = self.backbone(x)
        z = self.projector(h)
        return h, z


def infoNCE(features, temperature=0.5):
    x = F.normalize(features)
    cos_xx = x @ x.T / temperature
    cos_xx.fill_diagonal_(float("-inf"))
    
    batch_size = cos_xx.size(0) // 2
    targets = torch.arange(batch_size * 2, dtype=int, device=cos_xx.device)
    targets[:batch_size] += batch_size
    targets[batch_size:] -= batch_size

    return F.cross_entropy(cos_xx, targets)

backbones = {
   "resnet18": resnet18,
   "resnet34": resnet34,
   "resnet50": resnet50,
}

model = ResNetwithProjector(backbones[BACKBONE])

optimizer = SGD(
    model.parameters(),
    lr=BASE_LR * BATCH_SIZE / 256,
    momentum=MOMENTUM,
    weight_decay=WEIGHT_DECAY,
    nesterov=NESTEROV,
)

scheduler = CosineAnnealingLR(optimizer, T_max=N_EPOCHS)

###################### TRAINING LOOP #########################

device = "cuda"

model.to(device)
model.train()
training_start_time = time.time()

for epoch in range(N_EPOCHS):
    epoch_loss = 0.0
    start_time = time.time()

    for batch_idx, batch in enumerate(cifar10_loader_ssl):
        view1, view2, _ = batch
        view1 = view1.to(device)
        view2 = view2.to(device)

        optimizer.zero_grad()

        _, z1 = model(view1)
        _, z2 = model(view2)
        loss = infoNCE(torch.cat((z1, z2)))
        epoch_loss += loss.item()

        loss.backward()
        optimizer.step()

    end_time = time.time()
    if (epoch + 1) % PRINT_EVERY_EPOCHS == 0:
        print(
            f"Epoch {epoch + 1}, "
            f"average loss {epoch_loss / len(cifar10_loader_ssl):.4f}, "
            f"{end_time - start_time:.1f} s",
            flush=True
        )

    scheduler.step()

training_end_time = time.time()
hours = (training_end_time - training_start_time) / 60 // 60
minutes = (training_end_time - training_start_time) / 60 % 60
average = (training_end_time - training_start_time) / N_EPOCHS
print(
    f"Total training length for {N_EPOCHS} epochs: {hours:.0f}h {minutes:.0f}min",
    f"({average:.1f} sec/epoch)",
    flush=True
)

torch.save(model.state_dict(), MODEL_FILENAME)

###################### FINAL EVALUATION #########################

# model = ResNetwithProjector(backbones[BACKBONE])
# model.to(device)
# model.load_state_dict(torch.load(MODEL_FILENAME, weights_only=True))

def dataset_to_X_y(dataset):
    X = []
    y = []
    Z = []

    for batch_idx, batch in enumerate(DataLoader(dataset, batch_size=1024)):
        images, labels = batch

        h, z = model(images.to(device))

        X.append(h.cpu().numpy())
        Z.append(z.cpu().numpy())
        y.append(labels)

    X = np.vstack(X)
    Z = np.vstack(Z)
    y = np.hstack(y)

    return X, y, Z


model.eval()
with torch.no_grad():
    X_train, y_train, Z_train = dataset_to_X_y(cifar10_train)
    X_test, y_test, Z_test = dataset_to_X_y(cifar10_test)

for k in [1, 5, 10]:
    for metric in ["euclidean", "cosine"]:
        knn = KNeighborsClassifier(n_neighbors=k, metric=metric)
        knn.fit(X_train, y_train)
        print(
            f"kNN accuracy ({metric}, k={k}): {knn.score(X_test, y_test):.4f}",
            flush=True,
        )

# These params give better results than defaults (ridge penalty and lbfgs solver),
# but still often give convergence warnings, even when increasing max_iter.
# Training on GPU as we do below is much faster and tends to give better results.
lin = LogisticRegression(penalty=None, solver="saga")
lin.fit(X_train, y_train)
print(f"Linear accuracy (sklearn): {lin.score(X_test, y_test)}", flush=True)

########### LINEAR EVALUATION ON PRECOMPUTED REPRESENTATIONS ##########

N_EPOCHS = 100
ADAM_LR = 0.1    # lr=0.01 requires n_epochs=500 to get similar results

X_train = torch.tensor(X_train, device=device)
X_test = torch.tensor(X_test, device=device)
y_train = torch.tensor(y_train, device=device)
y_test = torch.tensor(y_test, device=device)

classifier = nn.Linear(X_train.shape[1], 10)
classifier.to(device)
classifier.train()

optimizer = Adam(classifier.parameters(), lr=ADAM_LR)
scheduler = CosineAnnealingLR(optimizer, T_max=N_EPOCHS)

for epoch in range(N_EPOCHS):
    batches = torch.randperm(len(X_train)).view(-1, 1000)
    for idx in batches:
        optimizer.zero_grad()
        logits = classifier(X_train[idx])
        loss = F.cross_entropy(logits, y_train[idx])
        loss.backward()
        optimizer.step()
    scheduler.step()

classifier.eval()
with torch.no_grad():
    yhat = classifier(X_test)

acc = (yhat.argmax(axis=1) == y_test).cpu().numpy().mean()
print(f"Linear accuracy (Adam on precomputed representations): {acc}", flush=True)

############### LINEAR EVALUATION WITH AUGMENTATIONS ##################

transforms_classifier = transforms.Compose(
    [
        transforms.RandomResizedCrop(size=32, scale=(CROP_LOW_SCALE, 1)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
)

cifar10_train_classifier = CIFAR10(
    root=".", train=True, download=False, transform=transforms_classifier
)

cifar10_loader_classifier = DataLoader(
    cifar10_train_classifier,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=N_CPU_WORKERS,
)

classifier = nn.Linear(model.backbone_output_dim, 10)
model.backbone.requires_grad = False

optimizer = Adam(classifier.parameters(), lr=ADAM_LR)
scheduler = CosineAnnealingLR(optimizer, T_max=N_EPOCHS)

classifier.to(device)
classifier.train()
training_start_time = time.time()

for epoch in range(N_EPOCHS):
    epoch_loss = 0.0
    start_time = time.time()

    for batch_idx, batch in enumerate(cifar10_loader_classifier):
        view, y = batch

        optimizer.zero_grad()

        h, _ = model(view.to(device))
        logits = classifier(h)
        loss = F.cross_entropy(logits, y.to(device))
        epoch_loss += loss.item()

        loss.backward()
        optimizer.step()

    end_time = time.time()
    if (epoch + 1) % PRINT_EVERY_EPOCHS == 0:
        print(
            f"Epoch {epoch + 1}, "
            f"average loss {epoch_loss / len(cifar10_loader_classifier):.4f}, "
            f"{end_time - start_time:.1f} s",
            flush=True
        )

    scheduler.step()

training_end_time = time.time()
hours = (training_end_time - training_start_time) / 60 // 60
minutes = (training_end_time - training_start_time) / 60 % 60
print(
    f"Total classifier training length for {N_EPOCHS} epochs: {hours:.0f}h {minutes:.0f}min",
    flush=True
)

classifier.eval()
with torch.no_grad():
    yhat = []
    y = []

    for batch_idx, batch in enumerate(DataLoader(cifar10_test, batch_size=1024)):
        images, labels = batch

        h, _ = model(images.to(device))
        logits = classifier(h)

        yhat.append(logits.cpu().numpy())
        y.append(labels)

    yhat = np.vstack(yhat)
    y = np.hstack(y)

acc = (yhat.argmax(axis=1) == y).mean()
print(f"Linear accuracy (trained with augmentations): {acc}", flush=True)
