# CIFAR-10 CNN in PyTorch

## Overview

This is a personal learning project built while studying convolutional neural
networks (CNNs) and PyTorch. It trains a small CNN to classify 32x32 color
images from CIFAR-10 into 10 classes.

The implementation is in [`cnn_cifar.ipynb`](cnn_cifar.ipynb). The notebook
loads and normalizes images, sends batches through the CNN, updates the model
during training, and measures accuracy on the test set.

## What I learned / Key concepts

- Loading an image dataset with `torchvision`
- Converting images to tensors and normalizing their three color channels
- Creating batches with PyTorch `DataLoader`
- Building a CNN with convolution, ReLU, and max-pooling layers
- Flattening convolutional features before fully connected layers
- Training with a loss function, backpropagation, and an optimizer
- Moving the model and data to an available device (`cuda` or `cpu`)
- Evaluating predictions without tracking gradients

## Model architecture

The model is defined by the `CNN` class in the notebook:

| Stage | Operation | Output shape for one image |
|---|---|---:|
| Input | CIFAR-10 RGB image | `3 x 32 x 32` |
| 1 | `Conv2d(3, 32, kernel_size=3, padding=1)` -> ReLU -> `MaxPool2d(2, 2)` | `32 x 16 x 16` |
| 2 | `Conv2d(32, 64, kernel_size=3, padding=1)` -> ReLU -> `MaxPool2d(2, 2)` | `64 x 8 x 8` |
| 3 | `Conv2d(64, 128, kernel_size=3, padding=1)` -> ReLU -> `MaxPool2d(2, 2)` | `128 x 4 x 4` |
| 4 | Flatten | `2,048` values |
| 5 | `Linear(4 x 4 x 128, 256)` -> ReLU | `256` |
| 6 | `Linear(256, 10)` | `10` class scores |

The final 10 values are raw class scores (logits), one for each CIFAR-10
class. During evaluation, the class with the largest score is selected.

## Dataset

CIFAR-10 contains 60,000 small color images divided into these classes:
airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.
The standard split has 50,000 training images and 10,000 test images.

The notebook uses `torchvision.datasets.CIFAR10` with `root="./"` and
`download=True`. When the dataset is missing, torchvision downloads and
extracts the CIFAR-10 archive into `./cifar-10-batches-py/`, including the
training batches and test batch. The directory is intentionally ignored by
Git, so it remains a local working copy rather than a committed project file.

Images are transformed with `ToTensor()` and then normalized per channel with
mean `(0.5, 0.5, 0.5)` and standard deviation `(0.5, 0.5, 0.5)`.

## Training and evaluation

- Training and test data use batches of `64`.
- Training batches are shuffled; test batches are not explicitly shuffled.
- The loss function is `CrossEntropyLoss`.
- The optimizer is Adam with its default settings in the notebook.
- The model trains for `10` epochs.
- Each epoch prints the average training loss across its batches.
- Evaluation switches the model to evaluation mode and uses `torch.no_grad()`.
  It counts correct predictions over the test set and reports accuracy.
- The code uses CUDA when available and otherwise uses the CPU.

The saved notebook output records training loss decreasing from `1.3691` after
epoch 1 to `0.1608` after epoch 10, followed by a test accuracy of `76.36%`.
These are the results already recorded in the notebook; no training duration
was recorded. The notebook does not create a separate validation split or save
a trained model checkpoint.

## Installation and setup

The project has no dependency manifest. The notebook imports PyTorch,
torchvision, and PyTorch's `DataLoader`, and it is run through Jupyter.

After cloning the repository, move into its directory and create a virtual
environment:

```bash
git clone https://github.com/<your-github-username>/<your-repository>.git
cd <your-repository>
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
python -m pip install torch torchvision jupyter
```

The current local repository does not have a configured GitHub remote, so the
clone URL above is a template. Replace the placeholders with the URL of the
GitHub repository you create.

## How to run

Start Jupyter from the project root:

```bash
jupyter notebook
```

Open `cnn_cifar.ipynb` and run the cells from top to bottom. On the first run,
`download=True` allows torchvision to download and prepare CIFAR-10 when
`./cifar-10-batches-py/` is not present. The notebook prints the selected
device, the average loss for each epoch, and the final test accuracy.

There is currently no standalone Python training script; the notebook is the
project's executable implementation.

## Current limitations and possible future improvements

This is a compact learning implementation rather than a production or
state-of-the-art image classifier. Possible next steps include:

- Trying different numbers of convolutional layers or channel sizes
- Experimenting with learning rates, batch sizes, and the number of epochs
- Adding data augmentation to the training transform
- Tracking training and test loss and accuracy over time
- Adding a validation split and per-class evaluation
- Saving and loading model checkpoints
