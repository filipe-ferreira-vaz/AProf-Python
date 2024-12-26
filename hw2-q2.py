#!/usr/bin/env python

# Deep Learning Homework 2

import argparse

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
import numpy as np

import utils


class ConvBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=None,
            maxpool=True,
            batch_norm=True,
            dropout=0.0
        ):
        super().__init__()

        # Q2.1. Initialize convolution, maxpool, activation and dropout layers 
        self.conv = nn.Conv2d(in_channels,out_channels, kernel_size=3, stride=1, padding=1)
        self.batch_norm = nn.BatchNorm2d(out_channels) if batch_norm else None
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.dropout = nn.Dropout2d(p=0.1)
        
        # Q2.2 Initialize batchnorm layer 
        
    def forward(self, x):
        # input for convolution is [b, c, w, h]
        # Implement execution of layers in right order

        x = self.conv(x)
        if self.batch_norm:
            x = self.batch_norm(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.dropout(x)
        return x


class CNN(nn.Module):
    def __init__(self, dropout_prob, maxpool=True, batch_norm=True, conv_bias=True):
        super(CNN, self).__init__()
        channels = [3, 32, 64, 128]
        fc1_out_dim = 1024
        fc2_out_dim = 512
        self.maxpool = maxpool
        self.batch_norm = batch_norm
        self.n_classes = 6

        # Initialize convolutional blocks
        self.conv1 = ConvBlock(channels[0], channels[1], dropout_prob, batch_norm)
        self.conv2 = ConvBlock(channels[1], channels[2], dropout_prob, batch_norm)
        self.conv3 = ConvBlock(channels[2], channels[3], dropout_prob, batch_norm)

        self.flatten = nn.Flatten()
        self.averagepool = nn.AdaptiveAvgPool2d((1, 1)) if batch_norm else nn.Identity()
        
        # Initialize layers for the MLP block
        self.fc1 = nn.Linear(128 * 1 * 1, fc1_out_dim) if batch_norm else nn.Linear(128 * 6 * 6, fc1_out_dim) # need to automate this
        # Assume input image size is 48x48. After 3 conv blocks, the spatial size becomes 6x6

        self.relu1 = nn.ReLU()
        self.batch_norm1 = nn.BatchNorm1d(fc1_out_dim) if batch_norm else nn.Identity()
        self.dropout1 = nn.Dropout(p=0.1)

        self.fc2 = nn.Linear(1024, fc2_out_dim)
        self.relu2 = nn.ReLU()
        # self.dropout2 = nn.Dropout(p=dropout_prob)

        self.fc3 = nn.Linear(512, self.n_classes) # can we pass y as a paramter here?

        # For Q2.2 initalize batch normalization
        
        

    def forward(self, x):
        x = x.reshape(x.shape[0], 3, 48, -1) # input vector is reshaped to number of output channels,
                                             # output width and inferred output height

        # Implement execution of convolutional blocks 
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        # Flattent output of the last conv block
        x = self.averagepool(x) if self.batch_norm else x
        x = self.flatten(x)
        
        # Implement MLP part
        x = self.fc1(x)
        x = self.relu1(x)
        if self.batch_norm:
            x = self.batch_norm1(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.relu2(x)
        #x = self.dropout2(x)

        x = self.fc3(x)
        
        # For Q2.2 implement global averag pooling
        

        return F.log_softmax(x, dim=1)
 

def train_batch(X, y, model, optimizer, criterion, **kwargs):
    """
    X (n_examples x n_features)
    y (n_examples): gold labels
    model: a PyTorch defined model
    optimizer: optimizer used in gradient step
    criterion: loss function
    """
    optimizer.zero_grad()
    out = model(X, **kwargs)
    loss = criterion(out, y)
    loss.backward()
    optimizer.step()
    return loss.item()


def predict(model, X, return_scores=True):
    """X (n_examples x n_features)"""
    scores = model(X)  # (n_examples x n_classes)
    predicted_labels = scores.argmax(dim=-1)  # (n_examples)

    if return_scores:
        return predicted_labels, scores
    else:
        return predicted_labels


def evaluate(model, X, y, criterion=None):
    """
    X (n_examples x n_features)
    y (n_examples): gold labels
    """
    model.eval()
    with torch.no_grad():
        y_hat, scores = predict(model, X, return_scores=True)
        loss = criterion(scores, y)
        n_correct = (y == y_hat).sum().item()
        n_possible = float(y.shape[0])

    return n_correct / n_possible, loss


def plot(epochs, plottable, ylabel='', name=''):
    plt.figure()#plt.clf()
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.plot(epochs, plottable)
    plt.savefig('%s.pdf' % (name), bbox_inches='tight')


def get_number_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def plot_file_name_sufix(opt, exlude):
    """
    opt : options from argument parser
    exlude : set of variable names to exlude from the sufix (e.g. "device")

    """
    return '-'.join([str(value) for name, value in vars(opt).items() if name not in exlude])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-epochs', default=40, type=int,
                        help="""Number of epochs to train for. You should not
                        need to change this value for your plots.""")
    parser.add_argument('-batch_size', default=8, type=int,
                        help="Size of training batch.")
    parser.add_argument('-learning_rate', type=float, default=0.01,
                        help="""Learning rate for parameter updates""")
    parser.add_argument('-l2_decay', type=float, default=0)
    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-optimizer',
                        choices=['sgd', 'adam'], default='sgd')
    parser.add_argument('-no_maxpool', action='store_true')
    parser.add_argument('-no_batch_norm', action='store_true')
    parser.add_argument('-data_path', type=str, default='intel_landscapes.v2.npz',)
    parser.add_argument('-device', choices=['cpu', 'cuda', 'mps'], default='cpu')

    opt = parser.parse_args()

    # Setting seed for reproducibility
    utils.configure_seed(seed=42)

    # Load data
    data = utils.load_dataset(data_path=opt.data_path)
    dataset = utils.ClassificationDataset(data)
    train_dataloader = DataLoader(
        dataset, batch_size=opt.batch_size, shuffle=True)
    dev_X, dev_y = dataset.dev_X.to(opt.device), dataset.dev_y.to(opt.device)
    test_X, test_y = dataset.test_X.to(opt.device), dataset.test_y.to(opt.device)

    # initialize the model
    model = CNN(
        opt.dropout,
        maxpool=not opt.no_maxpool,
        batch_norm=not opt.no_batch_norm
    ).to(opt.device)

    # get an optimizer
    optims = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD}

    optim_cls = optims[opt.optimizer]
    optimizer = optim_cls(
        model.parameters(), lr=opt.learning_rate, weight_decay=opt.l2_decay
    )

    # get a loss criterion
    criterion = nn.NLLLoss()

    # training loop
    epochs = np.arange(1, opt.epochs + 1)
    train_mean_losses = []
    valid_accs = []
    train_losses = []
    for ii in epochs:
        print('\nTraining epoch {}'.format(ii))
        model.train()
        for X_batch, y_batch in train_dataloader:
            X_batch = X_batch.to(opt.device)
            y_batch = y_batch.to(opt.device)
            loss = train_batch(
                X_batch, y_batch, model, optimizer, criterion)
            train_losses.append(loss)

        mean_loss = torch.tensor(train_losses).mean().item()
        print('Training loss: %.4f' % (mean_loss))

        train_mean_losses.append(mean_loss)
        val_acc, val_loss = evaluate(model, dev_X, dev_y, criterion)
        valid_accs.append(val_acc)
        print("Valid loss: %.4f" % val_loss)
        print('Valid acc: %.4f' % val_acc)

    test_acc, _ = evaluate(model, test_X, test_y, criterion)
    test_acc_perc = test_acc * 100
    test_acc_str = '%.2f' % test_acc_perc
    print('Final Test acc: %.4f' % test_acc)
    # plot
    sufix = plot_file_name_sufix(opt, exlude={'data_path', 'device'})

    plot(epochs, train_mean_losses, ylabel='Loss', name='CNN-3-train-loss-{}-{}'.format(sufix, test_acc_str))
    plot(epochs, valid_accs, ylabel='Accuracy', name='CNN-3-valid-accuracy-{}-{}'.format(sufix, test_acc_str))

    print('Number of trainable parameters: ', get_number_trainable_params(model))

if __name__ == '__main__':
    main()
