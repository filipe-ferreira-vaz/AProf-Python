#!/usr/bin/env python

# Deep Learning Homework 1

import argparse

import numpy as np
import matplotlib.pyplot as plt

import time
import utils


class LinearModel(object):
    def __init__(self, n_classes, n_features, **kwargs):
        self.W = np.zeros((n_classes, n_features))

    def update_weight(self, x_i, y_i, **kwargs):
        raise NotImplementedError

    def train_epoch(self, X, y, **kwargs):
        for x_i, y_i in zip(X, y):
            self.update_weight(x_i, y_i, **kwargs)

    def predict(self, X):
        """X (n_examples x n_features)"""
        scores = np.dot(self.W, X.T)  # (n_classes x n_examples)
        predicted_labels = scores.argmax(axis=0)  # (n_examples)
        return predicted_labels

    def evaluate(self, X, y):
        """
        X (n_examples x n_features)
        y (n_examples): gold labels
        """
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible


class Perceptron(LinearModel):
    def update_weight(self, x_i, y_i, **kwargs):

        """Initialize number of mistakes"""
        self.k = 0

        """Predict y-hat"""
        y_hat = np.argmax(np.dot(self.W, x_i))
        #print("self.W[y_i]", self.W[y_i])
        #print("x_i", x_i)
        #print("y-hat", self.W[y_i] @ x_i)
        #print("y-hat arg", y_hat)

        """Check error and update W and k"""
        if y_hat != y_i:
            self.W[y_i] += x_i
            self.W[y_hat] += -x_i
            self.k+=1

        return self.W



        """
        x_i (n_features): a single training example
        y_i (scalar): the gold label for that example
        other arguments are ignored
        """



        raise NotImplementedError # Q1.1 (a)


class LogisticRegression(LinearModel):
    def update_weight(self, x_i, y_i, learning_rate=0.001, l2_penalty=0.0, **kwargs):

        """Define conditional probability"""
        exponentials = np.exp(np.dot(self.W,x_i))
        # print("Shape of exponentials: ", exponentials.shape)
        Z_x = np.sum(exponentials)
        P_W = np.dot(exponentials,1/Z_x)
        # print("Shape of P_W: ", P_W.shape)
    

        """Predict label (y_hat) --- baseado no slide 8 da lecture 3"""
        # self.W = np.argmax(np.sum(np.log(P_W)))
        y_hat = np.argmax(np.dot(self.W,x_i))
        # print("y-hat: ",y_hat)
        # print("y: ",y_i)

        """Create one-hot vector for the predicted label"""
        n_labels = self.W.shape[0]
        e_y_hat = np.zeros(n_labels)
        e_y_hat[y_hat] = 1

        """Create one-hot vector for actual label"""
        e_y = np.zeros(n_labels) 
        e_y[y_i] = 1

        """Define cross product between e_y and x_i"""
        ey_cross_xi = np.outer(e_y,x_i)
        outer_product = np.outer(e_y_hat, x_i)

        """Declare prouct between P_W, e_y_hat and x_i; initialize to 0"""
        # PW_dot_eyhat_cross_xi = 0

        """Sum across all y_hat --- baseado no slide 21 da lecture 3"""
        # PW_dot_eyhat_cross_xi += np.dot(P_W[i],np.outer(e_y_hat,x_i))
        PW_dot_eyhat_cross_xi = np.sum(P_W[:, np.newaxis, np.newaxis] * outer_product, axis=0)

        """Update weights"""
        self.W += learning_rate*(ey_cross_xi - PW_dot_eyhat_cross_xi) - l2_penalty*learning_rate*self.W

        # print("Shape of e_y x x_i: ", ey_cross_xi.shape)
        # print("Shape of P_W * (e_y_hat x x_i): ", PW_dot_eyhat_cross_xi.shape)

        # self.W += learning_rate*np.outer((e_y-P_W),x_i) - l2_penalty*learning_rate*self.W

        # print("W: ", self.W)

        return self.W

        """
        x_i (n_features): a single training example
        y_i: the gold label for that example
        learning_rate (float): keep it at the default value for your plots
        """
        # raise NotImplementedError # Q1.2 (a,b)


class MLP(object):
    def __init__(self, n_classes, n_features, hidden_size):
        # Initialize an MLP with a single hidden layer.
        self.epsilon = 1e-6
        mu, sigma = 0.1, 0.1
        self.b_hidden = np.zeros(hidden_size)
        #print("b_hidden shape:", self.b_hidden.shape)
        self.W_hidden = np.random.normal(mu,sigma,(hidden_size,n_features))
        #print("W_hidden shape:", self.W_hidden.shape)
        self.b_output = np.zeros(n_classes)
        #print("b_output shape:", self.b_output.shape)
        self.W_output = np.random.normal(mu,sigma,(n_classes,hidden_size))
        #print("W_output shape:", self.W_output.shape)

        #print("MLP initialized")
        # raise NotImplementedError # Q1.3 (a) init
    
    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0).astype(float)

    def softmax(self, x):
        exps = np.exp(x - np.max(x)) # shift the values to prevent overflow
        return exps / np.sum(exps)

    def predict(self, X):
        # Compute the forward pass of the network. At prediction time, there is
        # no need to save the values of hidden nodes.

        #print("predict has been called")

        #print("Shape of X:", X.shape)

        z_hidden = np.dot(X, self.W_hidden.T) + self.b_hidden
        #print("Shape of z_hidden: ", z_hidden.shape)
        h_hidden = self.relu(z_hidden)
        z_output = np.dot(h_hidden, self.W_output.T) + self.b_output
        probs = self.softmax(z_output)
        result = np.argmax(probs)

        return result

    def evaluate(self, X, y):
        """
        X (n_examples x n_features)
        y (n_examples): gold labels
        """

        #print("evaluate has been called")

        # Identical to LinearModel.evaluate()
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible

    def train_epoch(self, X, y, learning_rate=0.001, **kwargs):
        """
        Dont forget to return the loss of the epoch.
        """

        #print("train_epoch has been called")

        # n_samples = X.shape[0]
        epoch_loss = 0
        for xi, yi in zip(X, y):
        #for i in range(n_samples):
        #    xi = X[i:i+1]
        #    yi = y[i:i+1]

            # Forward pass
            z_hidden = np.dot(xi, self.W_hidden.T) + self.b_hidden
            #print("z_hidden: ",z_hidden)
            h_hidden = self.relu(z_hidden)
            #print("h_hidden: ",h_hidden)
            output_input = np.dot(h_hidden, self.W_output.T) + self.b_output
            #print("output_input: ",output_input)
            probs = self.softmax(output_input)
            #print("output: ",output)

            #print("Shape of output: ", output.shape)

            #compute loss
            y_one_hot = np.zeros(self.W_output.shape[0])
            y_one_hot[yi] = 1
            # print("Shape of one hot: ", y_one_hot.shape)
            #loss = -np.log(np.dot(output, y_one_hot.T))
            #print("output[0,yi]: ", output[0,yi])
            loss = -np.log(probs[yi] + self.epsilon)
            epoch_loss += loss

            #Backward pass
            #Compute output gradient
            grad_z_output = (probs - y_one_hot)
            print("output error shape: ", grad_z_output.shape)
            print("hidden output shape: ",h_hidden.shape)
            #Compute gradients of hidden layer parameters
            print("graz_z_output shape: ", grad_z_output.shape)
            print("h_hidden shape: ", h_hidden.shape)
            grad_W_output = np.outer(grad_z_output,h_hidden)
            print("grad_W_output shape: ",grad_W_output.shape)
            grad_b_output = grad_z_output.squeeze()

            #Compute gradient of hidden layer below
            grad_h_hidden = self.W_output.T @ grad_z_output
            #print("hidden error shape: ", grad_h_hidden.shape)
            #Compute gradient of hidden layer below
            grad_z_hidden = np.multiply(grad_h_hidden,self.relu_derivative(z_hidden))

            #Compute gradients of hidden layer parameters
            grad_W_hidden = np.outer(grad_z_hidden,xi)
            grad_b_hidden = grad_z_hidden

            #print("grad_b_hidden shape: ", grad_b_hidden.shape)

            #Update the weights
            self.W_hidden -= learning_rate * grad_W_hidden
            self.b_hidden -= learning_rate * grad_b_hidden
            self.W_output -= learning_rate * grad_W_output
            self.b_output -= learning_rate * grad_b_output

            print("epoch_loss shape: ", epoch_loss.shape)
            print("epoch_loss value: ", epoch_loss)
        
        return epoch_loss
            
        raise NotImplementedError # Q1.3 (a) train_epoch


def plot(epochs, train_accs, val_accs, filename=None):
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.plot(epochs, train_accs, label='train')
    plt.plot(epochs, val_accs, label='validation')
    plt.legend()
    if filename:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()

def plot_loss(epochs, loss, filename=None):
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(epochs, loss, label='train')
    plt.legend()
    if filename:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()


def plot_w_norm(epochs, w_norms, filename=None):
    plt.xlabel('Epoch')
    plt.ylabel('W Norm')
    plt.plot(epochs, w_norms, label='train')
    plt.legend()
    if filename:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model',
                        choices=['perceptron', 'logistic_regression', 'mlp'],
                        help="Which model should the script run?")
    parser.add_argument('-epochs', default=20, type=int,
                        help="""Number of epochs to train for. You should not
                        need to change this value for your plots.""")
    parser.add_argument('-hidden_size', type=int, default=100,
                        help="""Number of units in hidden layers (needed only
                        for MLP, not perceptron or logistic regression)""")
    parser.add_argument('-learning_rate', type=float, default=0.001,
                        help="""Learning rate for parameter updates (needed for
                        logistic regression and MLP, but not perceptron)""")
    parser.add_argument('-l2_penalty', type=float, default=0.0,)
    parser.add_argument('-data_path', type=str, default='intel_landscapes.npz',)
    opt = parser.parse_args()

    utils.configure_seed(seed=42)

    add_bias = opt.model != "mlp"
    data = utils.load_dataset(data_path=opt.data_path, bias=add_bias)
    train_X, train_y = data["train"]
    dev_X, dev_y = data["dev"]
    test_X, test_y = data["test"]
    n_classes = np.unique(train_y).size
    n_feats = train_X.shape[1]

    # initialize the model
    if opt.model == 'perceptron':
        model = Perceptron(n_classes, n_feats)
    elif opt.model == 'logistic_regression':
        model = LogisticRegression(n_classes, n_feats)
    else:
        model = MLP(n_classes, n_feats, opt.hidden_size)
    epochs = np.arange(1, opt.epochs + 1)
    train_loss = []
    weight_norms = []
    valid_accs = []
    train_accs = []

    start = time.time()

    print('initial train acc: {:.4f} | initial val acc: {:.4f}'.format(
        model.evaluate(train_X, train_y), model.evaluate(dev_X, dev_y)
    ))
    
    for i in epochs:
        print('Training epoch {}'.format(i))
        train_order = np.random.permutation(train_X.shape[0])
        train_X = train_X[train_order]
        train_y = train_y[train_order]
        if opt.model == 'mlp':
            loss = model.train_epoch(
                train_X,
                train_y,
                learning_rate=opt.learning_rate
            )
        else:
            model.train_epoch(
                train_X,
                train_y,
                learning_rate=opt.learning_rate,
                l2_penalty=opt.l2_penalty,
            )
        
        train_accs.append(model.evaluate(train_X, train_y))
        valid_accs.append(model.evaluate(dev_X, dev_y))
        if opt.model == 'mlp':
            print('loss: {:.4f} | train acc: {:.4f} | val acc: {:.4f}'.format(
                loss, train_accs[-1], valid_accs[-1],
            ))
            train_loss.append(loss)
        elif opt.model == "logistic_regression":
            weight_norm = np.linalg.norm(model.W)
            print('train acc: {:.4f} | val acc: {:.4f} | W norm: {:.4f}'.format(
                 train_accs[-1], valid_accs[-1], weight_norm,
            ))
            weight_norms.append(weight_norm)
        else:
            print('train acc: {:.4f} | val acc: {:.4f}'.format(
                 train_accs[-1], valid_accs[-1],
            ))
    elapsed_time = time.time() - start
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    print('Training took {} minutes and {} seconds'.format(minutes, seconds))
    print('Final test acc: {:.4f}'.format(
        model.evaluate(test_X, test_y)
        ))

    # plot
    plot(epochs, train_accs, valid_accs, filename=f"Q1-{opt.model}-accs.pdf")
    if opt.model == 'mlp':
        plot_loss(epochs, train_loss, filename=f"Q1-{opt.model}-loss.pdf")
    elif opt.model == 'logistic_regression':
        plot_w_norm(epochs, weight_norms, filename=f"Q1-{opt.model}-w_norms.pdf")
    with open(f"Q1-{opt.model}-results.txt", "w") as f:
        f.write(f"Final test acc: {model.evaluate(test_X, test_y)}\n")
        f.write(f"Training time: {minutes} minutes and {seconds} seconds\n")


if __name__ == '__main__':
    main()
