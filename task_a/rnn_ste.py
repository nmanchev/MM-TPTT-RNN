"""
Synthetic classification task (Task A)
Network trained using STE-estimated Back-propagation Through Time (STPTT)

(C) 2022 Nikolay Manchev
This work is licensed under a Creative Commons Attribution-NonCommercial 4.0 International License.

This code supplements the paper Manchev, N. and Spratling, M., "Learning Multi-Modal Recurrent Neural Networks with Target Propagation"

"""

import torch
import sys
import torch.nn as nn
import numpy as np
import time

import pandas as pd

from torch.nn.parameter import Parameter
from torch.autograd import Variable
from torch import optim

from collections import OrderedDict

from task_a import TaskAClass
import matplotlib.pyplot as plt

np.set_printoptions(precision=10, threshold=sys.maxsize, suppress=True)

class SRNN(object):       

    def __init__(self, X, y, X_test, y_test, seq_length, n_hid, init, hybrid, last_layer, batch_size, M, rng, noise=0):
        super(SRNN, self).__init__()

        self.n_inp = X.shape[2]  # [seq size n_inp]
        self.n_out = y.shape[1]  # [size n_out]

        self.M = M

        self.X = Variable(torch.from_numpy(X))
        self.y = Variable(torch.from_numpy(y))
        self.X_test = Variable(torch.from_numpy(X_test))
        self.y_test = Variable(torch.from_numpy(y_test))

        self.seq_length = seq_length
        self.n_hid = n_hid
        self.hybrid = hybrid

        self.noise = noise
        self.last_layer = last_layer
        self.batch_size = batch_size
        self.rng = rng

        self.h0 = torch.zeros(self.batch_size, self.n_hid)
        
        self.Wxh = Parameter(init(torch.empty(self.n_inp, self.n_hid)))
        self.Whh = Parameter(init(torch.empty(self.n_hid, self.n_hid)))
        self.Why = Parameter(init(torch.empty(self.n_hid, self.n_out)))
        self.bh = Parameter(torch.zeros(self.n_hid))
        self.by = Parameter(torch.zeros(self.n_out))
        
        self.params = OrderedDict()

        self.params["Wxh"] = self.Wxh
        self.params["Whh"] = self.Whh
        self.params["Why"] = self.Why
        self.params["bh"] = self.bh
        self.params["by"] = self.by

        self.activ = torch.sigmoid
        self.sftmx = nn.Softmax(dim=1)

    
    def _sample(self, x):
        rand = torch.rand(size=x.shape)
        if self.hybrid:            
            ret = x
            ret[0:x.shape[0]//2,:] = (rand[0:ret.shape[0]//2,:] < x[0:ret.shape[0]//2,:]).float()
        else:
            ret = (rand < x).type(torch.FloatTensor)
        return ret


    def _parameters(self):
        for key, value in self.params.items():
            yield value


    def _zero_grads(self):
        for p in self._parameters():
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()


    @staticmethod
    def _cross_entropy(y_hat, y):
        return torch.mean(torch.sum(-y * torch.log(y_hat), 1))


    @staticmethod
    def _mse(x, y):
        return torch.mean((x - y) ** 2)


    def _gaussian(self, x):
        return torch.randn(size=x.shape) * self.noise


    def _validate(self, x):
        n_val_samples = x.shape[1]
        h0 = torch.zeros(n_val_samples, self.n_hid)
        h = torch.empty(self.seq_length, n_val_samples, self.n_hid)
        h[0, :, :],_,_ = self._f(x[0, :, :], h0)
        for t in range(1, self.seq_length):
            h[t, :, :],_,_ = self._f(x[t, :, :], h[t - 1].clone())

        out = h[-1] @ self.Why + self.by

        if self.last_layer == "softmax":
            out = self.sftmx(out)
        elif self.last_layer != "linear":
            raise Exception("Unsupported classification type.")

        return out


    def run_validation(self, x, y, avg_probs_100=True):

        valid_cost = 0
        valid_err = 0
        
        if avg_probs_100:
            out = torch.stack([self._validate(x) for i in range(100)]).mean(axis=0)
        else:
            out = self._validate(x)

        if self.last_layer == "softmax":
            valid_cost += self._cross_entropy(out, y)
            y = torch.argmax(y, 1)
            y_hat = torch.argmax(out.data, 1)
            valid_err = (~torch.eq(y_hat, y)).float().mean()
        elif self.last_layer == "linear":
            valid_cost = self._mse(out, y).sum()
            valid_err = (((y - out) ** 2).sum(axis=1) > 0.04).float().mean()
        else:
            raise Exception("Unsupported classification type.")

        return valid_cost, valid_err


    def _f(self, x, hs):
        ha_t = hs @ self.Whh + x @ self.Wxh + self.bh
                
        h_t = self.activ(ha_t)
                
        hs_t = self._sample(h_t)

        return hs_t, ha_t, h_t


    def _hidden(self, x):
        
        ha = torch.empty(self.seq_length, self.batch_size, self.n_hid)
        hs = torch.empty(self.seq_length, self.batch_size, self.n_hid)
        h = torch.empty(self.seq_length, self.batch_size, self.n_hid)

        hs[0, :, :], ha[0, :, :], h[0, :, :] = self._f(x[0, :, :], self.h0)

        for t in range(1, self.seq_length):
            hs[t, :, :], ha[t, :, :], h[t, :, :] = self._f(x[t, :, :], hs[t - 1])
            
        return hs, ha, h


    def _grad_calc(self, Ht, Hstm1, Xt, dE_dha_tp1, W):
        dE_dha_t = (dE_dha_tp1 @ W.T) * Ht * (1 - Ht)
        g_b = dE_dha_t.sum(axis=0)
        g_W = Hstm1.T @ dE_dha_t
        g_X = Xt.T @ dE_dha_t
        return dE_dha_t, g_b, g_W, g_X

   
    def _step_ste(self, x, y, f_optimizer):

        with torch.no_grad():    
        
            out = torch.zeros(self.batch_size, self.n_out)
            
            for i in range(self.M):
                
                Hs, Ha, H = self._hidden(x)
                out_ = Hs[-1] @ self.Why + self.by
                out = out + out_
                
        
            out = out / self.M
            
            if self.last_layer == "softmax":
                out = self.sftmx(out)
                cost = self._cross_entropy(out, y)
            elif self.last_layer == "linear":
                cost = self._mse(out, y).sum()
            else:
                raise Exception("Unsupported classification type.")
            
            # Calculate gradients
            dE_dya = (out - y) / self.batch_size  # equivalent to T.grad(cost, ya)
            dE_dby = dE_dya.sum(axis=0)
            dE_dWhy = Hs[-1].T @ dE_dya

            gHa = torch.empty(self.seq_length, self.batch_size, self.n_hid) 
            gWhh = torch.empty(self.seq_length, self.n_hid, self.n_hid)
            gWxh = torch.empty(self.seq_length, self.n_inp, self.n_hid)
            gbh = torch.empty(self.seq_length, self.n_hid)
                    
            tmax = self.seq_length-1
            
            gHa[tmax], gbh[tmax], gWhh[tmax], gWxh[tmax] = self._grad_calc(H[-1], Hs[-2], x[-1], dE_dya, self.Why)        
                    
            for t in range(tmax-1, -1, -1):
                if t == 0:
                    gHa[t], gbh[t], gWhh[t], gWxh[t] = self._grad_calc(H[t], self.h0, x[t], gHa[t+1], self.Whh)
                else:
                    gHa[t], gbh[t], gWhh[t], gWxh[t] = self._grad_calc(H[t], Hs[t-1], x[t], gHa[t+1], self.Whh)

            gWhh = gWhh.sum(axis=0)
            gWxh = gWxh.sum(axis=0)
            gbh = gbh.sum(axis=0)


        f_optimizer.zero_grad()
        
        self.Whh.grad = gWhh
        self.Wxh.grad = gWxh
        self.Why.grad = dE_dWhy
        self.bh.grad = gbh
        self.by.grad = dE_dby
        
        f_optimizer.step()

        return cost


    def _get_minibatch(self, batch_idx):
        batch_start_idx = batch_idx * self.batch_size
        batch_end_idx = batch_start_idx + self.batch_size
        x = self.X[:, batch_start_idx:batch_end_idx, :]
        y = self.y[batch_start_idx:batch_end_idx, :]

        return x, y


    def fit(self, f_optimizer, maxiter, ilr=None, g_optimizer=None, check_interval=10):

        training = True
        epoch = 1
        best = 0

        n_batches = self.X.shape[1] // self.batch_size

        while training & (epoch <= maxiter):

            if epoch == 1:
                with torch.no_grad():
                    cost, best = self.run_validation(self.X_test, self.y_test)
                acc = 100 * (1 - best)
                print(("It:  0\t\t\tLoss: %.3f" + 20*"\t" + "Val.loss: %.2f\tHighest acc: %.2f") % (0, cost, acc))

            cost = 0

            for i in range(n_batches):
                x, y = self._get_minibatch(i)
                cost += self._step_ste(x, y, f_optimizer)
                if torch.isnan(cost):
                    print("Cost is NaN. Aborting....")
                    training = False
                    break

            cost = cost / n_batches
            
            if epoch % check_interval == 0:

                with torch.no_grad():
                    valid_cost, valid_err = self.run_validation(self.X_test, self.y_test)
                
                print_str = "It: {:10s}\tLoss: %.3f\t".format(str(epoch)) % cost

                whh_grad_np = self.Whh.detach().numpy()

                if np.isnan(whh_grad_np).any():
                    print_str += "ρ|Whh|: -----\t"
                else:
                    print_str += "ρ|Whh|: %.3f\t" % np.max(abs(np.linalg.eigvals(whh_grad_np)))

                dWhh = np.linalg.norm(self.Whh.grad.numpy())
                dWxh = np.linalg.norm(self.Wxh.grad.numpy())
                dWhy = np.linalg.norm(self.Why.grad.numpy())

                acc = 100 * (1 - valid_err)

                if acc > best:
                    best = acc

                print_str += "dWhh: %.5f\t dWxh: %.5f\t dWhy: %.5f\t" % (dWhh, dWxh, dWhy)
                print_str += "Acc: %.2f\tVal.loss: %.2f\tHighest acc: %.2f" % (acc, valid_cost, best)

                print(print_str)

                if valid_err < 0.0001:
                    print("PROBLEM SOLVED.")
                    training = False

            epoch += 1

        return best.item(), cost.item()


    def plot_classA(self, file_name, avg_probs_100=False):
        rng = np.random.RandomState(1234)
        task = TaskAClass(rng)
        x, y = task.generate(1000, self.seq_length)
        y = np.argmax(y, 1)

        x_test = Variable(torch.from_numpy(x))

        with torch.no_grad():

            if avg_probs_100:
                raise Exception("Not implemented")
            else:
                p = self._validate(x_test)
                y_hat = torch.argmax(p.data, 1)

        pred = np.stack([np.squeeze(x[0, :]), y_hat.data.numpy()], axis=1)

        distDF = pd.DataFrame(pred, columns=["X", "y_pred"]) \
            .groupby(["y_pred"], as_index=False) \
            .agg({"X": ["mean", "std"]})

        distDF.columns = ["y_pred", "X_mean", "X_std"]

        plt.clf()

        plt.scatter(x[0, :], y, alpha=0.4, c="b")
        plt.scatter(x[0, :], y_hat, alpha=0.4, c="r")

        # plot dist
        plt.plot((distDF["X_mean"], distDF["X_mean"]), (distDF["y_pred"], distDF["y_pred"] + 0.5), 'k-')

        plt.savefig(file_name)


def get_classA(T, n_samples=3000, n_test=100):
    rng = np.random.RandomState(1234)
    task = TaskAClass(rng)
    X, y = task.generate(n_samples, T)

    X_test, y_test = task.generate(n_test, T)

    return X, y, X_test, y_test


def get_opt(opt, params, lr):
    if opt == "SGD":
        optimizer = optim.SGD(params, lr=lr, momentum=0.0, nesterov=False)
    elif opt == "Nesterov":
        optimizer = optim.SGD(params, lr=lr, momentum=0.9, nesterov=True)
    elif opt == "RMS":
        optimizer = optim.RMSprop(params, lr=lr)
    elif opt == "Adam":
        optimizer = optim.Adam(params, lr=lr)
    elif opt == "Adagrad":
        optimizer = torch.optim.Adagrad(params, lr=lr)
    else:
        raise Exception("Unknown optimiser %s." % opt)

    return optimizer


def run_experiment(seed, init, hybrid, task_name, opt, seq, hidden, batch, maxiter, learning_rate, M, check_interval=10):
    torch.manual_seed(seed)
    model_rng = np.random.RandomState(seed)

    if task_name == "task_A":
        n_samples = 3000
        n_test = 100
        last_layer = "softmax"
        X, y, X_test, y_test = get_classA(seq, n_samples, n_test)  # X [n_batches, batch_size, n_inp]
    else:
        print("Unknown task %s. Aborting..." % task_name)
        return

    print("SRNN ST Network")
    print("--------------------")
    print("Hybrid     : %s" % hybrid)
    print("MCMC       : %i" % M)
    print("task name  : %s" % task_name)
    print("train size : %i" % (X.shape[1]))
    print("test size  : %i" % (X_test.shape[1]))
    print("batch size : %i" % batch)
    print("T          : %i" % seq)
    print("n_hid      : %i" % hidden)
    print("init       : %s" % init.__name__)
    print("maxiter    : %i" % maxiter)
    print("chk        : %i" % check_interval)
    print("--------------------")
    print("optimiser  : %s" % opt)
    print("type       : straight-through")
    print("lr         : %.5f" % learning_rate)
    print("--------------------")

    model = SRNN(X, y, X_test, y_test, seq, hidden, init, hybrid, last_layer, batch, M, model_rng)
    
    model_parameters = [model.Whh, model.bh, model.Wxh, model.Why, model.by]
    optimizer = get_opt(opt, model_parameters, learning_rate)
    
    val_acc, tr_cost = model.fit(optimizer, maxiter, check_interval=check_interval)
            
    file_name = "rnn_ste_" + "t"+ str(seq) + "_" + "_taskA_i" \
                + str(learning_rate) + "_"  + init.__name__ + opt.lower()

    model.plot_classA(file_name + ".png")

    return val_acc, tr_cost


def main():
    batch = 20
    hidden = 100
    maxiter = 1000
    learning_rate = 0.1
    M = 20

    seed = 1234

    init = nn.init.orthogonal_
    hybrid = True

    # Experiment 1 - shallow depth
    seq = 5

    run_experiment(seed, init, hybrid, "task_A", "Adagrad", seq, hidden, batch, maxiter, learning_rate, M, check_interval=100)

    # Experiment 2 - deeper network    
    seq = 30

    run_experiment(seed, init, hybrid, "task_A", "Adagrad", seq, hidden, batch, maxiter, learning_rate, M, check_interval=100)


if __name__ == '__main__':
    main()