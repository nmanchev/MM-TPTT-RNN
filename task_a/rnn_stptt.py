"""
Synthetic classification task (Task A)
Network trained using Stochastic Target Propagation Through Time (STPTT)

(C) 2022 Nikolay Manchev
This work is licensed under a Creative Commons Attribution-NonCommercial 4.0 International License.

This code supplements the paper Manchev, N. and Spratling, M., "Learning Multi-Modal Recurrent Neural Networks with Target Propagation"

"""

import torch
import sys
import torch.nn as nn
import numpy as np

import pandas as pd

from torch.nn.parameter import Parameter
from torch.autograd import Variable
from torch import optim

from collections import OrderedDict

from task_a import TaskAClass
import matplotlib.pyplot as plt

np.set_printoptions(precision=10, threshold=sys.maxsize, suppress=True)

class SRNN(object):

    def __init__(self, X, y, X_test, y_test, seq_length, n_hid, init, stochastic, hybrid, last_layer,
                 noise, batch_size, M, rng):
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
        self.stochastic = stochastic
        self.hybrid = hybrid
        self.noise = noise
        self.last_layer = last_layer
        self.batch_size = batch_size
        self.rng = rng

        # assert seq_length >= 10, "seq_length must be at least 10"
        
        self.h0 = torch.zeros(self.batch_size, self.n_hid)

        self.Wxh = Parameter(init(torch.empty(self.n_inp, self.n_hid)))
        self.Whh = Parameter(init(torch.empty(self.n_hid, self.n_hid)))
        self.Why = Parameter(init(torch.empty(self.n_hid, self.n_out)))
        self.bh = Parameter(torch.zeros(self.n_hid))
        self.by = Parameter(torch.zeros(self.n_out))

        self.Vhh = Parameter(init(torch.empty(self.n_hid, self.n_hid)))
        self.ch = Parameter(torch.zeros(self.n_hid))

        self.activ = torch.sigmoid
        self.sftmx = nn.Softmax(dim=1)
        self.params = OrderedDict()

        self.params["Wxh"] = self.Wxh
        self.params["Whh"] = self.Whh
        self.params["Why"] = self.Why
        self.params["bh"] = self.bh
        self.params["by"] = self.by
        self.params["Vhh"] = self.Vhh
        self.params["ch"] = self.ch


    def _sample(self, x):
        rand = torch.rand(size=x.shape)
        if self.hybrid:            
            ret = x
            #ret[:,0:x.shape[1]//2] = (rand[:,0:ret.shape[1]//2] < x[:,0:ret.shape[1]//2]).float()
            ret[0:x.shape[0]//2,:] = (rand[0:ret.shape[0]//2,:] < x[0:ret.shape[0]//2,:]).float()
        else:
            ret = (rand < x).type(torch.FloatTensor)
        return ret


    def _f(self, x, hs):
        if self.stochastic:
            hs = self._sample(hs)
        return self.activ(hs @ self.Whh + x @ self.Wxh + self.bh)


    def _g(self, x, hs):
        return torch.tanh(hs @ self.Vhh + x @ self.Wxh + self.ch)


    def _hidden(self, x):
        h = torch.empty(self.seq_length, self.batch_size, self.n_hid)
        h[0, :, :] = self._f(x[0, :, :], self.h0)

        for t in range(1, self.seq_length):
            h[t, :, :] = self._f(x[t, :, :], h[t - 1].clone())
        return h


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


    def _get_targets(self, x, hs_tmax, h, cost, ilr):
        h_ = torch.zeros(self.seq_length, self.batch_size, self.n_hid)
        h_[-1, :, :] = hs_tmax - ilr * torch.autograd.grad(cost, hs_tmax, retain_graph=True)[0]
        h_[-1, :, :] = h[-1, :, :] - hs_tmax + h_[-1, :, :]

        for t in range(self.seq_length - 2, -1, -1):
            h_[t] = h[t] - self._g(x[t + 1, :, :], h[t + 1]) + self._g(x[t + 1, :, :], h_[t + 1].detach())

        return h_


    def _calc_g_grads(self, x, h):

        dVhh = torch.zeros((self.seq_length, self.n_hid, self.n_hid), requires_grad=False)
        dch = torch.zeros((self.seq_length, self.n_hid), requires_grad=False)

        for t in range(1, len(h)):
            dVhh[t], dch[t] = torch.autograd.grad(self._mse(self._g(x[t, :, :], h[t]), h[t - 1].detach()),
                                                  (self.Vhh, self.ch), retain_graph=True)

        self.Vhh.grad = dVhh.sum(0)
        self.ch.grad = dch.sum(0)


    def _calc_f_grads(self, x, h, h_, cost):
        dWhh = torch.zeros((self.seq_length, self.n_hid, self.n_hid), requires_grad=False)
        dWxh = torch.zeros((self.seq_length, self.n_inp, self.n_hid), requires_grad=False)
        dbh = torch.zeros((self.seq_length, self.n_hid), requires_grad=False)

        dWhy, dby = torch.autograd.grad(cost, (self.Why, self.by), retain_graph=True)

        dWhh[0], dWxh[0], dbh[0] = torch.autograd.grad(self._mse(h[0], h_[0]),
                                                       (self.Whh, self.Wxh, self.bh), retain_graph=True)

        for t in range(1, len(h)):
            dWhh[t], dWxh[t], dbh[t] = torch.autograd.grad(self._mse(h[t], h_[t].detach()),
                                                           (self.Whh, self.Wxh, self.bh), retain_graph=True)

        self.Whh.grad = dWhh.sum(0)
        self.Wxh.grad = dWxh.sum(0)
        self.bh.grad = dbh.sum(0)
        self.Why.grad = dWhy
        self.by.grad = dby.clone()


    def forward(self, x, y):
        h = self._hidden(x)
        hs_tmax = self._sample(h[-1]).clone().detach().requires_grad_(True)
        out = hs_tmax @ self.Why + self.by

        if self.last_layer == "softmax":
            out = self.sftmx(out)
        elif self.last_layer == "linear":
            pass
        else:
            raise Exception("Unsupported classification type.")

        return hs_tmax, h, out
        

    def _validate(self, x):
        n_val_samples = x.shape[1]
        h0 = torch.zeros(n_val_samples, self.n_hid)
        h = torch.empty(self.seq_length, n_val_samples, self.n_hid)
        h[0, :, :] = self._f(x[0, :, :], h0)
        for t in range(1, self.seq_length):
            h[t, :, :] = self._f(x[t, :, :], h[t - 1].clone())

        out = h[-1] @ self.Why + self.by

        if self.last_layer == "softmax":
            out = self.sftmx(out)
        elif self.last_layer != "linear":
            raise Exception("Unsupported classification type.")

        return out


    def run_validation(self, x, y, avg_probs_100=True):

        valid_cost = 0
        valid_err = 0

        # Stochastic validation
        if self.stochastic & avg_probs_100:
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


    def _step_g(self, x, y, g_optimizer):
        g_optimizer.zero_grad()

        h = self._hidden(x)

        # Corrupt targets with noise
        if self.noise != 0:
            h = self._hidden(x)
            h = h.detach() + self._gaussian(h)

        self._calc_g_grads(x, h)

        g_optimizer.step()


    def _step_f(self, ilr, x, y, f_optimizer):
        f_optimizer.zero_grad()

        out = torch.zeros(self.batch_size, self.n_out)
        for i in range(self.M):
            hs_tmax, h, out_ = self.forward(x, y)
            out = out + out_
        out = out / self.M

        if self.last_layer == "softmax":
            cost = self._cross_entropy(out, y)
        elif self.last_layer == "linear":
            cost = self._mse(out, y).sum()
        else:
            raise Exception("Unsupported classification type.")

        with torch.no_grad():
            h_ = self._get_targets(x, hs_tmax, h, cost, ilr)

        self._calc_f_grads(x, h, h_, cost)

        f_optimizer.step()

        return cost


    def fit(self, ilr, maxiter, g_optimizer, f_optimizer, check_interval=10):

        training = True
        epoch = 1
        best = 0

        n_batches = self.X.shape[1] // self.batch_size

        while training & (epoch <= maxiter):

            if epoch == 1:
                with torch.no_grad():
                    _, best = self.run_validation(self.X_test, self.y_test)
                acc = 100 * (1 - best)
                print("Epoch -- \t Cost -- \t Test Acc: %.2f \t Highest: %.2f" % (acc, acc))

            cost = 0

            # Inverse mappings
            for i in range(n_batches):
                batch_start_idx = i * self.batch_size
                batch_end_idx = batch_start_idx + self.batch_size
                x = self.X[:, batch_start_idx:batch_end_idx, :]
                y = self.y[batch_start_idx:batch_end_idx, :]

                self._step_g(x, y, g_optimizer)

            # Forward mappings
            for i in range(n_batches):
                batch_start_idx = i * self.batch_size
                batch_end_idx = batch_start_idx + self.batch_size
                x = self.X[:, batch_start_idx:batch_end_idx, :]
                y = self.y[batch_start_idx:batch_end_idx, :]

                cost += self._step_f(ilr, x, y, f_optimizer)
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
                vhh_grad_np = self.Vhh.detach().numpy()

                if np.isnan(whh_grad_np).any():
                    print_str += "ρ|Whh|: -----\t"
                else:
                    print_str += "ρ|Whh|: %.3f\t" % np.max(abs(np.linalg.eigvals(whh_grad_np)))

                if np.isnan(vhh_grad_np).any():
                    print_str += "ρ|Vhh|: -----\t"
                else:
                    print_str += "ρ|Vhh|: %.3f\t" % np.max(abs(np.linalg.eigvals(vhh_grad_np)))

                dWhh = np.linalg.norm(self.Whh.grad.numpy())
                dWxh = np.linalg.norm(self.Wxh.grad.numpy())
                dWhy = np.linalg.norm(self.Why.grad.numpy())

                acc = 100 * (1 - valid_err)

                if acc > best:
                    best = acc

                print_str += "dWhh: %.5f\t dWxh: %.5f\t dWhy: %.5f\t" % (dWhh, dWxh, dWhy)
                print_str += "Acc: %.2f\tVal.loss: %.2f\tHighest: %.2f" % (acc, valid_cost, best)

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

        distDF = pd.DataFrame(pred, columns=["X", "y_pred"])\
                   .groupby(["y_pred"], as_index=False)\
                   .agg({"X": ["mean", "std"]})

        distDF.columns = ["y_pred", "X_mean", "X_std"]

        plt.clf()

        plt.scatter(x[0, :], y, alpha=0.4, c="b")
        plt.scatter(x[0, :], y_hat, alpha=0.4, c="r")
        plt.plot((distDF["X_mean"], distDF["X_mean"]), (distDF["y_pred"], distDF["y_pred"]+0.5), 'k-')

        plt.savefig(file_name)        


def get_classA(T, n_samples=3000, n_test=100):
    rng = np.random.RandomState(1234)
    task = TaskAClass(rng)
    X, y = task.generate(n_samples, T)

    X_test, y_test = task.generate(n_test, T)

    return X, y, X_test, y_test


def run_experiment(seed, init, task_name, opt, seq, hidden, stochastic, hybrid, batch, maxiter, i_learning_rate,
                   f_learning_rate, g_learning_rate, noise, M, check_interval=10):
    
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

    model = SRNN(X, y, X_test, y_test, seq, hidden, init, stochastic, hybrid, last_layer, noise, batch, M, model_rng)

    model_g_parameters = [model.Vhh, model.ch]

    model_f_parameters = [model.Whh, model.bh, model.Wxh, model.Why, model.by]

    if opt == "SGD":
        g_optimizer = optim.SGD(model_g_parameters, lr=g_learning_rate, momentum=0.0, nesterov=False)
        f_optimizer = optim.SGD(model_f_parameters, lr=f_learning_rate, momentum=0.0, nesterov=False)
    elif opt == "Nesterov":
        g_optimizer = optim.SGD(model_g_parameters, lr=g_learning_rate, momentum=0.9, nesterov=True)
        f_optimizer = optim.SGD(model_f_parameters, lr=f_learning_rate, momentum=0.9, nesterov=True)
    elif opt == "RMS":
        g_optimizer = optim.RMSprop(model_g_parameters, lr=g_learning_rate)
        f_optimizer = optim.RMSprop(model_f_parameters, lr=f_learning_rate)
    elif opt == "Adam":
        g_optimizer = optim.Adam(model_g_parameters, lr=g_learning_rate)
        f_optimizer = optim.Adam(model_f_parameters, lr=f_learning_rate)
    elif opt == "Adagrad":
        g_optimizer = torch.optim.Adagrad(model_g_parameters, lr=g_learning_rate)
        f_optimizer = torch.optim.Adagrad(model_f_parameters, lr=f_learning_rate)
    else:
        print("Unknown optimiser %s. Aborting..." % opt)
        return


    print("SRNN TPTT Network")
    print("--------------------")
    print("stochastic : %s" % stochastic)
    if stochastic:
        print("MCMC       : %i" % M)
        print("Hybrid     : %s" % hybrid)
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
    print("optimiser : %s" % opt)
    print("ilr       : %.5f" % i_learning_rate)
    print("flr       : %.5f" % f_learning_rate)
    print("glr       : %.5f" % g_learning_rate)
    if noise != 0:
        print("noise     : %.5f" % noise)
    else:
        print("noise     : ---")
    print("--------------------")

    val_acc, tr_cost = model.fit(i_learning_rate, maxiter, g_optimizer, f_optimizer, check_interval)
    file_name = "rnn_stptt_" + "t" + str(seq) + "_taskA_i" \
                + str(i_learning_rate) + "_f" + str(f_learning_rate) + "_g" \
                + str(g_learning_rate) + "_" + init.__name__  + opt.lower()

    model.plot_classA(file_name + ".png")

    return val_acc, tr_cost


def main():
    batch = 20
    hidden = 100
    maxiter = 1000
    i_learning_rate = 0.1
    f_learning_rate = 0.1
    g_learning_rate = 0.00001
    noise = 0.0
    M = 20

    seed = 1234

    init = nn.init.orthogonal_

    sto = True
    hybrid = True

    # Experiment 1 - shallow depth
    seq = 5

    run_experiment(seed, init, "task_A", "Adagrad", seq, hidden, sto, hybrid, batch, maxiter,
                   i_learning_rate, f_learning_rate, g_learning_rate, noise, M, check_interval=100)

    # Experiment 2 - deeper network
    seq = 30

    run_experiment(seed, init, "task_A", "Adagrad", seq, hidden, sto, hybrid, batch, maxiter,
                   i_learning_rate, f_learning_rate, g_learning_rate, noise, M, check_interval=100)

if __name__ == '__main__':
    main()
