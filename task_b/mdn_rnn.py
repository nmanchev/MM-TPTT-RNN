"""
Synthetic regression task using a Mixture Density Recurrent Neural Network
Network trained using Backpropagation Through Time (BPTT)

(C) 2022 Nikolay Manchev
This work is licensed under a Creative Commons Attribution-NonCommercial 4.0 International License.

This code supplements the paper Manchev, N. and Spratling, M., "Learning Multi-Modal Recurrent Neural Networks with Target Propagation"
"""

import numpy as np

import torch
import torch.nn as nn
import sys
import matplotlib.pyplot as plt

from torch.nn.parameter import Parameter
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

from task_b import TaskBClass


def get_classB(T, n_samples=3000, n_test=100):
    rng = np.random.RandomState(1234)
    task = TaskBClass(rng)
    X, y = task.generate(n_samples, T)

    X_test, y_test = task.generate(n_test, T)

    return X, y, X_test, y_test


class SRNN(object):

    def __init__(self, X, y, X_test, y_test, seq_length, n_hid, k, init, 
                 noise, batch_size, rng, dev):
        super(SRNN, self).__init__()
        
        self.dev = dev

        self.n_inp = X.shape[2]  # [seq size n_inp]
        self.n_out = y.shape[1]  # [size n_out]

        self.X = Variable(torch.from_numpy(X).to(dev))
        self.y = Variable(torch.from_numpy(y).to(dev))
        self.X_test = Variable(torch.from_numpy(X_test).to(dev))
        self.y_test = Variable(torch.from_numpy(y_test).to(dev))

        self.seq_length = seq_length
        self.n_hid = n_hid
        self.k = k
        self.noise = noise
        self.batch_size = batch_size
        self.rng = rng

        self.h0 = torch.zeros(self.batch_size, self.n_hid)
        
        self.Wxh = Parameter(init(torch.empty(self.n_inp, n_hid, device=dev)))
        self.Whh = Parameter(init(torch.empty(self.n_hid, n_hid, device=dev)))
        self.bh = Parameter(torch.zeros(n_hid, device=dev))
 
        self.mu = Parameter(init(torch.empty(n_hid, self.n_inp*k, device=dev)))
        self.pi = Parameter(init(torch.empty(n_hid, self.n_inp*k, device=dev)))
        self.var = Parameter(init(torch.empty(n_hid, self.n_inp*k, device=dev)))
    
        self.b_mu = Parameter(torch.zeros(self.n_inp*k, device=dev))
        self.b_pi = Parameter(torch.zeros(self.n_inp*k, device=dev))
        self.b_var = Parameter(torch.zeros(self.n_inp*k, device=dev))

        self.params = [self.Wxh, self.Whh, self.bh, self.mu, 
                       self.pi, self.var, self.b_mu, self.b_pi, self.b_var]
        
    
    def _f(self, x, h):        
        return torch.tanh(h @ self.Whh + x @ self.Wxh + self.bh)

    
    def get_mixture_coef(self, y):
        
        pi = y @ self.pi + self.b_pi
        mu = y @ self.mu + self.b_mu
        var = y @ self.var + self.b_var
            
        pi = F.softmax(pi, 2)
        var = torch.exp(var)

        return pi, mu, var


    def _hidden(self, x):
        
        seq_length = x.shape[0]
        batch_size = x.shape[1]
        
        h0 = torch.zeros(batch_size, self.n_hid, device=self.dev)
        h = torch.empty(seq_length, batch_size, self.n_hid, device=self.dev)

        h[0, :, :] = self._f(x[0, :, :], h0)

        for t in range(1, self.seq_length):
            h[t, :, :] = self._f(x[t, :, :], h[t - 1].clone())
        return h


    def forward(self, x):

        h = self._hidden(x)
                        
        y = h[-1].unsqueeze(axis=0)        
        
        pi, mu, var = self.get_mixture_coef(y)
        
        return (pi, mu, var), h

    
    def calc_pdf(self, y, mu, var):
        sigma = torch.sqrt(var)        
        m = torch.distributions.Normal(loc=mu, scale=sigma)
        value = torch.exp(m.log_prob(y))        
        return value
    
    
    def calc_loss(self, out, pi):
        prob_density = torch.sum(torch.mul(out.squeeze(axis=0), pi.squeeze(axis=0)), dim=1, keepdim=True)
        loss = -torch.log(prob_density).mean()
        return loss
    
    
    def calc_grads(self, loss):
      for p in self.params:
          if p.requires_grad:
              p.grad = torch.autograd.grad(loss, p, retain_graph=True)[0].clone()


    def sample(self, pi, mu, var, samples=10):
        np.random.seed(1234)
        batch_size, k = pi.shape
        sigma = np.sqrt(var)
        out = np.zeros((batch_size, samples, self.n_inp))
        
        for i in range(batch_size):
            for j in range(samples):            
                idx = np.random.choice(range(k), p=pi[i])
                for li in range(self.n_inp):
                    out[i,j,li] = np.random.normal(mu[i, idx*(li + self.n_inp)], sigma[i, idx])
        return out


    def plot_sample(self, file_name):
    
        with torch.no_grad():
    
            (pi, mu, var),_ = self.forward(self.X_test)
    
            if torch.isnan(pi).any() or torch.isnan(mu).any() or torch.isnan(var).any():
                print("Probabilities contain NaN. Not plotting.")
                return
                    
            if self.dev != "cpu":
                pi = pi.cpu()
                mu = mu.cpu()
                var = var.cpu()
                X_test = self.X_test.cpu()
            else:
                X_test = self.X_test
            
            pi_vals = pi.data.numpy().squeeze(axis=0)
            mu_vals = mu.data.numpy().squeeze(axis=0)
            var_vals = var.data.numpy().squeeze(axis=0)
                        
            preds = self.sample(pi_vals, mu_vals, var_vals, 5)
            
            plt.clf()

            for i in range(preds.shape[1]):
                plt.plot(X_test[0,:,:], preds[:, i], 'g.', alpha=0.3, label='predicted')
    
            plt.gca().set_ylim([-0.1,1.1])
    
            if file_name == None:    
                plt.show()
            else:
                plt.savefig(file_name)


    def fit(self, opt, maxiter, check_interval=100):
        
        training = True
        epoch = 1
        lowest_loss = sys.float_info.max
    
        while training & (epoch <= maxiter):
                       
            if epoch == 1:
                with torch.no_grad():
                    (pi, mu, var),_ = self.forward(self.X)                    
                    out = self.calc_pdf(self.y, mu, var)        
                    loss = self.calc_loss(out, pi)                    
                    lowest_loss = loss
                    print("Epoch: --- \t Loss: {:.4f} \t Lowest: {:.4f}".format(loss, loss))

            opt.zero_grad()
            
            (pi, mu, var),_ = self.forward(self.X)
            
            out = self.calc_pdf(self.y, mu, var)
            loss = self.calc_loss(out, pi)
            
            self.calc_grads(loss)
            
            opt.step()
            
            if epoch % check_interval == 0:
                if loss < lowest_loss:
                    lowest_loss = loss
                print("Epoch: {} \t Loss: {:.4f} \t Lowest: {:.4f}".format(epoch, loss, lowest_loss))
            

            epoch += 1

        return lowest_loss.data.cpu().item()


def run_experiment(seed, init, task_name, opt, seq, hidden, maxiter, i_learning_rate,
                   noise, k=3, check_interval=100):
    
    dev = "cpu"
    torch.manual_seed(seed)
    model_rng = np.random.RandomState(seed)

    if task_name == "task_B":
        n_samples = 3000
        n_test = 1000
        X, y, X_test, y_test = get_classB(seq, n_samples, n_test)  # X [seq, size, n_inp]
    else:
        print("Unknown task %s. Aborting..." % task_name)
        return
    
    batch_size = n_samples

    model = SRNN(X, y, X_test, y_test, seq, hidden, k, init, noise, batch_size, model_rng, dev)

    if opt == "SGD":
        optm = optim.SGD(model.params, lr=i_learning_rate, momentum=0.0, nesterov=False)
    elif opt == "Nesterov":
        optm = optim.SGD(model.params, lr=i_learning_rate, momentum=0.9, nesterov=True)
    elif opt == "RMS":
        optm = optim.RMSprop(model.params, lr=i_learning_rate)
    elif opt == "Adam":
        optm = optim.Adam(model.params, lr=i_learning_rate)
    elif opt == "Adagrad":
        optm = torch.optim.Adagrad(model.params, lr=i_learning_rate)
    else:
        print("Unknown optimiser %s. Aborting..." % opt)
        return

    print("SRNN MDN-BPTT Network")
    print("---------------------")
    print("k          : %s" % k)
    print("task name  : %s" % task_name)
    print("train size : %i" % (X.shape[1]))
    print("test size  : %i" % (X_test.shape[1]))
    print("batch size : %i" % batch_size)
    print("T          : %i" % seq)
    print("n_hid      : %i" % hidden)
    print("init       : %s" % init.__name__)
    print("maxiter    : %i" % maxiter)
    print("chk        : %i" % check_interval)
    print("--------------------")
    print("optimiser : %s" % opt)
    print("lr        : %.5f" % i_learning_rate)
    
    if noise != 0:
        print("noise     : %.5f" % noise)
    else:
        print("noise     : ---")
    print("--------------------")

    tr_loss = model.fit(optm, maxiter, check_interval)
    
    file_name = "rnn_bptt_mdn_t_" + str(seq) + "_taskB_i" + str(i_learning_rate) + "_" + init.__name__ + opt.lower() + ".png"

    model.plot_sample(file_name )

    return tr_loss


if __name__ == '__main__':    

    torch.autograd.set_detect_anomaly(False)
    seed = 1

    n_hid = 50
    k = 3
    init = nn.init.orthogonal_
    noise = 0
    lr = 0.01
    maxiter = 4000
    opt = "Adagrad"

    # Experiment 1 - shallow depth
    seq = 5
        
    run_experiment(seed, init, "task_B", opt, seq, n_hid, maxiter, lr, noise, check_interval=100)
    
    # Experiment 2 - deeper network    
    seq = 30
    
    run_experiment(seed, init, "task_B", opt, seq, n_hid, maxiter, lr, noise, check_interval=100)
    
    

