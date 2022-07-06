"""
Synthetic classification task (Task A)
Data generator class

(C) 2022 Nikolay Manchev
This work is licensed under a Creative Commons Attribution-NonCommercial 4.0 International License.

This code supplements the paper Manchev, N. and Spratling, M., "Learning Multi-Modal Recurrent Neural Networks with Target Propagation"

"""

import numpy as np

import matplotlib.pyplot as plt

class TaskAClass(object):
    def __init__(self, rng, floatX="float32"):
        self.rng = rng
        self.floatX = floatX
        self.classifType = "lastSoftmax"
        self.nin = 1
        self.nout = 10

        
    def datagen(self, size):
        x = np.linspace(0, 1, size).reshape((size,1))
        e = self.rng.uniform(-0.1, 0.1, (size,1))
        y = x + 0.3 * np.sin(2*np.pi*x) + e
        c = (np.round(x,1) * 9).astype(int)
        c = np.eye(self.nout)[c]
        return y,c    


    def generate(self, batchsize, length):
        rand_vals = self.rng.uniform(0, 1, size=(length-1, batchsize))
        x, y = self.datagen(batchsize)

        x = x.reshape(self.nin, batchsize)
        x = np.append(x, rand_vals, axis=0).reshape(length, batchsize, self.nin).astype(self.floatX)
        y = y.reshape(batchsize, self.nout).astype(self.floatX)
      
        shuffler = self.rng.permutation(batchsize)
                
        x_shuffled = np.take(x, shuffler, axis=1)
        y_shuffled = np.take(y, shuffler, axis=0)
            
        return x_shuffled, y_shuffled


    def generate_old(self, batchsize, length):
        rand_vals = np.zeros(shape=(length-1, batchsize))
        x, y = self.datagen(batchsize)
        x = x.reshape(self.nin, batchsize)
        seq = np.append(x, rand_vals, axis=0)

        shuffler = self.rng.permutation(batchsize, )
        x_shuffled = seq.reshape(length, batchsize, self.nin).astype(self.floatX)[:,shuffler,:]
        y_shuffled = y.reshape(batchsize, self.nout)[shuffler]
        return x_shuffled, y_shuffled


if __name__ == '__main__':
    print("Testing Task A generator ..")
    task = TaskAClass(np.random.RandomState(1234))
    seq, targ = task.generate(3, 1) # [length batch_size n_inp]

    print("Seq 0")
    print(seq[:, 0, :])
    print("Targ 0", targ[0])
    print()
    print("Seq 1")
    print(seq[:, 1, :])
    print("Targ 1", targ[1])
    print()
    print("Seq 2")
    print(seq[:, 2, :])
    print("Targ2", targ[2])
