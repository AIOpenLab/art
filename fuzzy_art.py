"""
Fuzzy ART

An unsupervised clustering algorithm 
"""
from functools import partial

import numpy as np


l1_norm = partial(np.linalg.norm, ord=1, axis=-1)


class FuzzyART(object):
    def __init__(self, alpha=1.0, gamma=0.01, rho=0.5, complement_coding=True):
        """        
        :param alpha: learning rate [0,1] 
        :param gamma: regularization term >0
        :param rho: vigilance [0,1]
        :param complement_coding: use complement coding scheme for inputs
        """
        self.alpha = alpha  # learning rate
        self.beta = 1 - alpha
        self.gamma = gamma  # choice parameter
        self.rho = rho  # vigilance
        self.complement_coding = complement_coding

        self.w = None

    def _init_weights(self, x):
        self.w = np.atleast_2d(x)

    def _complement_code(self, x):
        if self.complement_coding:
            return np.hstack((x, 1-x))
        else:
            return x

    def _add_category(self, x):
        self.w = np.vstack((self.w, x))

    def _match_category(self, x):
        fuzzy_weights = np.minimum(x, self.w)
        fuzzy_norm = l1_norm(fuzzy_weights)
        scores = fuzzy_norm / (self.gamma + l1_norm(self.w))
        threshold = fuzzy_norm / l1_norm(x) >= self.rho
        if np.all(threshold == False):
            return -1
        else:
            return np.argmax(scores * threshold.astype(int))

    def train(self, x, epochs=1):
        """        
        :param x: 2d array of size (samples, features), where all features are
         in [0, 1]
        :param epochs: number of training epochs, the training samples are 
        shuffled after each epoch  
        :return: self
        """
        samples = self._complement_code(np.atleast_2d(x))

        if self.w is None:
            self._init_weights(samples[0])

        for epoch in range(epochs):
            for sample in np.random.permutation(samples):
                category = self._match_category(sample)
                if category == -1:
                    self._add_category(sample)
                else:
                    w = self.w[category]
                    self.w[category] = (self.alpha * np.minimum(sample, w) +
                                        self.beta * w)
        return self

    def test(self, x):
        """        
        :param x: 2d array of size (samples, features), where all features are
         in [0, 1] 
        :return: category IDs for each provided sample
        """
        samples = self._complement_code(np.atleast_2d(x))

        categories = np.zeros(len(samples))
        for i, sample in enumerate(samples):
            categories[i] = self._match_category(sample)
        return categories


if __name__ == '__main__':
    import sklearn.datasets as ds
    iris = ds.load_iris()
    data = iris['data'] / np.max(iris['data'], axis=0)
    net = FuzzyART(alpha=0.5, rho=0.75)
    net.train(data, epochs=100)
    print(net.w.shape)
    print(net.test(data).astype(int))
    print(iris['target'])
