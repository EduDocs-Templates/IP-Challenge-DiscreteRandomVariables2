import unittest
from os import path
import numpy as np
from numpy.random import binomial
from scipy.stats import binom
from sklearn.metrics import mean_squared_error
import pandas as pd

class Test1Challenge(unittest.TestCase):

  def test_file(self):
    self.assertTrue(path.isfile('ejpmf.csv'))
    self.assertTrue(path.isfile('epmf.csv'))
    self.assertTrue(path.isfile('epmf_conv.csv'))

  def test_values(self):
    n = 8
    p = 0.5
    q = 0.5

    jpmf = np.zeros([n+1, n+1])
    for idx in range(0, n+1):
        for idy in range(0,idx+1):
             jpmf[idx, idy] = binom.pmf(idx,n,p) * binom.pmf(idy,idx,q)

    pmf_X = np.sum(jpmf, axis=1)
    pmf_Y = np.sum(jpmf, axis=0)
    pmf_conv = np.convolve(pmf_X, pmf_Y)

    pmf = np.zeros([2*n+1])
    for idu in range(0,2*n+1):
        for idx in range(0,min(idu+1,n+1)):
            if (idu-idx) < n+1:
                pmf[idu] = pmf[idu] + jpmf[idx,idu-idx]

    ejpmf_df = pd.read_csv('ejpmf.csv', index_col=0)
    epmf_df = pd.read_csv('epmf.csv', index_col=0)
    epmf_conv = pd.read_csv('epmf_conv.csv', index_col=0)

    ejpmf = ejpmf_df.to_numpy()
    epmf = epmf_df['0'].to_numpy()
    epmf_conv = epmf_conv['0'].to_numpy()

    self.assertLess(mean_squared_error(ejpmf,jpmf), 0.01)
    self.assertLess(mean_squared_error(epmf,pmf), 0.01)
    self.assertLess(mean_squared_error(epmf_conv,pmf_conv), 0.1)

if __name__ == '__main__':
  unittest.main()
