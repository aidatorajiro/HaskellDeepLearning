import sys, os
import numpy as np
import pickle

with open("sample_weight.pkl", 'rb') as f:
    network = pickle.load(f)

for i in ['W1', 'W2', 'W3', 'b1', 'b2', 'b3']:
    np.savetxt("sample_weight_" + i + ".csv", network[i], delimiter=",")