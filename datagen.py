import numpy as np
import scipy as sp
import scipy.io as sio
import pickle

dataset1 = pickle.load(open("dataset.p","rb"))
gtruth1 = pickle.load(open("gtruth.p","rb")) 

dataset = []
gtruth = []

for t in range(200):
    
    dataset.append(dataset1[t])
    gtruth.append(gtruth1[t])

pickle.dump(dataset, open("dataset.p","wb"))
pickle.dump(gtruth, open("gtruth.p","wb"))


