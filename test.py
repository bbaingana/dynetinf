from netinf import netinf

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import networkx as nx
import pickle

if __name__=='__main__':
    # Unpickle the dataset
    dataset = pickle.load(open("dataset.p","rb"))
    gtruth = pickle.load(open("gtruth.p","rb")) # ground truth

    # Set algorithm parameters
    
    lamb = 100 #sparsity promoting parameter
    mu = 0.000001 # Learning rate for SGD algorithm
    

    theta = 0.001
    forget = 0.95 # Forgetting factor
    
    T = len(dataset)
    
    # Error vector
    err = []
    
    # Running total of forgetting factors
    forget_t = 0

    # Maximal number of iterations per time interval
    kmax = 4

    # data introspection
    inst0 = dataset[0]
    X0 = inst0['X']
    N = X0.shape[0]

    # Initialization
    bhat = np.asmatrix(np.ones((N, 1)))
    bhat_old = bhat

    Ahat = np.asmatrix(np.ones((N, N)))
    Ahat_old = Ahat


    print "\nLambda\t Beta\t Time interval\t Error\n"

    for t in range(T):
        instdata = dataset[t]
        instgtruth = gtruth[t]
        A = instgtruth['A']
        b = instgtruth['b']
        
        X = instdata['X']
        Yt = instdata['Y']
        
        Pt = Yt*Yt.T
        forget_t = 1 + forget*forget_t
        lamb_t = lamb

        # Recursive data updates
        if t == 0:
            Ptau = Pt
	    Qtau = Yt
        else:
            Ptau = forget*Ptau + Pt
	    Qtau = forget*Qtau + Yt

        
        result_dict = netinf.fista(X, Ptau, Qtau, Ahat, Ahat_old, \
		      bhat, bhat_old, forget_t, lamb_t, kmax)
            
        Ahat = result_dict['Ahat']
        bhat = result_dict['bhat']

	errt = np.linalg.norm(A-Ahat)/np.linalg.norm(A)
    
        print "{0}\t{1}\t{2}\t{3}".format(lamb_t, forget, t, errt)

        err.append(errt)

    # Plot errors and adj. matrices here
    plt.plot(range(T), err, 'r-')
    plt.title('Relative error plot')
    plt.ylabel('Relative Error')
    plt.xlabel('time slot')
    plt.show()
