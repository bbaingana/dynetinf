import numpy as np
import scipy as sp
import math

def softThresh(M, mu):
    P = np.asmatrix(np.zeros(M.shape))
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            P[i,j] = np.sign(M[i,j])*max(abs(M[i,j]) - mu, 0)
    return P

def fista(X, Ptau, Qtau, Ahat, Ahat_old, \
		bhat, bhat_old, forget_t, lamb_t, kmax):
    """ fista() tracks the sequence of unknown graphs captured by
    adjacency matrices, over which cascade data Yt propagate"""
    
    N = Qtau.shape[0]
    K = Qtau.shape[1]
    
    t_seq_old = 1
    t_seq = (1 + math.sqrt(1 + 4*(t_seq_old**2)))/2 

    # Compute Lipschitz constant
    M1 = np.hstack((Ptau, Qtau*X.T))
    M2 = np.hstack((X*Qtau.T, forget_t*X*X.T))
    M3 = np.vstack((M1, M2))
    L = maxEigVal(M3)

    result_dict = {}

    for k in range(kmax):
        for i in range(N):
            curr = [i]
            indices = list(set(range(N)).difference(set(curr))) 

	    # Variables using accelerating combination of last two iterates
	    b_ii = bhat[i, 0] + ((t_seq_old-1)/t_seq)*(bhat[i, 0] - bhat_old[i, 0])
	    a_i = Ahat[i, :] + ((t_seq_old-1)/t_seq)*(Ahat[i, :] - Ahat_old[i, :])
	    a_i_tilde = a_i[:, indices].T

	    # Auxiliary quantities
	    p_t = Ptau[:, i]
	    p_ti = p_t[indices, :]

	    q_t = Qtau[i, :]
	    P_ti = Ptau[indices, :]
	    P_ti = P_ti[:, indices]

	    Q_ti = Qtau[indices, :]
	    x_i = X[i, :].T


	    # Step 1: compute gradients

	    nablaf_ai = (-1.0)*(p_ti - P_ti*a_i_tilde - Q_ti*x_i*b_ii)

	    nablaf_bii = (-1.0)*(q_t*x_i - a_i_tilde.T*Q_ti*x_i - \
			    forget_t*b_ii*(np.linalg.norm(x_i)**2))

	    # Step 2: update B (gradient descent)
	    bhat_old[i, 0] = bhat[i, 0]

	    bhat[i, 0] = b_ii - (1.0/L)*nablaf_bii[0,0]

	    # Step 3: update A (gradient descent + soft-thresholding)

            a_i_tilde = softThresh(a_i_tilde-(1.0/L)*nablaf_ai, lamb_t/L)
	    Ahat_old[i, :] = Ahat[i, :]


	    Ahat[i, :] = np.hstack((a_i_tilde[0:i, :].T, \
			    np.asmatrix(np.zeros((1,1))), \
			    a_i_tilde[i:, :].T))
        t_seq_old = t_seq
	t_seq = (1 + math.sqrt(1 + 4*(t_seq_old**2)))/2
        
    result_dict['Ahat'] = Ahat
    result_dict['bhat'] = bhat
    
    return result_dict

def maxEigVal(M):
    eigvals,eigvecs = np.linalg.eig(M)
    return max(eigvals)
