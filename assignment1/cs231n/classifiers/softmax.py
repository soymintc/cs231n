from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def calc_sample_loss(W, x, yi, C, D):
    """Calculate loss output for W.T.dot(x), 
    given #classes, #dimensions and λR(W)"""
    s = np.zeros(C) # zero vector (C, 1)
    exp_s = np.zeros(C)
    exp_s_sum = 0 # sum of s
    probs = np.zeros(C) # s_class/s_sum (C, 1)
    logits = np.zeros(C) # (C, 1)
    imp = np.zeros(C) # impulse fn for true label
    imp[yi] = 1 # set only the true index label as 1    
    cross_ent = 0 # cross entropy 
    
    # Calc dot product
    for j in range(C): # j: class
        for k in range(D): # k: dimension
            s_product = W.T[j][k] * x[k] # W.T * x; each product
            s[j] += s_product

    # Numerical stabilization
    max_s = max(s) # save max value for numerical stabilization
    for j in range(C): # j: class
        shifted_s = s[j] - max_s
        exp_s[j] = np.exp(shifted_s) # save e^(shifted_s)
        exp_s_sum += exp_s[j]

    # Calc probabilities, logits, and cross-entropy
    for j in range(C):
        # 0 <= yi < C --> true label idx
        probs[j] = exp_s[j] / exp_s_sum # normalize
        logits[j] = -np.log(probs[j]) # calc logit
        cross_ent += imp[j] * logits[j] # sum(p * -log(q))
        
    return cross_ent # set cross-entropy as loss

    
def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Fix all to explicit loop forms
    N, D = X.shape # N: #samples, D: #dimensions
    D, C = W.shape # D: #dimensions, C: #classes
    h = 1e-10 # epsilon
    
    # Regularization
    reg_value = 0
    for i in range(D):
        for j in range(C):
            reg_value += reg * (W[i][j])**2 # save w_ij^2, normalized w/ λ
    
    # Loss evaluation
    n_computed = 0 # time saver for grad computation
    losses = np.zeros(N) # (N, 1) for loss for each sample
    for i in range(N):
        yi = y[i] # true label for sample i
        labels = np.zeros(C)
        labels[yi] = 1
        #x = X[i] # select a single sample w/ (D, 1) dimensions
        
        # Compute loss
        scores = W.T.dot(X[i, :]) # (C, 1) of dot product per class
        f_i = scores - max(scores) # for numerical stabilization, sample i
        sum_i = sum(np.exp(f_i)) # sum of f_i: shifted scores sum
        softmaxes = np.exp(f_i) / sum_i # stabilized softmax
        logits = -np.log(softmaxes)
        losses[i] = np.sum(logits.dot(labels)) # sum(q * -log(p)): cross entropy
        
        # Compute gradient
        for j in range(C): # j: class
            p = np.exp(f_i[j])/sum_i
            dL_div_df = p - labels[j] # dL/df = pj - d(j==yi)
            for k in range(D): # k: dimension
                # add up grads for this class, regarding this sample
                # add dL/df * df/dW, for (i=sample ; j=class, k=dimension)
                #     = sum([(p_class - y_class) * x_k for k in dimensions])
                # dW: DxC, X:NxD
                dW[k, j] += dL_div_df * X[i, k]
        
    loss = np.mean(losses) + reg_value # calc mean sample loss:  # loss(W): mean(logit) + λR(W)
    
    dW = dW/N + 2*reg*W # grad = dW/N + 2λW
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Fix all to explicit loop forms
    N, D = X.shape # N: #samples, D: #dimensions
    D, C = W.shape # D: #dimensions, C: #classes
    h = 1e-10 # epsilon
    
    # Regularization
    reg_value = 0
    for i in range(D):
        for j in range(C):
            reg_value += reg * (W[i][j])**2 # save w_ij^2, normalized w/ λ
    
    # Loss evaluation
    n_computed = 0 # time saver for grad computation
    losses = np.zeros(N) # (N, 1) for loss for each sample
    for i in range(N):
        yi = y[i] # true label for sample i
        labels = np.zeros(C)
        labels[yi] = 1
        #x = X[i] # select a single sample w/ (D, 1) dimensions
        
        # Compute loss
        scores = W.T.dot(X[i, :]) # (C, 1) of dot product per class
        f_i = scores - max(scores) # for numerical stabilization, sample i
        sum_i = sum(np.exp(f_i)) # sum of f_i: shifted scores sum
        softmaxes = np.exp(f_i) / sum_i # stabilized softmax
        logits = -np.log(softmaxes)
        
        losses[i] = np.sum(logits.dot(labels)) # sum(q * -log(p)): cross entropy
        
        # Compute gradient
        for j in range(C): # j: class
            p = np.exp(f_i[j])/sum_i
            dL_div_df = p - labels[j] # dL/df = pj - d(j==yi) [softmax(f)-level gradient]
            # add up grads for this class, regarding this sample
            # add dL/df * df/dW, for (i=sample ; j=class, k=dimension)
            #     = sum([(p_class - y_class) * x_k for k in dimensions])
            # dW: DxC, X:NxD
            dW[:, j] += dL_div_df * X[i, :]
        
    loss = np.mean(losses) + reg_value # loss(W): mean(sample losses) + λR(W)
    
    dW = dW/N + 2*reg*W # grad = dW/N + 2λW
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
