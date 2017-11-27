import numpy as np
from random import shuffle
from past.builtins import xrange

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
  num_train = X.shape[0]
  x_dim = X.shape[1]
  #num_train = 2
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    #print("correct_c.s. " , correct_class_score )
    #print("scores")
    #print(scores)
    #print("scores -= np.max(scores)" , np.max(scores))
    max_scores_ind = np.argmax(scores)
    print("max_ind, max:" , max_scores_ind , scores[max_scores_ind])
    scores -= np.max(scores)
    #print(scores)
    #print("np.exp(scores)")
    #print(np.exp(scores))
    correct_class_score = scores[y[i]]
    #print("correct_c.s. " , correct_class_score )
    p = np.sum(np.exp(scores))
    #print(p , "# p = np.sum(np.exp(scores))")
    #print(- correct_class_score + np.log(p) , "# loss = - correct_class_score + np.log(p)")
    loss += - correct_class_score + np.log(p)
    dW[:,y[i]:y[i]+1] -= np.reshape(X[i],(x_dim, 1))
    max=max_scores_ind
    dW[:,y[max]:y[max]+1] -= - np.reshape(X[i],(x_dim, 1))
    #p = np.exp(correct_class_score) / np.sum(np.exp(scores))
    #print(p , "# p = np.exp(correct_class_score) / np.sum(np.exp(scores))")
    #print(-np.log(p) , "# -np.log(p)")
    #print("")
  pass
  loss /= num_train
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

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
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

