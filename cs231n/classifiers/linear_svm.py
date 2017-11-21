import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        #dW[:,j:j+1] += X[i]
        #print('dw:', j, dW[:,j:j+1].shape)
        #print('X[i]', i, X[i].shape)
        #print('np.reshape:X[i]', i, np.reshape(X[i],(3073, 1)).shape)
        dW[:,j:j+1] += np.reshape(X[i],(3073, 1))
        dW[:,y[i]:y[i]+1] -= np.reshape(X[i],(3073, 1))

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW += reg *2* W

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]
  scores = X.dot(W)
  correct_class_score = X.dot(W)[np.arange(num_train),y].reshape(num_train,1)
  loss = np.sum(np.maximum((scores+1-correct_class_score),0))
  loss -= num_train*1 # cancel the part of true class 
  loss /= num_train
  loss += reg * np.sum(W * W)
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  # crrsponds to: dW[:,j:j+1] += np.reshape(X[i],(3073, 1))
  # print('intermediate values at loss: ', np.maximum((scores+1-correct_class_score),0).shape)
  # print('intermediate values at loss: ', np.maximum((scores+1-correct_class_score),0))
  dW_compn_mask = np.maximum((scores+1-correct_class_score),0) # margin
  gt_zero_indices = dW_compn_mask > 0 # stackoverflow , 28430904 ,
  dW_compn_mask[gt_zero_indices] = 1
  #print('dW_compn_mask.shape, dW_compn_mask: ' , dW_compn_mask.shape, dW_compn_mask)
  dW_compn_count = np.sum(dW_compn_mask, axis=1) # Compute sum of each row;
  #print('dW_compn_count.shape, dW_compn_count: ' , dW_compn_count.shape, dW_compn_count)
  #(values,counts) = np.unique(dW_compn_count,return_counts=True)
  #print('values,counts',values,counts)
  targets = y.reshape(-1)
  one_hot_correct_class = np.eye(num_classes)[targets]
  dW_compn_mask -= one_hot_correct_class
  dW_compn_count = np.sum(dW_compn_mask, axis=1) # Compute sum of each row;
  #print('dW_compn_count.shape, dW_compn_count: ' , dW_compn_count.shape, dW_compn_count)
  #(values,counts) = np.unique(dW_compn_count,return_counts=True)
  #print('values,counts',values,counts)
  #print('one_hot_correct_class:',one_hot_correct_class)
  #one_hot_correct_class = (dW_compn_count.reshape(num_train,1).T).dot(one_hot_correct_class)
  #print('(dW_compn_count) shape:',dW_compn_count.shape)
  #print('dW_compn_count:',dW_compn_count)
  #print(one_hot_correct_class * dW_compn_count[:,None])
  dW_compn_mask += -1 * one_hot_correct_class * dW_compn_count[:,None]
  #print('dW_compn_mask:',dW_compn_mask)
  #dW_correct_class = X.T.dot(one_hot_correct_class)
  dW = X.T.dot(dW_compn_mask)
  #print('X.T.shape' , X.T.shape)
  #print('one_hot_correct_class.shape' , one_hot_correct_class.shape)
  #print('dW_correct_class.shape' , dW_correct_class.shape)
  # dW -= (num_classes-1)* np.sum(X, axis=0).reshape(W.shape[0],1).dot(counts.reshape(1,num_classes))
  # dW -= (num_classes)* dW_correct_class
  dW /= num_train
  dW += reg *2* W
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
