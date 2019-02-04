# Mkhanyisi Gamedze
# Stanford CS231n 
# Colby CS291

import numpy as np
from random import shuffle

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

	#############################################################################
	# TODO: Compute the softmax loss and its gradient using explicit loops.			#
	# Store the loss in loss and the gradient in dW. If you are not careful			#
	# here, it is easy to run into numeric instability. Don't forget the				#
	# regularization!																														#
	#############################################################################
	
	# initialization should be similar to linear SVM. Only difference is on treatment of scores
	dW = np.zeros(W.shape) # initialize the gradient as zero 
	# compute the loss and the gradient (prefilled)
	num_classes = W.shape[1]
	num_train = X.shape[0]
	loss = 0.0
	
	# compute score matrix just once, to get all scores, instead of computing scores for every X[i] element
	scores = X.dot(W) #	compute class scores # shape (N,C)
	#scores = np.dot(X, W)
	
	# compute loss and gradients for each x[i]
	for xi in range(num_train):
		current_scores = scores[xi, :]

		# Fix for numerical stability by subtracting max from score vector (ensure all score values below 0 now)
		shift_scores = current_scores - np.max(current_scores)

		# cross entropy loss formula from notes (Loss_xi = -f_yi + class_sum(exp(f_j)))
		loss_xi = -shift_scores[y[xi]] + np.log(np.sum(np.exp(shift_scores)))
		loss += loss_xi  # cross entropy loss added to total loss for all images

		for j in range(num_classes):
			softmax_score = np.exp(shift_scores[j]) / np.sum(np.exp(shift_scores))

			# Gradient calculation.
			if j == y[xi]:
				dW[:, j] += (-1 + softmax_score) * X[xi]
			else:
				dW[:, j] += softmax_score * X[xi]
	
	
	#loss = loss / num_train + 0.5*reg*np.sum(W * W)
	#dW = dW / num_train + reg*W
	
	# Right now the loss is a sum over all training examples, but we want it
	# to be an average instead so we divide by num_train.
	loss /= num_train
	dW /= num_train

	# Add regularization to the loss.
	loss += 0.5*reg * np.sum(W * W)
	dW +=  reg * W	 
	
	return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
	"""
	Softmax loss function, vectorized version.

	Inputs and outputs are the same as softmax_loss_naive.
	"""
	#############################################################################
	# TODO: Compute the softmax loss and its gradient using no explicit loops.	#
	# Store the loss in loss and the gradient in dW. If you are not careful			#
	# here, it is easy to run into numeric instability. Don't forget the				#
	# regularization!																														#
	#############################################################################
	
	# Initialize the loss and gradient to zero.
	total_loss = 0.0
	loss = 0.0
	dW = np.zeros_like(W)
	
	# initialization should be similar to linear SVM. Only difference is on treatment of scores
	dW = np.zeros(W.shape) # initialize the gradient as zero 
	# compute the loss and the gradient (prefilled)
	num_classes = W.shape[1]
	num_train = X.shape[0]
	loss = 0.0
	
	# compute score matrix just once, to get all scores, instead of computing scores for every X[i] element
	scores = X.dot(W) #	compute class scores # shape (N,C)
	#scores = np.dot(X, W)
	
	# Fix for numerical stability by subtracting max from score vector (ensure all score values below 0 now for each row)
	shift_scores = scores - np.max(scores, axis=1).reshape(-1, 1)
	
	# xi losses
	s = np.exp(shift_scores) / (np.sum(np.exp(shift_scores), axis=1).reshape(-1, 1)) # 
	total_loss = -np.sum(np.log(s[range(num_train), y]))
	loss = total_loss/num_train + 0.5*reg*np.sum(W * W) # average all losses and regularize them
	
	# compute gradient
	p = np.exp(shift_scores)/np.sum(np.exp(shift_scores), axis=1).reshape(-1, 1) # probabilities for correct score
	indices = np.zeros(p.shape)
	indices[range(num_train), y] = 1 # place 1 in indices with correct ground class label
	dW = np.dot(X.T, p - indices) # shape (D,C)
	dW = dW/num_train + reg*W
	
	return loss, dW

