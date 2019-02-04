# Mkhanyisi Gamedze
# Stanford CS231n 
# Colby CS291

import numpy as np
from random import shuffle

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

	# compute the loss and the gradient (prefilled)
	num_classes = W.shape[1]
	num_train = X.shape[0]
	loss = 0.0
	for i in range(num_train):
		scores = X[i].dot(W) #	compute class scores
		correct_class_score = scores[y[i]] 
		num_misclassified = 0 # keep track of number of loss contributors
		for j in range(num_classes):
			if j == y[i]:
				continue
			margin = scores[j] - correct_class_score + 1 # note delta = 1
			if margin > 0:
				loss += margin
				# Gradient for incorrect class weight.
				dW[:, j] += X[i, :] #	 add picture pixel values to the weight matrix class values for its pixels
				num_misclassified +=1
		# Gradient for correct class weight.
		dW[:, y[i]] += (num_misclassified * -X[i]) # add pixel weight scores on class for correct class on image i

	# Right now the loss is a sum over all training examples, but we want it
	# to be an average instead so we divide by num_train.
	loss /= num_train
	dW /= num_train

	# Add regularization to the loss.
	loss += reg * np.sum(W * W)
	dW += 2 * reg * W	 
	#############################################################################
	# TODO:		-> (Task completed as instructed)																																		 #
	# Compute the gradient of the loss function and store it dW.								#
	# Rather that first computing the loss and then computing the derivative,		#
	# it may be simpler to compute the derivative at the same time that the			#
	# loss is being computed. As a result you may need to modify some of the		#
	# code above to compute the gradient.																				#
	#############################################################################

	return loss, dW


def svm_loss_vectorized(W, X, y, reg):
	"""
	Structured SVM loss function, vectorized implementation.

	Inputs and outputs are the same as svm_loss_naive.
	"""
	#############################################################################
	# TODO:																																			#
	# Implement a vectorized version of the structured SVM loss, storing the		#
	# result in loss.																														#
	#############################################################################
	
	loss = 0.0
	dW = np.zeros(W.shape) # initialize the gradient as zero
	num_train = X.shape[0]
	scores = np.dot(X, W) # all class scores for all images	 (shape = (N, C))
	
	# get all correct class scores
	correct_class_scores = np.choose(y, scores.T)	 # use y (correct classes indice) to select elements from scores.T
	correct_classes = [np.arange(y.shape[0]), y] # shape = (2, N)

	correct_class_scores = scores[tuple(correct_classes)] # selects by NxC pair of coordinates, result ->	 shape (N)
	correct_class_scores = correct_class_scores[:, np.newaxis] # shape (N, 1)
	
	margins = scores - correct_class_scores + 1 # resultant shape -> (N, C)
	
	margins[tuple(correct_classes)] = 0 # give all correct classes a score of zero
	
	losses = np.maximum(margins, 0) # do comparison and for each loss below zero, make it zero
	
	loss = np.sum(losses) / num_train
	loss += 0.5 * reg * np.sum(np.square(W))
	
	#################################################################
	# TODO:																																			#
	# Implement a vectorized version of the gradient for the structured SVM			#
	# loss, storing the result in dW.																						#
	#																																						#
	# Hint: Instead of computing the gradient from scratch, it may be easier		#
	# to reuse some of the intermediate values that you used to compute the			#
	# loss.																																			#
	#############################################################################	 
	
	# Put 1 in for each [training point, class] pair above the margin
	classes_beyond_margin = (margins > 0).astype(np.float64) # shape (N, C)
	
	# For the correct classes, replace the value with the count of classes above the margin
	num_classes_beyond_margin = np.sum((classes_beyond_margin), axis=1) # shape N
	classes_beyond_margin[tuple(correct_classes)] = -num_classes_beyond_margin
	
	# Dot-product each row with our X
	dW = X.T.dot(classes_beyond_margin)	 # shape (D,C)
	
	dW /= num_train
	dW += reg * W

	return loss, dW
