import numpy as np


def indices_of_k(arr, k):
    """	
	Args:
	    arr: (N,) numpy VECTOR of integers from 0 to 9
	    k: int, scalar between 0 to 9
	Return:
	    indices: (M,) numpy VECTOR of indices where the value is matches k
	
	Given an array of integer values, use np.where or np.argwhere to return
	an array of all of the indices where the value equals k.
	Hint: You may need to index into the output of np.argwhere.
	"""
    indices = np.where(arr == k)[0]
    return indices


def argmax_1d(arr):
    """	
	Args:
	    arr: (N,) numpy VECTOR of random numbers
	Return:
	    arg_max: int, scalar index of the largest number in the array
	
	Given an array of integer values, use np.argmax to return the index of
	the largest value in the array. If there are duplicate largest values, return the
	first index encountered
	"""
    max_arg = np.argmax(arr)
    return max_arg


def mean_rows(arr):
    """	
	Args:
	    arr: N x M numpy array of random numbers
	Return:
	    means: (N,) numpy VECTOR
	
	Given a two dimensional array, use np.mean and the axis parameter to calculate
	the mean of each row.
	"""
    means = np.mean(arr, axis=1)
    return means

def sum_squares(arr):
    """	
	Args:
	    arr: N x M numpy array of random numbers
	Return:
	    squared_sums: N x 1 numpy array (NOT vector)
	
	Given a two dimensional array, use np.square or elementwise squaring to square every
	value in the array. Then, use np.sum, the axis parameter, and the keepdims parameter to
	sum the columns in each row of the squared array and keep the output as a 2 dimensional array.
	
	Example:
	arr:
	[[1,1,1],
	 [2,2,2],
	 [3,3,3]]
	squared_sums:
	[[3],
	 [12],
	 [27]]
	"""
    squared = np.square(arr)
    squared_sums = np.sum(squared, axis=1, keepdims=True)
    return squared_sums

def fast_manhattan(x, y):
    """	
	Args:
	    x: N x D numpy array
	    y: M x D numpy array
	Return:
	    dist: N x M numpy array, where dist[i, j] is the Manhattan distance between
	    x[i, :] and y[j, :]
	"""
    x_exp = np.expand_dims(x, axis=1)
    y_exp = np.expand_dims(y, axis=0)
    dist = np.sum(np.abs(x_exp - y_exp), axis=2)
    return dist

def multiple_choice():
    """	
	Return:
	    choice: int
	    - return 0 if: `fast_manhattan` has lower space and time complexity.
	    - return 1 if: `slow_manhattan` has lower space complexity and the same time complexity.
	    - return 2 if: `fast_manhattan` has higher space complexity and lower time complexity.
	    - return 3 if: Both are about the same in space and time complexity.
	"""
    return 1
