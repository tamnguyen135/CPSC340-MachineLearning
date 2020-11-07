import numpy as np

def example(x):
    return np.sum(x**2)


def example_grad(x):
    return 2*x

def foo(x):
    result = 1
    λ = 4 # this is here to make sure you're using Python 3
    for x_i in x:
        result += x_i**λ
    return result

def foo_grad(x):
	result = []
	for x_i in x:
		result = np.append(result, 4*(x_i)**3)
	return result

def bar(x):
    return np.prod(x)

def bar_grad(x):
	result = []
	y = []
	for i in range(0, len(x)):
		y = x
		y = np.delete(y, i)
		result = np.append(result, np.prod(y))
	return result

