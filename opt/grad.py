import numpy as np

def gradient_descent(x0, steps, grad):

    yield x0
    xi = x0
    for step in steps:
        xi = xi - step * grad(xi)
        yield xi

    return
