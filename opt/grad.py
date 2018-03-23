
def gradient_descent(x0, step, grad):
    """ Vanilla gradient descent

    Params
    ------

    x0 : initial guess

    step : step size (ie learning rate)

    grad : gradient function

    Yields
    ------

    xi : next step along gradient direction
    """

    yield x0

    xi = x0
    while True:
        xi = xi - step * grad(xi)
        yield xi

    return
