
def gd(x0, step, grad, **kwargs):
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
        xi = xi - step(xi) * grad(xi)
        yield xi

    return


def mgd(x0, step, grad, **kwargs):
    """ Polyak's heavy ball method for gradient descent

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

    friction = kwargs.get("friction", 0)

    pi = 0
    xi = x0
    while True:
        pi = -step(xi) * grad(xi) + friction * pi
        xi = xi + pi
        yield xi

    #v = mu * v - learning_rate * dx # integrate velocity
    #x += v # integrate position

    return
