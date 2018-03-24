
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
        xi = xi - step * grad(xi)
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

    xlast = x0
    xi = xlast
    while True:
        tmp = xi - step * grad(xi) + friction * (xi - xlast)
        yield tmp
        xlast = xi
        xi = tmp

    return
