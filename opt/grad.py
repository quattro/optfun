from numpy.linalg import multi_dot


def gd(x0, step, grad, **kwargs):
    """ Vanilla gradient descent
    """

    yield x0

    xi = x0
    while True:
        xi = xi - step(xi) * grad(xi)
        yield xi

    return


def mgd(x0, step, grad, **kwargs):
    """ Polyak's heavy ball method for gradient descent
    """

    yield x0

    friction = kwargs.get("friction", 0)

    pi = 0
    xi = x0
    while True:
        pi = -step(xi) * grad(xi) + friction * pi
        xi = xi + pi
        yield xi

    return


def agd(x0, step, grad, **kwargs):
    """ Nesterov's accelerated gradient descent
    """
    yield x0

    friction = kwargs.get("friction", 0)

    pi = 0
    xi = x0
    while True:
        p_tmp = pi
        pi = friction * pi - step(xi) * grad(xi)
        xi = xi - friction * p_tmp + (1 + friction) * pi
        yield xi

    return


def lcgd(x0, step, grad, **kwargs):
    """ Conjugate Gradient Descent for linear systems
    """
    yield x0

    hess = kwargs.get("hess", lambda x: None)

    rlast = grad(x0)
    pi = -rlast
    xi = x0
    while True:
        ai = rlast.dot(rlast) / multi_dot([pi, hess(xi), pi])
        xi = xi + ai * pi
        yield xi

        ri = rlast + ai * hess(xi).dot(pi)
        bi = ri.dot(ri) / rlast.dot(rlast)
        pi = -ri + bi * pi
        rlast = ri

    return
