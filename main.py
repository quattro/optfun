#! /usr/bin/env python

import argparse as ap
import os
import sys
import numpy as np
import numpy.linalg as linalg
import opt
import scipy.stats as stats

from numpy.linalg import multi_dot


def get_matrix(n, beta, alpha):
    """
    Generate a random matrix with condition number kappa = beta / alpha
    """
    A = stats.norm.rvs(size=(n, n))
    Q, R = linalg.qr(A)
    S = stats.norm.rvs(size=n)
    S = 10 ** S
    Smin = min(S)
    Smax = max(S)
    S = (S - Smin) / (Smax - Smin)
    S = alpha + S * (beta - alpha)
    A = multi_dot([Q.T, np.diag(S), Q])

    return A


def quadratic(size=2, beta=10, alpha=1):
    """
    Generate a random quadratic form
    """
    Q = get_matrix(size, beta, alpha)
    b = stats.norm.rvs(size=size)
    c = stats.norm.rvs()
    f = lambda x: 0.5 * multi_dot([x, Q, x]) - b.dot(x) + c
    g = lambda x: Q.dot(x) - b
    h = lambda x: Q

    evals = linalg.eigvals(Q)
    beta = max(evals)
    alpha = min(evals)

    return (f, g, h, beta, alpha)


def run(optfun, f, x0, step, grad, verbose, error_tol, output, **kwargs):
    for idx, local in enumerate(optfun(x0, step, grad, **kwargs)):
        lsol_cur = f(local)
        if verbose:
            output.write("Step {}: f({}) = {}{}".format(idx, local, f(local), os.linesep))
        else:
            output.write("\rStep {}: f(xi) = {}".format(idx, f(local)))
            output.flush()

        if idx > 0:
            diff = lsol_last - lsol_cur
            if 0 <= diff <= error_tol:
                break

        lsol_last = lsol_cur

    if not verbose:
        output.write(os.linesep)

    return


def main(args):
    argp = ap.ArgumentParser(description="")
    argp.add_argument("-d", "--dim", type=int, default=2)
    argp.add_argument("-a", "--alpha", type=int, default=1, help="minimum eigenvalue")
    argp.add_argument("-b", "--beta", type=int, default=10, help="maximum eigenvalue")
    argp.add_argument("-e", "--error-tol", type=float, default=1e-3)
    argp.add_argument("-v", "--verbose", action="store_true", default=False)
    argp.add_argument("-o", "--output", type=ap.FileType("w"), default=sys.stdout)

    args = argp.parse_args(args)

    # build function
    f, g, h, beta, alpha = quadratic(args.dim, args.beta, args.alpha)

    # Backtracking line search
    def backtrack(xi, rate):
        rho = 0.9
        c = 0.5
        gi = g(xi)
        while True:
            v1 = f(xi - rate * gi)
            v2 = f(xi) - c * rate * gi.dot(gi)
            if v1 <= v2:
                break

            rate = rho * rate
        return rate

    # Exact line search for quadratic
    def exact(xi, rate):
        gi = g(xi)
        Q = h(xi)
        return gi.dot(gi) / multi_dot([gi, Q, gi])

    # initial guess
    x0 = np.random.normal(size=args.dim)

    args.output.write("Condition number: {}".format(beta / alpha) + os.linesep)

    #GD
    step = lambda x: 2 / (beta + alpha)
    args.output.write("Gradient Descent with step {}".format(step(0)) + os.linesep)
    run(opt.gd, f, x0, step, g, args.verbose, args.error_tol, args.output)

    step = lambda x: 1 / beta
    args.output.write("Gradient Descent with step {}".format(step(0)) + os.linesep)
    run(opt.gd, f, x0, step, g, args.verbose, args.error_tol, args.output)

    #GD + backtracking line search
    line_step = lambda x: backtrack(x, 0.2)
    args.output.write("Gradient Descent with backtracking line search" + os.linesep)
    run(opt.gd, f, x0, line_step, g, args.verbose, args.error_tol, args.output)

    #GD + exact line search
    line_step = lambda x: exact(x, step(0))
    args.output.write("Gradient Descent with exact line search" + os.linesep)
    run(opt.gd, f, x0, line_step, g, args.verbose, args.error_tol, args.output)

    #GD + momentum
    step = lambda x: 4 / (np.sqrt(beta) + np.sqrt(alpha)) ** 2
    friction = (np.sqrt(beta) - np.sqrt(alpha)) / (np.sqrt(beta) + np.sqrt(alpha))
    args.output.write("Polyak Heavy-ball Gradient Descent with step {} and friction {}".format(step(0), friction) + os.linesep)
    run(opt.mgd, f, x0, step, g, args.verbose, args.error_tol, args.output, friction=friction)

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
