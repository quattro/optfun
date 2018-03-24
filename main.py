#! /usr/bin/env python

import argparse as ap
import os
import sys
import numpy as np
import numpy.linalg as linalg
import opt
import scipy.stats as stats

from numpy.linalg import multi_dot


VANILLA = "vanilla"
POLYAK = "polyak"


def quadratic(size=2):
    Q = stats.wishart.rvs(size ** 2, np.eye(size))
    b = stats.norm.rvs(size=size)
    c = stats.norm.rvs()
    f = lambda x: multi_dot([x, Q, x]) - b.dot(x) + c
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
            elif diff < 0:
                if verbose:
                    output.write("Step DIVERGED: f({}) = {}{}".format(local, f(local), os.linesep))
                else:
                    output.write("\rStep DIVERGED: f(xi) = {}".format(f(local)))
                    output.flush()

                break

        lsol_last = lsol_cur

    if not verbose:
        output.write(os.linesep)

    return


def main(args):
    argp = ap.ArgumentParser(description="")
    argp.add_argument("-d", "--dim", type=int, default=2)
    argp.add_argument("-l", "--learn_rate", type=float, default=0.1)
    argp.add_argument("-e", "--error-tol", type=float, default=1e-3)
    argp.add_argument("-r", "--opt_rate", action="store_true", default=False)
    argp.add_argument("-v", "--verbose", action="store_true", default=False)
    argp.add_argument("-o", "--output", type=ap.FileType("w"), default=sys.stdout)

    args = argp.parse_args(args)

    # build function
    f, g, h, beta, alpha = quadratic(args.dim)

    # use optimal step size?
    if args.opt_rate:
        step = lambda x: 1 / beta
    else:
        step = lambda x: args.learn_rate

    # Backtracking line search
    def backtrack(xi, rate):
        rho = 0.9
        c = 1e-4
        while True:
            gi = g(xi)
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

    #GD
    args.output.write("Gradient Descent" + os.linesep)
    run(opt.gd, f, x0, step, g, args.verbose, args.error_tol, args.output)

    #GD + backtracking line search
    line_step = lambda x: backtrack(x, step(0))
    args.output.write("Gradient Descent with Backtracking line search" + os.linesep)
    run(opt.gd, f, x0, line_step, g, args.verbose, args.error_tol, args.output)

    #GD + exact line search
    line_step = lambda x: exact(x, step(0))
    args.output.write("Gradient Descent with exact line search" + os.linesep)
    run(opt.gd, f, x0, line_step, g, args.verbose, args.error_tol, args.output)

    #GD + momentum
    args.output.write("Polyak Heavy-ball Gradient Descent" + os.linesep)
    run(opt.mgd, f, x0, step, g, args.verbose, args.error_tol, args.output, friction=0.25)

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
