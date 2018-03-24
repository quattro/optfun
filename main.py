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

    evals = linalg.eigvals(Q)
    beta = max(evals)
    alpha = min(evals)

    return (f, g, beta, alpha)


def run(optfun, f, x0, step, grad, verbose, error_tol, output, **kwargs):
    for idx, local in enumerate(optfun(x0, step, grad, **kwargs)):
        lsol_cur = f(local)
        if verbose:
            output.write("Step {}: f({}) = {}{}".format(idx, local, f(local), os.linesep))
        else:
            output.write("\rStep {}: f(xi) = {}".format(idx, f(local)))
            output.flush()

        if idx > 0 and lsol_last - lsol_cur < error_tol:
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
    f, g, beta, alpha = quadratic(args.dim)

    # use optimal step size?
    if args.opt_rate:
        step_size = 1 / beta
    else:
        step_size = args.learn_rate

    # initial guess
    x0 = np.random.normal(size=args.dim)

    #GD
    args.output.write("Vanilla Gradient Descent" + os.linesep)
    run(opt.gd, f, x0, step_size, g, args.verbose, args.error_tol, args.output)

    #GD + momentum
    args.output.write("Polyak Heavy-ball Gradient Descent" + os.linesep)
    run(opt.mgd, f, x0, step_size, g, args.verbose, args.error_tol, args.output, friction=0.25)

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
