#! /usr/bin/env python
import argparse as ap
import os
import sys

import numpy as np
import numpy.linalg as linalg
import opt
import scipy.stats as stats

from numpy.linalg import multi_dot



def quadratic(size=2):
    Q = stats.wishart.rvs(size ** 2, np.eye(size))
    b = stats.norm.rvs(size=size)
    c = stats.norm.rvs()
    f = lambda x: multi_dot([x, Q, x]) + b.dot(x) + c
    g = lambda x: Q.dot(x) + b

    evals = linalg.eigvals(Q)
    beta = max(evals)
    alpha = min(evals)

    return (f, g, beta, alpha)


def main(args):
    argp = ap.ArgumentParser(description="")
    argp.add_argument("-d", "--dim", type=int, default=2)
    argp.add_argument("-l", "--learn_rate", type=float, default=0.1)
    argp.add_argument("-e", "--error-tol", type=float, default=1e-3)
    argp.add_argument("--opt-rate", action="store_true", default=False, help="Use optimal learning rate?")
    argp.add_argument("-o", "--output", type=ap.FileType("w"), default=sys.stdout)

    args = argp.parse_args(args)

    f, g, beta, alpha = quadratic(args.dim)
    if args.opt_rate:
        step_size = 1 / beta
    else:
        step_size = args.learn_rate

    x0 = np.random.normal(size=args.dim)
    for idx, local in enumerate(opt.gradient_descent(x0, step_size, g)):
        lsol_cur = f(local)
        args.output.write("Step {}: f({}) = {}{}".format(idx, local, f(local), os.linesep))

        if idx > 0 and lsol_last - lsol_cur < args.error_tol:
            break

        lsol_last = lsol_cur

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
