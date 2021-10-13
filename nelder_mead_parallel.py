"""
This code implements Lee and Wiswall (2007), Comput Econ
"""

import warnings
import sys
from numpy import (atleast_1d, eye, argmin, zeros, shape, squeeze,
                   asarray, sqrt, Inf, asfarray, isinf)
import numpy as np

_status_message = {'success': 'Optimization terminated successfully.',
                   'maxfev': 'Maximum number of function evaluations has '
                              'been exceeded.',
                   'maxiter': 'Maximum number of iterations has been '
                              'exceeded.',
                   'pr_loss': 'Desired error not necessarily achieved due '
                              'to precision loss.',
                   'nan': 'NaN result encountered.',
                   'out_of_bounds': 'The result is outside of the provided '
                                    'bounds.'}


class OptimizeResult(dict):
    """ Represents the optimization result.
    Attributes
    ----------
    x : ndarray
        The solution of the optimization.
    success : bool
        Whether or not the optimizer exited successfully.
    status : int
        Termination status of the optimizer. Its value depends on the
        underlying solver. Refer to `message` for details.
    message : str
        Description of the cause of the termination.
    fun, jac, hess: ndarray
        Values of objective function, its Jacobian and its Hessian (if
        available). The Hessians may be approximations, see the documentation
        of the function in question.
    hess_inv : object
        Inverse of the objective function's Hessian; may be an approximation.
        Not available for all solvers. The type of this attribute may be
        either np.ndarray or scipy.sparse.linalg.LinearOperator.
    nfev, njev, nhev : int
        Number of evaluations of the objective functions and of its
        Jacobian and Hessian.
    nit : int
        Number of iterations performed by the optimizer.
    maxcv : float
        The maximum constraint violation.
    Notes
    -----
    There may be additional attributes not listed above depending of the
    specific solver. Since this class is essentially a subclass of dict
    with attribute accessors, one can see which attributes are available
    using the `keys()` method.
    """

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __repr__(self):
        if self.keys():
            m = max(map(len, list(self.keys()))) + 1
            return '\n'.join([k.rjust(m) + ': ' + repr(v)
                              for k, v in sorted(self.items())])
        else:
            return self.__class__.__name__ + "()"

    def __dir__(self):
        return list(self.keys())

    
def _update_k_(x, args):
    
    xj, xjm1, xbar, f0, fj, fjm1 = x

#     print(x)
#     [print(xx) for xx in x]
#     xj = x[0]
#     xjm1 = x[1]
#     xbar = x[2]
#     f0   = x[3]
#     fj  = x[4]
#     fjm1 = x[5]

    func, rho, chi, psi, bounds, lower_bound, upper_bound = args
 
    
    """
    xbar: centroid
    xr, fxr
    xe, fxe
    xc, fxc
    xcc, fxcc
    """
    
#     print(f'xbar = {xbar}')

    # xjnext
    # fjnext
    doshrink = 0
    fcount = 0
    
    # calculate a reflection point
    # xr = (1 + rho) * xbar - rho * sim[-1] 
    xr = (1 + rho) * xbar - rho * xj
    if bounds is not None:
        xr = np.clip(xr, lower_bound, upper_bound)
        
    fxr = func(xr) # evaluate on reflextion point
    fcount += 1
    
#     print(f'xr = {xr}, fxr = {fxr}')
    
    # case 1: expansion
    if fxr < f0:

        xe = (1 + rho * chi) * xbar - rho * chi * xj

        if bounds is not None:
            xe = np.clip(xe, lower_bound, upper_bound)
        fxe = func(xe) # evaluate on expansion point
        fcount += 1

#         print(f'xe = {xe}, fxr = {fxe}')


        if fxe < fxr:
            xjnext = xe
            fjnext = fxe
            
        else:
            xjnext = xr
            fjnext = fxr
            
    else:  # fsim[0] <= fxr

        #case 2
        if fxr < fjm1:
            xjnext = xr
            fjnext = fxr

        #case 3
        else:  # fxr >= fsim[-2]
            # Perform contraction
            # case 3.1
            if fxr < fj:
                xc = (1 + psi * rho) * xbar - psi * rho * xj
                if bounds is not None:
                    xc = np.clip(xc, lower_bound, upper_bound)
                fxc = func(xc) # evaluate on contraction point
                fcount += 1

#                 print(f'xc = {xc}, fxr = {fxc}')
                if fxc <= fxr:       
                    xjnext = xc
                    fjnext = fxc

                else:
                    #guessing
                    xjnext = xr
                    fjnext = frc                    
                    doshrink = 1

            # case 3.2
            else:
                # Perform inside contraction
                xcc = (1 - psi) * xbar + psi * xj
                if bounds is not None:
                    xcc = np.clip(xcc, lower_bound, upper_bound)
                fxcc = func(xcc) # evaluate on contraction point
                fcount += 1
                
#                 print(f'xc = {xcc}, fxr = {fxcc}')


                if fxcc < fj:
                    xjnext = xcc
                    fjnext = fxcc
                else:
                    #guessing
                    xjnext = xj
                    fjnext = fj                             
                    doshrink = 1
    
    
    return xjnext, fjnext, fcount, doshrink


import functools
def minimize_neldermead_parallel(func, x0, args=(), callback=None,
                                 maxiter=None, maxfev=None, disp=False,
                                 return_all=False, initial_simplex=None,
                                 xatol=1e-4, fatol=1e-4, adaptive=False, bounds=None,
                                 executor = None, num_process = 2, p = 2 ,
                                 ):


    
    # assert len(x0) + 1 > p, 'dim of x must be larger than p'
    
    
    if executor is None:
        from concurrent.futures import ProcessPoolExecutor
        executor = ProcessPoolExecutor(num_process)


    maxfun = maxfev
    retall = return_all

    fcalls = [0]
    # warning: this part is not tested.
    if len(args) > 0:
        func   = functools.partial(func, args)
        # fcalls, func = _wrap_function(func, args)
        


    if adaptive:
        dim = float(len(x0))
        rho = 1
        chi = 1 + 2/dim
        psi = 0.75 - 1/(2*dim)
        sigma = 1 - 1/dim
    else:
        rho = 1
        chi = 2
        psi = 0.5
        sigma = 0.5

    nonzdelt = 0.05
    zdelt = 0.00025

    x0 = asfarray(x0).flatten()

    if bounds is not None:
        lower_bound, upper_bound = bounds.lb, bounds.ub
        # check bounds
        if (lower_bound > upper_bound).any():
            raise ValueError("Nelder Mead - one of the lower bounds is greater than an upper bound.")
        if np.any(lower_bound > x0) or np.any(x0 > upper_bound):
            warnings.warn("Initial guess is not within the specified bounds",
                          OptimizeWarning, 3)
    else:
        lower_bound, upper_bound = None, None

    if bounds is not None:
        x0 = np.clip(x0, lower_bound, upper_bound)

    if initial_simplex is None:
        N = len(x0)

        sim = np.empty((N + 1, N), dtype=x0.dtype) #simplex has N+1 points, each of which is in R^n
        sim[0] = x0
        for k in range(N):
            y = np.array(x0, copy=True)
            if y[k] != 0:
                y[k] = (1 + nonzdelt)*y[k]
            else:
                y[k] = zdelt
            sim[k + 1] = y
    else:
        sim = np.asfarray(initial_simplex).copy()
        if sim.ndim != 2 or sim.shape[0] != sim.shape[1] + 1:
            raise ValueError("`initial_simplex` should be an array of shape (N+1,N)")
        if len(x0) != sim.shape[1]:
            raise ValueError("Size of `initial_simplex` is not consistent with `x0`")
        N = sim.shape[1]

    if retall:
        allvecs = [sim[0]]

    # If neither are set, then set both to default
    if maxiter is None and maxfun is None:
        maxiter = N * 200
        maxfun = N * 200
    elif maxiter is None:
        # Convert remaining Nones, to np.inf, unless the other is np.inf, in
        # which case use the default to avoid unbounded iteration
        if maxfun == np.inf:
            maxiter = N * 200
        else:
            maxiter = np.inf
    elif maxfun is None:
        if maxiter == np.inf:
            maxfun = N * 200
        else:
            maxfun = np.inf

    if bounds is not None:
        sim = np.clip(sim, lower_bound, upper_bound)

    one2np1 = list(range(1, N + 1))
    fsim = np.empty((N + 1,), float)

    fsim[:] = np.array(list(executor.map(func, sim)))
    fcalls[0] += N+1
    
    # gets an error with partial(update_k, ...)
    update_k = functools.partial(_update_k_, args = (func, rho, chi, psi, bounds, lower_bound, upper_bound,))


    ind = np.argsort(fsim)
    fsim = np.take(fsim, ind, 0)
    sim = np.take(sim, ind, 0)

    iterations = 1

    # main algo
    while (fcalls[0] < maxfun and iterations < maxiter):
        if (np.max(np.ravel(np.abs(sim[1:] - sim[0]))) <= xatol and
                np.max(np.abs(fsim[0] - fsim[1:])) <= fatol):
            break

        # add.reduce is same with np.sum
        xbar = np.add.reduce(sim[:-p], 0) / (N + 1 - p)
        
        # prep input data
        worstps    = sim[-p:]
        worstps_p1 = sim[-(p+1):-1]
        
        worstfs    = fsim[-p:]
        worstfs_p1 = fsim[-(p+1):-1] 
        
        update_data = [(worstps[i], worstps_p1[i], xbar, fsim[0], worstfs[i], worstfs_p1[i],) for i in range(p)]

        
        # update_k(xj, xjm1, xbar, f0, fj, fjm1)
        res = executor.map(update_k, update_data)
        res = list(res)
        

        
        do_shrink_indices = [res[i][-1] for i in range(p)]
        sim_new           = [res[i][0] for i in range(p)]
        fsim_new          = [res[i][1] for i in range(p)]
        fcounts_new       = [res[i][2] for i in range(p)]
        fcalls[0]        += np.sum(fcounts_new)
        
        sim[-p:]  = sim_new
        fsim[-p:] = fsim_new
                
        
        # if all p points return do_dhrink == 1
        if np.prod(do_shrink_indices) == 1:
            
            #here????
            sim[1:] = sim[0] + sigma * (sim[1:] - sim[0])
            if bounds is not None:
                # this may not be ok for multi-dim points
                # sim[1:] = np.clip(sim[1:], lower_bound, upper_bound) 
                for i in one2np1:
                     sim[i] = np.clip(sim[i], lower_bound, upper_bound)
                    
                
            tmp = executor.map(func, sim[1:])
            fsim[1:] = list(tmp)# np.array?
            

                              

        ind = np.argsort(fsim)
        sim = np.take(sim, ind, 0)
        fsim = np.take(fsim, ind, 0)
        if callback is not None:
            callback(sim[0])
        iterations += 1
        if retall:
            allvecs.append(sim[0])

    x = sim[0]
    fval = np.min(fsim)
    warnflag = 0

    if fcalls[0] >= maxfun:
        warnflag = 1
        msg = _status_message['maxfev']
        if disp:
            print('Warning: ' + msg)
    elif iterations >= maxiter:
        warnflag = 2
        msg = _status_message['maxiter']
        if disp:
            print('Warning: ' + msg)
    else:
        msg = _status_message['success']
        if disp:
            print(msg)
            print("         Current function value: %f" % fval)
            print("         Iterations: %d" % iterations)
            print("         Function evaluations: %d" % fcalls[0])

    result = OptimizeResult(fun=fval, nit=iterations, nfev=fcalls[0],
                            status=warnflag, success=(warnflag == 0),
                            message=msg, x=x, final_simplex=(sim, fsim))
    if retall:
        result['allvecs'] = allvecs
    return result
