from orderedTableSearch import locate, locate_grid
from numba import njit
import numpy as np

@njit
def tridag(a, b, c, r, u):

    num = len(b)

    gam = np.zeros(num)

    bet = b[0]
    u[0] = r[0] / bet

    for i in range(1, num):

        gam[i] = c[i-1] / bet
        bet = b[i] - a[i-1] * gam[i]
        assert bet != 0.

        u[i] = (r[i] - a[i-1]*u[i-1])/bet

    for i in range(1, num):
        j = num - i - 1
        u[j] = u[j] - gam[j+1]*u[j+1]




@njit
def spline(x, y, yp1 = 0., ypn = 0., bc_type = 'natural'):

    assert len(x) == len(y)
    num = len(x)
    y2 = np.zeros(num)

    a = np.zeros(num)
    b = np.zeros(num)
    c = np.zeros(num)
    r = np.zeros(num)

    c[:-1] = x[1:] - x[:-1]
    r[:-1] = 6.0 * (y[1:] - y[:-1]) / c[:-1]
    r[1:] = r[1:] - r[:-1]
    a[1:] = c[:-1]
    b[1:] = 2.*(c[1:] + a[1:])
    b[0] = 1.0
    b[-1] = 1.0


    if bc_type == 'natural':
        r[0] = 0.
        c[0] = 0.

        r[-1] = 0.
        a[-1]  = 0.

    else:
        r[0] = (3.0/(x[1] - x[0])) * ((y[1] - y[0])/(x[1]- x[0]) - yp1)
        c[0] = 0.5

        r[-1] = (-3.0/(x[-1] - x[-2])) * ((y[-1] - y[-2])/(x[-1]- x[-2]) - ypn)
        a[-1] = 0.5

    tridag(a[1:], b, c[:-1], r, y2)


    return y2


@njit
def splint(xa, ya, y2a, x):

    klo = locate(x, xa) # check
    khi = klo + 1

    h = xa[khi] - xa[klo]
    a = (xa[khi] - x)/h
    b = (x - xa[klo])/h

    splint = a*ya[klo] + b*ya[khi] + ((a**3. - a) * y2a[klo] + (b**3. - b)*y2a[khi])*(h**2.) / 6.0

    return splint

@njit
def dsplint(xa, ya, y2a, x):

    klo = locate(x, xa) # check
    khi = klo + 1

    h = xa[khi] - xa[klo]
    a = (xa[khi] - x)/h
    b = (x - xa[klo])/h

    dadx = -1./h
    dbdx = 1./h
    
    dsplint = dadx*ya[klo] + dbdx*ya[khi] + ((3.*a**2. - 1.)*dadx * y2a[klo] + (3.*b**2. - 1.)*dbdx*y2a[khi])*(h**2.) / 6.0

    return dsplint


@njit
def dsplint_grid(xa, ya, y2a, xpoints):

    num = len(xpoints)
    dsplint_ans = np.zeros(num)

    klo_grid = locate_grid(xpoints, xa) # check
    khi_grid = klo_grid + 1

    for i in range(num):
        khi, klo = klo_grid[i], khi_grid[i]
        x = xpoints[i]

        h = xa[khi] - xa[klo]
        a = (xa[khi] - x)/h
        b = (x - xa[klo])/h

        dadx = -1./h
        dbdx = 1./h        

        dsplint_ans[i] = dadx*ya[klo] + dbdx*ya[khi] + ((3.*a**2. - 1.)*dadx * y2a[klo] + (3.*b**2. - 1.)*dbdx*y2a[khi])*(h**2.) / 6.0        
        # splint_ans[i] = a*ya[klo] + b*ya[khi] + ((a**3. - a) * y2a[klo] + (b**3. - b)*y2a[khi])*(h**2.) / 6.0

    return dsplint_ans

@njit
def splint_grid(xa, ya, y2a, xpoints):

    num = len(xpoints)
    splint_ans = np.zeros(num)

    klo_grid = locate_grid(xpoints, xa) # check
    khi_grid = klo_grid + 1

    for i in range(num):
        khi, klo = klo_grid[i], khi_grid[i]
        x = xpoints[i]

        h = xa[khi] - xa[klo]
        a = (xa[khi] - x)/h
        b = (x - xa[klo])/h

        splint_ans[i] = a*ya[klo] + b*ya[khi] + ((a**3. - a) * y2a[klo] + (b**3. - b)*y2a[khi])*(h**2.) / 6.0

    return splint_ans



@njit
def cubic_interp(xvals, xa, ya, y2a):

    if isinstance(xvals, float) or isinstance(xvals, int):    
        
        return splint(xa, ya, y2a, xvals)

    else: #if xvals is actually array-like
            
        return splint_grid(xa, ya, y2a, xvals)


@njit
def dcubic_interp(xvals, xa, ya, y2a):

    if isinstance(xvals, float) or isinstance(xvals, int):    
        
        return dsplint(xa, ya, y2a, xvals)

    else: #if xvals is actually array-like
            
        return dsplint_grid(xa, ya, y2a, xvals)


@nb.njit
def cubic_convert_y_ypp_to_polycoef(xs, ys, ypps):

    """
    Convert cubic spline parameters (x_i, y_i, y''_i)_{i=0,...N} 
    into cubic polynomial coefficients (a_j, b_j, c_j, d_j)_{j=0,...N-1} 
    where f_j(x) = a_j x^3 + b_j x^2 + c_j x + d_j
    """
    

    assert len(xs) == len(ys) == len(ypps)

    num_region = len(xs) - 1

    deltas = xs[1:] - xs[:-1]

    coef = np.zeros((num_region, 4,))

    for j in range(num_region):

        coef[j, 3] = (ypps[j+1] - ypps[j])/(6.*deltas[j])
        coef[j, 2] = (xs[j+1]*ypps[j] - xs[j]*ypps[j+1])/(2.*deltas[j])
        coef[j, 1] = (-ys[j] + ys[j+1])/deltas[j] \
                     + (-(xs[j+1]**2.0)/(2.*deltas[j]) + deltas[j]/6.)*ypps[j] \
                     + ((xs[j]**2.0)/(2.*deltas[j]) - deltas[j]/6.)*ypps[j+1]
        
        coef[j, 0] = (xs[j+1]*ys[j] - xs[j]*ys[j+1])/deltas[j] \
                     + ((xs[j+1]**3.0)/(6. * deltas[j]) - xs[j+1]*deltas[j]/6.)*ypps[j] \
                     + ((-xs[j]**3.0)/(6. * deltas[j]) + xs[j]*deltas[j]/6.)*ypps[j+1]

    return coef
    