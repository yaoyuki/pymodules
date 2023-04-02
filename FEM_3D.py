
# coding: utf-8

# In[1]:

import numpy as np
import numba as nb

#my library
from orderedTableSearch import locate, locate_grid


# In[2]:

@nb.jit(nopython = True)
def fem3d_eleval(x, y, z, 
                 x0, x1, 
                 y0, y1,
                 z0, z1,
                 v0, v1, v2, v3, v4, v5, v6, v7):
# evaluate 3d element
# :z0               
#           ||          ||
#           ||          ||
# y1 ------ v3 -------- v2 -------
#           ||          ||     
#           ||  (x, y)  ||     
#           ||    *     ||     
#           ||          ||
# y0 ------ v0 -------- v1 -------
#           ||          ||
#           ||          ||
#           x0          x1
# :z1               
#           ||          ||
#           ||          ||
# y1 ------ v7 -------- v6 -------
#           ||          ||     
#           ||  (x, y)  ||     
#           ||    *     ||     
#           ||          ||
# y0 ------ v4 -------- v5 -------
#           ||          ||
#           ||          ||
#           x0          x1

    
    xi  = (2.*x - x0 - x1)/(x1 - x0)
    eta = (2.*y - y0 - y1)/(y1 - y0)
    kai = (2.*z - z0 - z1)/(z1 - z0)
    
    #To safely implement this part
    if x == x0:
        xi = -1.0
    elif x == x1:
        xi = 1.0
    
    if y == y0:
        eta = -1.0
    elif y == y1:
        eta = 1.0
        
    if z == z0:
        kai = -1.0
    elif z == z1:
        kai = 1.0        
    
    return ( (1. - xi)*(1. - eta)*(1. - kai)*v0 + 
             (1. + xi)*(1. - eta)*(1. - kai)*v1 + 
             (1. + xi)*(1. + eta)*(1. - kai)*v2 +             
             (1. - xi)*(1. + eta)*(1. - kai)*v3 +
             (1. - xi)*(1. - eta)*(1. + kai)*v4 + 
             (1. + xi)*(1. - eta)*(1. + kai)*v5 + 
             (1. + xi)*(1. + eta)*(1. + kai)*v6 +             
             (1. - xi)*(1. + eta)*(1. + kai)*v7) / 8.0

@nb.jit(nopython = True)
def fem3d_peval(xval, yval, zval, xnodes, ynodes, znodes, vnodes):
    # Evaluate 3d FEM at a given point.
    # vnodes is assumed to be array-like vnodes = [xdim, ydim, zdim]
    # note that, if this is called for each points, it would be slow.
    
    #add a dimension check
    assert len(xnodes) == vnodes.shape[0] and len(ynodes) == vnodes.shape[1] and len(znodes) == vnodes.shape[2] 
    
    
    ix = locate(xval, xnodes)
    iy = locate(yval, ynodes)
    iz = locate(zval, znodes)
    
    
    return fem3d_eleval(xval, yval, zval,
                        xnodes[ix], xnodes[ix+1], 
                        ynodes[iy], ynodes[iy+1],
                        znodes[iz], znodes[iz+1],
                        vnodes[ix, iy, iz], vnodes[ix+1, iy, iz], vnodes[ix+1, iy+1, iz], vnodes[ix, iy+1, iz],
                        vnodes[ix, iy, iz+1], vnodes[ix+1, iy, iz+1], vnodes[ix+1, iy+1, iz+1], vnodes[ix, iy+1, iz+1])

# #make another function to evaluate on a grid or a line.

#!!!!!!CAUTION!!!!!! THIS FUNCTION MAY BE WRONG
# @nb.jit(nopython = True)    
# def fem2deval(xypoints, xnodes, ynodes, znodes): #, sort = False):

#     #xypoint is array-like: xypoint[0,:] is x, xypoint[1,:] is y
    
#     if xypoints.shape[0] != 2:
#         print('ERROR: xypoints.shape[0] != 2')
# #         return None
    
    
#     ans = np.zeros(xypoints.shape[1])
    
    
#     #numba did not accept the following two arguments
#     #ixs = np.empty(xypoints.shape[1]) 
#     #iys = np.empty(xypoints.shape[1])

#     ixs = locate_grid(xypoints[0,:], xnodes)
#     iys = locate_grid(xypoints[1,:], ynodes)
    
#     for i in range(len(ans)):
#         ans[i] = fem2d_eleval(
#             xypoints[0,i], xypoints[1,i], xnodes[ixs[i]], xnodes[ixs[i]+1], ynodes[iys[i]], ynodes[iys[i]+1],
#             znodes[ixs[i], iys[i]], znodes[ixs[i]+1, iys[i]], znodes[ixs[i]+1, iys[i]+1], znodes[ixs[i], iys[i]+1])
    
#     return ans
  
#!!!!!!CAUTION!!!!!! THIS FUNCTION MAY BE WRONG
# @nb.jit(nopython = True)    
# def fem2deval(xpoints, ypoints, xnodes, ynodes, znodes): #, sort = False):

#     #xypoint is array-like: xypoint[0,:] is x, xypoint[1,:] is y
    
#     if len(xpoints) != len(ypoints):
#         print('ERROR: len(xpoints) != len(ypoints)')
#         return None
    
#     num = len(xpoints)
    
#     ans = np.zeros(num)
    
#     #numba did not accept the following two arguments
#     #ixs = np.empty(xypoints.shape[1]) 
#     #iys = np.empty(xypoints.shape[1])

#     ixs = locate_grid(xpoints, xnodes)
#     iys = locate_grid(ypoints, ynodes)
    
#     for i in range(len(ans)):
#         ans[i] = fem2d_eleval(
#             xpoints[i], ypoints[i], xnodes[ixs[i]], xnodes[ixs[i]+1], ynodes[iys[i]], ynodes[iys[i]+1],
#             znodes[ixs[i], iys[i]], znodes[ixs[i]+1, iys[i]], znodes[ixs[i]+1, iys[i]+1], znodes[ixs[i], iys[i]+1])
    
#     return ans

# In[3]:

@nb.jit(nopython = True)    
def fem3deval_mesh(xs, ys, zs, xnodes, ynodes, znodes, vnodes): #, sort = False):
    
    #I assume xs and ys are strictly accending
    
    num_x = len(xs)
    num_y = len(ys)
    num_z = len(zs)
    
    
    ixs = locate_grid(xs, xnodes)
    iys = locate_grid(ys, ynodes)
    izs = locate_grid(zs, znodes)
    
    
    vs = np.ones((num_x, num_y, num_z,)) #be careful with the direction
    
    for ix, x in enumerate(xs):
        for iy, y in enumerate(ys):
            for iz, z in enumerate(zs):

            
                vs[ix, iy, iz] = fem3d_eleval(x, y, z,
                                              xnodes[ixs[ix]], xnodes[ixs[ix]+1], 
                                              ynodes[iys[iy]], ynodes[iys[iy]+1],
                                              znodes[izs[iz]], znodes[izs[iz]+1],
                                              vnodes[ixs[ix], iys[iy], izs[iz]], vnodes[ixs[ix]+1, iys[iy], izs[iz]],
                                              vnodes[ixs[ix]+1, iys[iy]+1, izs[iz]], vnodes[ixs[ix], iys[iy]+1, izs[iz]],
                                              vnodes[ixs[ix], iys[iy], izs[iz]+1], vnodes[ixs[ix]+1, iys[iy], izs[iz]+1],
                                              vnodes[ixs[ix]+1, iys[iy]+1, izs[iz]+1], vnodes[ixs[ix], iys[iy]+1, izs[iz]+1]           
                                             )

    return vs
    


