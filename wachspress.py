import numpy as np
import nb as nb


@nb.njit
def getNormals(v):
    
    num_points = len(v)
    
    un = np.zeros((num_points, 2))
    
    for i in range(num_points):
        d = v[(i+1)%num_points] - v[i]
        norm_d = np.linalg.norm(d)
        
        un[i, 0] =  d[1]/norm_d
        un[i, 1] = -d[0]/norm_d

        
        # for j in range(2):
        #     un[i, j] = d[j]/norm_d
        
    return un



@nb.njit
def wachspress2d(v, x):
    
    """
    Source : Floater, Gillette, Sukumar (2013), 
             with a minor modification to allow for x that is on the edge
    
    Inputs:
    v    : [x1 y1; x2 y2; ...; xn yn], the n vertices of the polygon in ccw
    x    : [x(1) x(2)], the point at which the basis functions are computed
    Outputs:
    phi  : output basis functions = [phi_1; ...; phi_n]
    """
    
    num_points = len(v)
    w   = np.zeros(num_points)
    # R   = np.zeros((num_points, 2))
    phi = np.zeros(num_points)
    h   = np.zeros(num_points)
    
    un = getNormals(v)
    p  = np.zeros((num_points,2))
    
    
    for i in range(num_points):
        h[i] = np.dot(v[i] - x, un[i])
        # print(h)
        # p[i] = un[i] / h
        
    # print(p)
    mat = np.zeros((2, 2))
    for i in range(num_points):
        im1 = (i-1)%num_points
        mat[0] = un[im1]
        mat[1] = un[i]
        w[i] = np.linalg.det(mat)
        for j in range(num_points):
            if (j != i) and (j != im1):
                w[i] = w[i] * h[j]
        # R[i] = p[im1] + p[i]
    
    wsum = np.sum(w)
    # print(wsum)
    phi = w/wsum
    return phi
