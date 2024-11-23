import numpy as np
import numba as nb
__epsilon__ = np.finfo(float).eps
@nb.jit(nopython = True)
# def brent(func, ax, cx, args = (), bx = None, tol=1.0e-8, itmax=500):
def brent(func, ax, bx, cx, args = (), tol=1.0e-8, itmax=500):
    
    #parameters
    CGOLD=0.3819660
    
    ZEPS=1.0e-3*__epsilon__
    #*np.finfo(float).eps
    
    brent = 1.0e20
    xmin = 1.0e20
    
    a=min(ax,cx)
    b=max(ax,cx)
 
#     bx = (a+b)/2.
        
    v=bx
    w=v
    x=v
    e=0.0 
    # fx=func(x) 
    fx=func(*((x, ) + args))
    fv=fx
    fw=fx
    
    d = 0.0
    
    it = 0
    for it in range(itmax):
        
        xm=0.5*(a+b)
        tol1=tol*abs(x)+ZEPS
        tol2=2.0*tol1
        if (abs(x-xm) <= (tol2-0.5*(b-a))):
            xmin=x
            brent=fx
            return xmin, brent
        
        if (abs(e) > tol1):
            r=(x-w)*(fx-fv)
            q=(x-v)*(fx-fw)
            p=(x-v)*q-(x-w)*r
            q=2.0*(q-r)
        
            if (q > 0.0): 
                p=-p
            
            q=abs(q)
            etemp=e
            e=d
            
            if abs(p) >= abs(0.5*q*etemp) or  p <= q*(a-x) or p >= q*(b-x):

                #e=merge(a-x,b-x, x >= xm )
                if x >= xm:
                    e = a-x
                else:
                    e = b-x
                d=CGOLD*e
                
            else:
                d=p/q
                u=x+d
                
                if (u-a < tol2 or b-u < tol2): 
                    d= abs(tol1)*np.sign(xm - x)  #sign(tol1,xm-x)

        else:
            
            if x >= xm:
                e = a-x
            else:
                e = b-x

            d=CGOLD*e
        
        u = 0.  #merge(x+d,x+sign(tol1,d), abs(d) >= tol1 )
        if abs(d) >= tol1:
            u = x+d
        else:
            u = x+abs(tol1)*np.sign(d)
        
        ###put your objective function also here###
        # fu = func(u)
        fu = func(*((u, ) + args))
        
        if (fu <= fx):
            if (u >= x):
                a=x
            else:
                b=x
                
            #shft(v,w,x,u)
            v = w
            w = x
            x = u
            #shft(fv,fw,fx,fu)
            fv = fw
            fw = fx
            fx = fu
            
            
        else:
            if (u < x):
                a=u
            else:
                b=u

            if fu <= fw or w == x:
                v=w
                fv=fw
                w=u
                fw=fu

            elif fu <= fv or v == x or v == w:
                v=u
                fv=fu
                
    if it == itmax-1:
        print('brent: exceed maximum iterations')

    return x, fx