
# coding: utf-8

# In[ ]:

def Rouwenhorst(rho, sig_z, num):
    """
    Implements Rouwenhorst AR(1) discretization
    
    z' = rho* z + sig_e * e'
    
    Input: rho, sig_z
    rho: AR(1) coeffient
    sig_z : standard deviation of AR(1) process (NOT the shock term std)
    
    Output: z, T
    z: grid
    T: Transition matrix T[i,j] = Prob(next=j|today = i)
    
    """
    import numpy as np
    p = (1+rho)/2.0
    q = (1+rho)/2.0
    psi = ((num-1)**0.5)*sig_z
    
    z = np.linspace(-psi, psi, num)
    
    T = np.array([[p, 1-p], [1-q, q]])
    
    if num == 2:
        return [z, T]
    elif num > 2:
        for i in range(3, num+1):

            # print(T)           
            # print(np.zeros(i-1))
            # print(np.zeros(i))
            #print(p*np.vstack((np.c_[T, np.zeros(i-1)], np.zeros(i))))
            #print((1-p)*np.vstack((np.c_[np.zeros(i-1), T], np.zeros(i))))
            #print((1-q)*np.vstack((np.zeros(i), np.c_[T, np.zeros(i-1)])))
            #print(q*np.vstack((np.zeros(i), np.c_[np.zeros(i-1), T])))
            
            
            T = p*np.vstack((np.c_[T, np.zeros(i-1)], np.zeros(i))) +            (1-p)*np.vstack((np.c_[np.zeros(i-1), T], np.zeros(i))) +            (1-q)*np.vstack((np.zeros(i), np.c_[T, np.zeros(i-1)])) +            q*np.vstack((np.zeros(i), np.c_[np.zeros(i-1), T]))
    
        for i in range(num):
            #print(T[i,:])
            T[i,:] = T[i,:] / np.sum(T[i,:])
            T[i,:] = T[i,:] / np.sum(T[i,:])#I need this part to normalize rigorously
            #print("{0:.20f}".format(sum(T[i,:])))
            
        return [z, T]
    else:
        print("Error: the number of discretization must be larger than 1.")
        return None


# In[ ]:

def Stationary(prob):
        N = prob.shape[0]
        from scipy import linalg as LA
        import numpy as np
        
        v, w = LA.eig(prob.T)
        
        ind = np.nonzero(np.abs(v - 1.0)<1e-8)[0]
        if len(ind) > 0:
            i = np.min(ind)#== statement is not good

            sdist = w[:,i]/np.sum(w[:,i])

            sdist = np.real_if_close(sdist) #discards the imaginary part if it is negligible
            return sdist
        else:
            print('error: no eigenvalue which is equal to one.')
            


# In[ ]:

#test code
if __name__ == '__main__':
    
    rho = 0.95
    stde = 0.006
    num = 7
    grid, T = Rouwenhorst(rho, stde/(1.0-rho**2)**0.5, num)

    for i in range(num):
        print("{0:.50f}".format(sum(T[i,:])))
    


# In[ ]:



