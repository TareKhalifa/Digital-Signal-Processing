
# coding: utf-8

# In[1]:


import numpy as np
import cmath


# In[5]:


#Discrete Fourier Transform (DFT) 
def DFT(x): 
    N=x.shape[0]
    X_function= []

    for m in range (N):
        summation=0
        for i in range (N):
            summation+=x[i]*(np.cos(2*np.pi*i*m/N)-1j*np.sin(2*np.pi*i*m/N))
        X_function.append(summation)
    return X_function  


# In[6]:


#Inverse Discrete Fourier Transform (IDFT)
def IDFT(x):
    N=x.shape[0]
    X_function= []

    for m in range (N):
        summation=0
        for i in range (N):
            summation+=x[i]*(np.cos(2*np.pi*i*m/N)+1j*np.sin(2*np.pi*i*m/N))
        X_function.append(summation/N)
    return X_function  


# In[7]:


#Sampling
def sample(X,rate):
    print (X.shape)
    samples_array=[]
    num=X.shape[0]/rate
    print (num)
    num=int(num)
    for i in range (X.shape[0]):
        if(i%num==0):
            samples_array.append(X[i])
        
    return samples_array 
# In[8]: 

#hanning
def hanning_window(x):
    for i in range (x.shape[0]):
        x[i]=x[i]*(0.5-0.5*np.cos(2*np.pi*i/x.shape[0]))
    return x

