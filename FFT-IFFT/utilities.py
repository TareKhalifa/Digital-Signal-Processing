
# coding: utf-8

# In[1]:


import numpy as np
import scipy.fftpack
import cmath as cmath
import math as math
import scipy.integrate as integrate
import scipy.special as special


# In[3]:


class FFT (object):
    def __init__(self,x):
        #making it power of two by appending zeros
        dim=math.ceil(math.log2(x.shape[0]))
        dim=2**dim
        diff=dim-x.shape[0]
        z=np.zeros(diff)
        x=np.append(x,z)
        self.c=2        
        self.n=x.shape[0]
        self.step=self.n/2  
        index_max="{0:b}".format(x.shape[0]-1)
        self.data_array=[]
        for i in range (x.shape[0]):
            temp="{0:b}".format(i)
            while (len(temp)<len(index_max)):
                temp='0'+temp
            temp=temp[::-1]
            length = len(temp)
            dim = 0
            for i in range(length):
                dim = dim + int(temp[i])
                dim = dim * 2
            dim=dim/2
            self.data_array.append(x[int(dim)])          
        self.fft_process()
        
    def fft_process(self): #FFT
        Wn= cmath.exp(-1j*(np.pi*2)/self.n)
        all_Wn=[]
        new_data_array=np.zeros(self.n,dtype= complex)
        t=0
        #get Wns
        for i in range (self.n):
            if(i%self.c==0):
                t=0
            else:
                t+=self.step

            all_Wn.append(Wn**t)                
        for i in range (self.n):
            if(i%self.c<self.c/2):
                new_data_array[i]+=1 * self.data_array[i]
                new_data_array[int(i+self.c/2)]+=1 * self.data_array[i]                
            else:
                new_data_array[i]+=all_Wn[i] * self.data_array[i]
                new_data_array[int(i-self.c/2)]+=all_Wn[int(i-self.c/2)] * self.data_array[i]
        self.data_array=new_data_array
        if(self.c!=self.n):
            self.c=self.c*2
            self.step=self.step/2
            self.fft_process()
    def fft(self):
        return self.data_array
        



# In[4]:


class IFFT (object):
    def __init__(self,x):
        self.n=x.shape[0]
        self.c=2
        self.step=self.n/2        
        
        index_max="{0:b}".format(x.shape[0]-1)
        self.data_array=[]
        for i in range (x.shape[0]):
            temp="{0:b}".format(i)
            while (len(temp)<len(index_max)):
                temp='0'+temp
            temp=temp[::-1]

            length = len(temp)
            dim = 0
            for i in range(length):
                dim = dim + int(temp[i])
                dim = dim * 2
            dim=dim/2
            self.data_array.append(x[int(dim)])
        self.ifft_process()
        
        
    def ifft_process(self):
        Wn= cmath.exp(-1j*(np.pi*2)/self.n)
        all_Wn=[]
        new_data_array=np.zeros(self.n,dtype= complex)
        t=0
        for i in range (self.n):
            if(i%self.c==0):
                t=0
            else:
                t+=self.step
            all_Wn.append (Wn**(-1*t))               
        for i in range (self.n):
            if(i%self.c<self.c/2):
                new_data_array[i]+=1 * self.data_array[i]
                new_data_array[int(i+self.c/2)]+=1 * self.data_array[i]                
            else:
                new_data_array[i]+=all_Wn[i] * self.data_array[i]
                new_data_array[int(i-self.c/2)]+=all_Wn[int(i-self.c/2)] * self.data_array[i]
        self.data_array=new_data_array
        if(self.c!=self.n):
            self.c=self.c*2
            self.step=self.step/2
            self.ifft_process()
        
    def ifft(self):
        return np.real(np.array(self.data_array)/self.n).astype(np.int16)
    
#  In[5]:


def lowpass_filter (x):
    flag=0
    for i in range (x.shape[0]):
        if np.abs(x[i])> 0.8*1e8:
            flag=1
        if(flag):
            x[i]=0   
    return x
        
    


# In[ ]:


def sample(x,num):
    if(x.shape==num):
        return x,num
    else:
        samples=[]
        n=x.shape[0]/num
        n=int(n+1)
        for i in range (x.shape[0]):
            if(i%n==0):
                samples.append(x[i])
        n=len(samples)
        if(len(samples)<num):
            diff=num-len(samples)
            z=np.zeros(diff)
            samples=np.array(samples)
            samples=np.append(samples,z)
        return samples,n   
