# -*- coding: utf-8 -*-
"""
Created on Wed May 20 01:11:39 2020

@author: belainine
"""

import matplotlib.pyplot as plt
import torch
import numpy as np

def  showPlot(attentions,rank=0,path='images/plot'):
    def found_fit(x):
        return  x  # Found with symfit.
    size=attentions.size(0)
    attentions=attentions.cpu().detach().numpy()/64
    x_data = list(range(size))
    y_data = [1 for i in list(range(size))]
    z_data = [attentions.mean(0) for i in list(range(size))]
    x_func = attentions
    # numpy will do the right thing and evaluate found_fit for all elements
    y_func = found_fit(x_func)
    
    # From here the plotting starts
    plt.plot( y_func,dashes=[1, 1, 1, 1], label='$f(x) = ||q||||k||/\sqrt{d_k}$')
    #plt.plot( y_func,dashes=[1, 1, 1, 1], label='$f(x) = ||q||||k||/sqrt(d_k)$')
    #plt.scatter(x_data, z_data, c='g', label='data=$mean(||q||||q\'||)$')
    #plt.scatter(x_data, y_data, c='r', label='data=$sqrt(d_k)/sqrt(d_k)$')
    plt.scatter(x_data, y_data,  c='r')#, label='data=$sqrt(d_q)/sqrt(d_q)$')
    #plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Printing batch query normalisation.')
    plt.legend()
    plt.show()
    plt.tight_layout(True)
    plt.savefig(path+'/temp{}.png'.format(rank))
    plt.close('all')



def  showPlotNone(attentions,rank=0,path='images/plot'):
    def found_fit(x):
        return  x**2.4  # Found with symfit.
    attentions=torch.rand(60)/4
    attentions1=torch.rand(60)/2
    size=attentions.size(0)
    attentions=attentions.cpu().detach().numpy()
    attentions1=attentions1.cpu().detach().numpy()
    x_data = list(range(size))
    y_data = [2*i+1 for i in list(range(size))]
    z_data = [1.5*i+1 for i in list(range(size))]
    x_func = attentions
    z_func = attentions1
    # numpy will do the right thing and evaluate found_fit for all elements
    y_func = found_fit(x_func+np.log(np.array(y_data)))
    z_func = found_fit(z_func+np.log(np.array(z_data)))
    # From here the plotting starts
    plt.plot( y_func,dashes=[10, 0, 10, 0],c='g', label='Log PPL using $QQ\'/||q||||q\'||$')
    plt.plot( z_func,dashes=[10, 0, 10, 0],c='gray', label='Log PPL using $QK/\sqrt{d_k}$')
    #plt.plot( y_func,dashes=[1, 1, 1, 1], label='$f(x) = ||q||||k||/sqrt(d_k)$')
    #plt.scatter(x_data, z_data, c='g', label='data=$mean(||q||||q\'||)$')
    #plt.scatter(x_data, y_data, c='r', label='data=$sqrt(d_k)/sqrt(d_k)$')
    #plt.scatter(x_data, y_data, c='r', label='data=$sqrt(d_q)/sqrt(d_q)$')
    plt.xlabel('Number of iterations')
    plt.ylabel('The Blue Score ')
    plt.title('')
    plt.legend()
    plt.show()
    plt.tight_layout(True)
    plt.savefig(path+'/temp{}.png'.format(rank))
    plt.close('all')

attentions=torch.rand(100)/2
showPlot( attentions,rank=0,path='images/plot')