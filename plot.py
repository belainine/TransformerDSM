# -*- coding: utf-8 -*-
"""
Created on Wed May 20 01:11:39 2020

@author: belainine
"""

import matplotlib.pyplot as plt
import torch


def  showPlot(attentions,rank=0,path='images/plot'):
    def found_fit(x):
        return  x  # Found with symfit.
    size=attentions.size(0)
    attentions=attentions.cpu().detach().numpy()
    x_data = list(range(size))
    y_data = [1 for i in list(range(size))]
    
    x_func = attentions
    # numpy will do the right thing and evaluate found_fit for all elements
    y_func = found_fit(x_func)
    
    # From here the plotting starts
    
    plt.scatter(x_data, y_data, c='r', label='data=$d_k/d_k$')
    plt.plot( y_func, label='$f(x) = ||q||*||q\'||/d_k$')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Printing batch query normalisation.')
    plt.legend()
    plt.show()
    plt.tight_layout(True)
    plt.savefig(path+'/temp{}.png'.format(rank))
    plt.close('all')

attentions=torch.rand(80)
showPlot( attentions,rank=0,path='..//images/plot')