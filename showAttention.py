<<<<<<< HEAD
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 23:55:11 2020

@author: belainine
"""
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.image as mpimg
matplotlib.use('Qt5Agg')
plt.switch_backend('Qt5Agg')
import matplotlib.ticker as ticker
from matplotlib.pyplot import cm
import numpy as np
import torch
import torch.nn.functional as F
def showPlot(points):
    def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / 32) for hid_j in range(32)]
    sinusoid_table=np.array([get_position_angle_vec(pos_i) for pos_i in range(40)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
    points=sinusoid_table
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
# example of attention
attentions=torch.tensor([[[[[0.0089, 0.0170, 0.0250, 0.0105, 0.0078, 0.0108, 0.0157],
           [0.0085, 0.0168, 0.0254, 0.0114, 0.0081, 0.0119, 0.0171],
           [0.0167, 0.0147, 0.0251, 0.0113, 0.0095, 0.0079, 0.0099],
           [0.0176, 0.0183, 0.0000, 0.0000, 0.0105, 0.0097, 0.0128],
           [0.0213, 0.0097, 0.0155, 0.0000, 0.0130, 0.0088, 0.0000],
           [0.0156, 0.0201, 0.0281, 0.0114, 0.0099, 0.0100, 0.0140],
           [0.0105, 0.0212, 0.0000, 0.0107, 0.0090, 0.0125, 0.0185]],

          [[0.0195, 0.0116, 0.0099, 0.0000, 0.0108, 0.0085, 0.0109],
           [0.0225, 0.0237, 0.0109, 0.0128, 0.0112, 0.0083, 0.0000],
           [0.0336, 0.0195, 0.0135, 0.0147, 0.0182, 0.0100, 0.0089],
           [0.0274, 0.0297, 0.0138, 0.0164, 0.0142, 0.0105, 0.0204],
           [0.0453, 0.0000, 0.0185, 0.0204, 0.0233, 0.0000, 0.0134],
           [0.0269, 0.0388, 0.0242, 0.0000, 0.0234, 0.0197, 0.0241],
           [0.0346, 0.0425, 0.0341, 0.0000, 0.0351, 0.0000, 0.0207]],

          [[0.0157, 0.0231, 0.0328, 0.0381, 0.0311, 0.0358, 0.0342],
           [0.0170, 0.0244, 0.0370, 0.0000, 0.0294, 0.0405, 0.0371],
           [0.0205, 0.0299, 0.0385, 0.0423, 0.0362, 0.0430, 0.0417],
           [0.0180, 0.0249, 0.0000, 0.0483, 0.0299, 0.0430, 0.0000],
           [0.0239, 0.0340, 0.0472, 0.0533, 0.0414, 0.0484, 0.0470],
           [0.0171, 0.0238, 0.0345, 0.0410, 0.0317, 0.0336, 0.0329],
           [0.0000, 0.0241, 0.0344, 0.0000, 0.0261, 0.0284, 0.0283]],

          [[0.0000, 0.0188, 0.0000, 0.0068, 0.0094, 0.0073, 0.0103],
           [0.0109, 0.0209, 0.0129, 0.0075, 0.0114, 0.0000, 0.0127],
           [0.0179, 0.0312, 0.0192, 0.0114, 0.0196, 0.0096, 0.0230],
           [0.0199, 0.0000, 0.0144, 0.0119, 0.0145, 0.0087, 0.0190],
           [0.0180, 0.0267, 0.0163, 0.0116, 0.0202, 0.0099, 0.0225],
           [0.0183, 0.0277, 0.0166, 0.0121, 0.0222, 0.0101, 0.0243],
           [0.0112, 0.0196, 0.0102, 0.0115, 0.0226, 0.0138, 0.0167]],

          [[0.0358, 0.0000, 0.0402, 0.0292, 0.0000, 0.0232, 0.0257],
           [0.0239, 0.0270, 0.0290, 0.0000, 0.0130, 0.0238, 0.0290],
           [0.0099, 0.0101, 0.0000, 0.0083, 0.0000, 0.0235, 0.0211],
           [0.0090, 0.0091, 0.0091, 0.0124, 0.0000, 0.0208, 0.0117],
           [0.0092, 0.0087, 0.0085, 0.0088, 0.0000, 0.0226, 0.0199],
           [0.0113, 0.0107, 0.0000, 0.0078, 0.0000, 0.0141, 0.0332],
           [0.0169, 0.0000, 0.0152, 0.0096, 0.0000, 0.0180, 0.0414]],

          [[0.0213, 0.0100, 0.0112, 0.0145, 0.0343, 0.0148, 0.0100],
           [0.0273, 0.0104, 0.0103, 0.0127, 0.0213, 0.0000, 0.0072],
           [0.0101, 0.0064, 0.0078, 0.0101, 0.0151, 0.0196, 0.0087],
           [0.0000, 0.0080, 0.0085, 0.0107, 0.0202, 0.0200, 0.0096],
           [0.0103, 0.0094, 0.0096, 0.0106, 0.0134, 0.0000, 0.0125],
           [0.0249, 0.0103, 0.0111, 0.0139, 0.0145, 0.0261, 0.0072],
           [0.0000, 0.0000, 0.0111, 0.0133, 0.0151, 0.0224, 0.0070]],

          [[0.0165, 0.0136, 0.0134, 0.0090, 0.0087, 0.0094, 0.0156],
           [0.0198, 0.0083, 0.0084, 0.0082, 0.0077, 0.0069, 0.0113],
           [0.0118, 0.0000, 0.0068, 0.0080, 0.0082, 0.0086, 0.0207],
           [0.0143, 0.0149, 0.0147, 0.0105, 0.0108, 0.0000, 0.0317],
           [0.0148, 0.0081, 0.0000, 0.0084, 0.0086, 0.0096, 0.0282],
           [0.0261, 0.0120, 0.0127, 0.0097, 0.0089, 0.0085, 0.0133],
           [0.0270, 0.0197, 0.0217, 0.0138, 0.0130, 0.0140, 0.0185]],

          [[0.0318, 0.0270, 0.0280, 0.0371, 0.0400, 0.0000, 0.0360],
           [0.0392, 0.0318, 0.0288, 0.0376, 0.0411, 0.0382, 0.0354],
           [0.0454, 0.0373, 0.0353, 0.0375, 0.0388, 0.0409, 0.0373],
           [0.0432, 0.0338, 0.0302, 0.0000, 0.0295, 0.0332, 0.0291],
           [0.0354, 0.0235, 0.0190, 0.0221, 0.0250, 0.0253, 0.0218],
           [0.0362, 0.0257, 0.0203, 0.0222, 0.0244, 0.0260, 0.0217],
           [0.0229, 0.0161, 0.0000, 0.0132, 0.0147, 0.0155, 0.0130]]],


         [[[0.0109, 0.0204, 0.0154, 0.0091, 0.0086, 0.0091, 0.0129],
           [0.0152, 0.0259, 0.0000, 0.0123, 0.0112, 0.0129, 0.0177],
           [0.0342, 0.0276, 0.0298, 0.0314, 0.0000, 0.0260, 0.0287],
           [0.0298, 0.0247, 0.0247, 0.0231, 0.0150, 0.0189, 0.0213],
           [0.0224, 0.0270, 0.0229, 0.0177, 0.0127, 0.0148, 0.0192],
           [0.0153, 0.0310, 0.0000, 0.0135, 0.0133, 0.0133, 0.0194],
           [0.0107, 0.0237, 0.0168, 0.0107, 0.0128, 0.0111, 0.0153]],

          [[0.0170, 0.0000, 0.0127, 0.0105, 0.0253, 0.0202, 0.0203],
           [0.0121, 0.0139, 0.0158, 0.0098, 0.0141, 0.0098, 0.0093],
           [0.0120, 0.0136, 0.0239, 0.0161, 0.0082, 0.0066, 0.0055],
           [0.0208, 0.0209, 0.0358, 0.0294, 0.0132, 0.0109, 0.0086],
           [0.0274, 0.0309, 0.0478, 0.0343, 0.0173, 0.0138, 0.0099],
           [0.0415, 0.0413, 0.0000, 0.0436, 0.0000, 0.0220, 0.0150],
           [0.0248, 0.0279, 0.0445, 0.0367, 0.0000, 0.0138, 0.0106]],

          [[0.0196, 0.0232, 0.0329, 0.0000, 0.0000, 0.0274, 0.0263],
           [0.0204, 0.0280, 0.0339, 0.0351, 0.0000, 0.0271, 0.0257],
           [0.0241, 0.0300, 0.0354, 0.0000, 0.0265, 0.0309, 0.0299],
           [0.0202, 0.0242, 0.0349, 0.0000, 0.0362, 0.0290, 0.0262],
           [0.0233, 0.0302, 0.0438, 0.0449, 0.0364, 0.0000, 0.0321],
           [0.0227, 0.0324, 0.0474, 0.0458, 0.0310, 0.0310, 0.0333],
           [0.0000, 0.0267, 0.0409, 0.0386, 0.0264, 0.0247, 0.0270]],

          [[0.0218, 0.0146, 0.0195, 0.0176, 0.0065, 0.0000, 0.0109],
           [0.0345, 0.0231, 0.0238, 0.0267, 0.0097, 0.0281, 0.0000],
           [0.0328, 0.0227, 0.0269, 0.0288, 0.0000, 0.0280, 0.0198],
           [0.0156, 0.0109, 0.0159, 0.0127, 0.0075, 0.0163, 0.0000],
           [0.0197, 0.0115, 0.0184, 0.0158, 0.0108, 0.0259, 0.0143],
           [0.0118, 0.0103, 0.0185, 0.0126, 0.0102, 0.0128, 0.0131],
           [0.0113, 0.0087, 0.0152, 0.0112, 0.0120, 0.0154, 0.0133]],

          [[0.0325, 0.0372, 0.0355, 0.0302, 0.0140, 0.0000, 0.0333],
           [0.0225, 0.0310, 0.0295, 0.0343, 0.0234, 0.0099, 0.0241],
           [0.0087, 0.0125, 0.0000, 0.0182, 0.0295, 0.0058, 0.0101],
           [0.0161, 0.0239, 0.0221, 0.0300, 0.0348, 0.0085, 0.0185],
           [0.0087, 0.0111, 0.0000, 0.0138, 0.0306, 0.0093, 0.0106],
           [0.0099, 0.0115, 0.0104, 0.0123, 0.0253, 0.0126, 0.0125],
           [0.0267, 0.0304, 0.0281, 0.0220, 0.0137, 0.0170, 0.0268]],

          [[0.0074, 0.0189, 0.0146, 0.0197, 0.0230, 0.0073, 0.0162],
           [0.0058, 0.0110, 0.0085, 0.0115, 0.0134, 0.0068, 0.0093],
           [0.0059, 0.0070, 0.0000, 0.0080, 0.0091, 0.0062, 0.0064],
           [0.0000, 0.0125, 0.0112, 0.0147, 0.0178, 0.0074, 0.0115],
           [0.0094, 0.0000, 0.0108, 0.0139, 0.0169, 0.0081, 0.0112],
           [0.0108, 0.0118, 0.0099, 0.0115, 0.0140, 0.0094, 0.0102],
           [0.0089, 0.0211, 0.0151, 0.0190, 0.0000, 0.0133, 0.0172]],

          [[0.0063, 0.0069, 0.0061, 0.0063, 0.0066, 0.0061, 0.0067],
           [0.0066, 0.0070, 0.0063, 0.0064, 0.0063, 0.0060, 0.0065],
           [0.0083, 0.0089, 0.0073, 0.0086, 0.0092, 0.0089, 0.0084],
           [0.0000, 0.0140, 0.0132, 0.0116, 0.0000, 0.0095, 0.0135],
           [0.0107, 0.0144, 0.0136, 0.0119, 0.0126, 0.0103, 0.0151],
           [0.0144, 0.0204, 0.0188, 0.0000, 0.0170, 0.0135, 0.0206],
           [0.0150, 0.0170, 0.0185, 0.0149, 0.0000, 0.0123, 0.0169]],

          [[0.0367, 0.0276, 0.0309, 0.0349, 0.0381, 0.0393, 0.0377],
           [0.0384, 0.0000, 0.0400, 0.0390, 0.0359, 0.0376, 0.0000],
           [0.0366, 0.0297, 0.0000, 0.0316, 0.0353, 0.0358, 0.0362],
           [0.0309, 0.0284, 0.0258, 0.0242, 0.0272, 0.0298, 0.0293],
           [0.0267, 0.0000, 0.0216, 0.0202, 0.0236, 0.0253, 0.0253],
           [0.0182, 0.0160, 0.0141, 0.0137, 0.0168, 0.0176, 0.0175],
           [0.0261, 0.0194, 0.0182, 0.0193, 0.0255, 0.0258, 0.0258]]]]],
       device='cuda:0')
def showAttention1(input_sentence, output_words, attentions,rank=0,path='images/fusion'):
    # Set up figure with colorbar

 
    
    attention=attentions[0][0].sum(0)



    figsize = [100, 100]     # figure size, inches
    
    fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True)
    
    # plot simple raster image on each sub-plot
   
    arr=  [ax]
    for i, axi in enumerate(arr):
        # i runs from 0 to (nrows*ncols-1)
        # axi is equivalent with ax[rowid][colid]

        # get indices of row/column
        rowid = 0
        colid = 0
        # write row/col indices as axes' title for identification
        # axi.set_title("Row:"+str(rowid)+", Col:"+str(colid))
        
        encodes=input_sentence.split(' ')
        axi.set_xticklabels(['']+ encodes , rotation=90,fontsize=8)
        
        decodes=output_words.split(' ')

        axi.set_yticklabels(   decodes+['<EOS>'],fontsize=8)
        
        axi.xaxis.set_major_locator(ticker.MultipleLocator(1))
        axi.yaxis.set_major_locator(ticker.MultipleLocator(1))
        img = attention.cpu().numpy()
        #img = attention.cpu().numpy()
        axi.imshow(img, alpha=0.9, cmap=cm.Blues)

        #fig.colorbar(cax)
    plt.tight_layout(True)
    plt.savefig(path+'/temp{}.png'.format(rank))
    plt.close(fig)
    del(fig)
    #plt.show()
def  showAttention(input_sentence,  attentions,output_words,rank=0,path='images'):
    h, w = 20, 20        # for raster image
   
    nrows, ncols = len(attentions) * attentions[0].size(0), attentions[0].size(1)  # array of sub-plots
    figsize = [200, 200]     # figure size, inches
    
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols)
    
    # plot simple raster image on each sub-plot
   
    arr= ax.flat if len(attentions) * attentions[0].size(1)*attentions[0].size(0)!=1 else [ax]
    for i, axi in enumerate(arr):
        # i runs from 0 to (nrows*ncols-1)
        # axi is equivalent with ax[rowid][colid]

        # get indices of row/column
        rowid = i // ncols
        colid = i % ncols
        # write row/col indices as axes' title for identification
        # axi.set_title("Row:"+str(rowid)+", Col:"+str(colid))
        
        encodes=input_sentence.split()
        axi.set_xticklabels(['']+ encodes , rotation=90,fontsize=2)
        
        decodes=output_words.split()

        axi.set_yticklabels(   decodes,fontsize=2)
        
        axi.xaxis.set_major_locator(ticker.MultipleLocator(1))
        axi.yaxis.set_major_locator(ticker.MultipleLocator(1))
        
        img = attentions[rowid%len(attentions)][rowid%attentions[0].size(0),colid].cpu().numpy()
        axi.imshow(img, alpha=0.9, cmap=cm.Blues)
    
    plt.tight_layout(True)
    #plt.show()
    plt.savefig(path+'/temp{}.png'.format(rank))
    plt.close('all')

output_words='je suis sûr que la présidence .'
input_sentence='i am sure that the presidency .'

attentions=[attentions[0] for i in range(4)]

showAttention(input_sentence,attentions, output_words)
#attentions=F.softmax(attentions, dim=-1)
showAttention1(input_sentence, output_words, attentions,rank=0,path='images/fusion')
#plt.matshow(attentions.cpu().numpy())

#fig.savefig('images/temp.png', dpi=fig.dpi)


def  showPlot(attentions,rank=0,path='images/plot'):
    def found_fit(x):
        return  x  # Found with symfit.
    size=attentions.size(0)
    attentions=attentions.cpu().cpu().numpy()
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
=======
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 23:55:11 2020

@author: belainine
"""
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.image as mpimg
matplotlib.use('Qt5Agg')
plt.switch_backend('Qt5Agg')
import matplotlib.ticker as ticker
from matplotlib.pyplot import cm
import numpy as np
import torch
import torch.nn.functional as F
def showPlot(points):
    def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / 32) for hid_j in range(32)]
    sinusoid_table=np.array([get_position_angle_vec(pos_i) for pos_i in range(40)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
    points=sinusoid_table
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)

attentions=torch.tensor([[[[[0.0089, 0.0170, 0.0250, 0.0105, 0.0078, 0.0108, 0.0157],
           [0.0085, 0.0168, 0.0254, 0.0114, 0.0081, 0.0119, 0.0171],
           [0.0167, 0.0147, 0.0251, 0.0113, 0.0095, 0.0079, 0.0099],
           [0.0176, 0.0183, 0.0000, 0.0000, 0.0105, 0.0097, 0.0128],
           [0.0213, 0.0097, 0.0155, 0.0000, 0.0130, 0.0088, 0.0000],
           [0.0156, 0.0201, 0.0281, 0.0114, 0.0099, 0.0100, 0.0140],
           [0.0105, 0.0212, 0.0000, 0.0107, 0.0090, 0.0125, 0.0185]],

          [[0.0195, 0.0116, 0.0099, 0.0000, 0.0108, 0.0085, 0.0109],
           [0.0225, 0.0237, 0.0109, 0.0128, 0.0112, 0.0083, 0.0000],
           [0.0336, 0.0195, 0.0135, 0.0147, 0.0182, 0.0100, 0.0089],
           [0.0274, 0.0297, 0.0138, 0.0164, 0.0142, 0.0105, 0.0204],
           [0.0453, 0.0000, 0.0185, 0.0204, 0.0233, 0.0000, 0.0134],
           [0.0269, 0.0388, 0.0242, 0.0000, 0.0234, 0.0197, 0.0241],
           [0.0346, 0.0425, 0.0341, 0.0000, 0.0351, 0.0000, 0.0207]],

          [[0.0157, 0.0231, 0.0328, 0.0381, 0.0311, 0.0358, 0.0342],
           [0.0170, 0.0244, 0.0370, 0.0000, 0.0294, 0.0405, 0.0371],
           [0.0205, 0.0299, 0.0385, 0.0423, 0.0362, 0.0430, 0.0417],
           [0.0180, 0.0249, 0.0000, 0.0483, 0.0299, 0.0430, 0.0000],
           [0.0239, 0.0340, 0.0472, 0.0533, 0.0414, 0.0484, 0.0470],
           [0.0171, 0.0238, 0.0345, 0.0410, 0.0317, 0.0336, 0.0329],
           [0.0000, 0.0241, 0.0344, 0.0000, 0.0261, 0.0284, 0.0283]],

          [[0.0000, 0.0188, 0.0000, 0.0068, 0.0094, 0.0073, 0.0103],
           [0.0109, 0.0209, 0.0129, 0.0075, 0.0114, 0.0000, 0.0127],
           [0.0179, 0.0312, 0.0192, 0.0114, 0.0196, 0.0096, 0.0230],
           [0.0199, 0.0000, 0.0144, 0.0119, 0.0145, 0.0087, 0.0190],
           [0.0180, 0.0267, 0.0163, 0.0116, 0.0202, 0.0099, 0.0225],
           [0.0183, 0.0277, 0.0166, 0.0121, 0.0222, 0.0101, 0.0243],
           [0.0112, 0.0196, 0.0102, 0.0115, 0.0226, 0.0138, 0.0167]],

          [[0.0358, 0.0000, 0.0402, 0.0292, 0.0000, 0.0232, 0.0257],
           [0.0239, 0.0270, 0.0290, 0.0000, 0.0130, 0.0238, 0.0290],
           [0.0099, 0.0101, 0.0000, 0.0083, 0.0000, 0.0235, 0.0211],
           [0.0090, 0.0091, 0.0091, 0.0124, 0.0000, 0.0208, 0.0117],
           [0.0092, 0.0087, 0.0085, 0.0088, 0.0000, 0.0226, 0.0199],
           [0.0113, 0.0107, 0.0000, 0.0078, 0.0000, 0.0141, 0.0332],
           [0.0169, 0.0000, 0.0152, 0.0096, 0.0000, 0.0180, 0.0414]],

          [[0.0213, 0.0100, 0.0112, 0.0145, 0.0343, 0.0148, 0.0100],
           [0.0273, 0.0104, 0.0103, 0.0127, 0.0213, 0.0000, 0.0072],
           [0.0101, 0.0064, 0.0078, 0.0101, 0.0151, 0.0196, 0.0087],
           [0.0000, 0.0080, 0.0085, 0.0107, 0.0202, 0.0200, 0.0096],
           [0.0103, 0.0094, 0.0096, 0.0106, 0.0134, 0.0000, 0.0125],
           [0.0249, 0.0103, 0.0111, 0.0139, 0.0145, 0.0261, 0.0072],
           [0.0000, 0.0000, 0.0111, 0.0133, 0.0151, 0.0224, 0.0070]],

          [[0.0165, 0.0136, 0.0134, 0.0090, 0.0087, 0.0094, 0.0156],
           [0.0198, 0.0083, 0.0084, 0.0082, 0.0077, 0.0069, 0.0113],
           [0.0118, 0.0000, 0.0068, 0.0080, 0.0082, 0.0086, 0.0207],
           [0.0143, 0.0149, 0.0147, 0.0105, 0.0108, 0.0000, 0.0317],
           [0.0148, 0.0081, 0.0000, 0.0084, 0.0086, 0.0096, 0.0282],
           [0.0261, 0.0120, 0.0127, 0.0097, 0.0089, 0.0085, 0.0133],
           [0.0270, 0.0197, 0.0217, 0.0138, 0.0130, 0.0140, 0.0185]],

          [[0.0318, 0.0270, 0.0280, 0.0371, 0.0400, 0.0000, 0.0360],
           [0.0392, 0.0318, 0.0288, 0.0376, 0.0411, 0.0382, 0.0354],
           [0.0454, 0.0373, 0.0353, 0.0375, 0.0388, 0.0409, 0.0373],
           [0.0432, 0.0338, 0.0302, 0.0000, 0.0295, 0.0332, 0.0291],
           [0.0354, 0.0235, 0.0190, 0.0221, 0.0250, 0.0253, 0.0218],
           [0.0362, 0.0257, 0.0203, 0.0222, 0.0244, 0.0260, 0.0217],
           [0.0229, 0.0161, 0.0000, 0.0132, 0.0147, 0.0155, 0.0130]]],


         [[[0.0109, 0.0204, 0.0154, 0.0091, 0.0086, 0.0091, 0.0129],
           [0.0152, 0.0259, 0.0000, 0.0123, 0.0112, 0.0129, 0.0177],
           [0.0342, 0.0276, 0.0298, 0.0314, 0.0000, 0.0260, 0.0287],
           [0.0298, 0.0247, 0.0247, 0.0231, 0.0150, 0.0189, 0.0213],
           [0.0224, 0.0270, 0.0229, 0.0177, 0.0127, 0.0148, 0.0192],
           [0.0153, 0.0310, 0.0000, 0.0135, 0.0133, 0.0133, 0.0194],
           [0.0107, 0.0237, 0.0168, 0.0107, 0.0128, 0.0111, 0.0153]],

          [[0.0170, 0.0000, 0.0127, 0.0105, 0.0253, 0.0202, 0.0203],
           [0.0121, 0.0139, 0.0158, 0.0098, 0.0141, 0.0098, 0.0093],
           [0.0120, 0.0136, 0.0239, 0.0161, 0.0082, 0.0066, 0.0055],
           [0.0208, 0.0209, 0.0358, 0.0294, 0.0132, 0.0109, 0.0086],
           [0.0274, 0.0309, 0.0478, 0.0343, 0.0173, 0.0138, 0.0099],
           [0.0415, 0.0413, 0.0000, 0.0436, 0.0000, 0.0220, 0.0150],
           [0.0248, 0.0279, 0.0445, 0.0367, 0.0000, 0.0138, 0.0106]],

          [[0.0196, 0.0232, 0.0329, 0.0000, 0.0000, 0.0274, 0.0263],
           [0.0204, 0.0280, 0.0339, 0.0351, 0.0000, 0.0271, 0.0257],
           [0.0241, 0.0300, 0.0354, 0.0000, 0.0265, 0.0309, 0.0299],
           [0.0202, 0.0242, 0.0349, 0.0000, 0.0362, 0.0290, 0.0262],
           [0.0233, 0.0302, 0.0438, 0.0449, 0.0364, 0.0000, 0.0321],
           [0.0227, 0.0324, 0.0474, 0.0458, 0.0310, 0.0310, 0.0333],
           [0.0000, 0.0267, 0.0409, 0.0386, 0.0264, 0.0247, 0.0270]],

          [[0.0218, 0.0146, 0.0195, 0.0176, 0.0065, 0.0000, 0.0109],
           [0.0345, 0.0231, 0.0238, 0.0267, 0.0097, 0.0281, 0.0000],
           [0.0328, 0.0227, 0.0269, 0.0288, 0.0000, 0.0280, 0.0198],
           [0.0156, 0.0109, 0.0159, 0.0127, 0.0075, 0.0163, 0.0000],
           [0.0197, 0.0115, 0.0184, 0.0158, 0.0108, 0.0259, 0.0143],
           [0.0118, 0.0103, 0.0185, 0.0126, 0.0102, 0.0128, 0.0131],
           [0.0113, 0.0087, 0.0152, 0.0112, 0.0120, 0.0154, 0.0133]],

          [[0.0325, 0.0372, 0.0355, 0.0302, 0.0140, 0.0000, 0.0333],
           [0.0225, 0.0310, 0.0295, 0.0343, 0.0234, 0.0099, 0.0241],
           [0.0087, 0.0125, 0.0000, 0.0182, 0.0295, 0.0058, 0.0101],
           [0.0161, 0.0239, 0.0221, 0.0300, 0.0348, 0.0085, 0.0185],
           [0.0087, 0.0111, 0.0000, 0.0138, 0.0306, 0.0093, 0.0106],
           [0.0099, 0.0115, 0.0104, 0.0123, 0.0253, 0.0126, 0.0125],
           [0.0267, 0.0304, 0.0281, 0.0220, 0.0137, 0.0170, 0.0268]],

          [[0.0074, 0.0189, 0.0146, 0.0197, 0.0230, 0.0073, 0.0162],
           [0.0058, 0.0110, 0.0085, 0.0115, 0.0134, 0.0068, 0.0093],
           [0.0059, 0.0070, 0.0000, 0.0080, 0.0091, 0.0062, 0.0064],
           [0.0000, 0.0125, 0.0112, 0.0147, 0.0178, 0.0074, 0.0115],
           [0.0094, 0.0000, 0.0108, 0.0139, 0.0169, 0.0081, 0.0112],
           [0.0108, 0.0118, 0.0099, 0.0115, 0.0140, 0.0094, 0.0102],
           [0.0089, 0.0211, 0.0151, 0.0190, 0.0000, 0.0133, 0.0172]],

          [[0.0063, 0.0069, 0.0061, 0.0063, 0.0066, 0.0061, 0.0067],
           [0.0066, 0.0070, 0.0063, 0.0064, 0.0063, 0.0060, 0.0065],
           [0.0083, 0.0089, 0.0073, 0.0086, 0.0092, 0.0089, 0.0084],
           [0.0000, 0.0140, 0.0132, 0.0116, 0.0000, 0.0095, 0.0135],
           [0.0107, 0.0144, 0.0136, 0.0119, 0.0126, 0.0103, 0.0151],
           [0.0144, 0.0204, 0.0188, 0.0000, 0.0170, 0.0135, 0.0206],
           [0.0150, 0.0170, 0.0185, 0.0149, 0.0000, 0.0123, 0.0169]],

          [[0.0367, 0.0276, 0.0309, 0.0349, 0.0381, 0.0393, 0.0377],
           [0.0384, 0.0000, 0.0400, 0.0390, 0.0359, 0.0376, 0.0000],
           [0.0366, 0.0297, 0.0000, 0.0316, 0.0353, 0.0358, 0.0362],
           [0.0309, 0.0284, 0.0258, 0.0242, 0.0272, 0.0298, 0.0293],
           [0.0267, 0.0000, 0.0216, 0.0202, 0.0236, 0.0253, 0.0253],
           [0.0182, 0.0160, 0.0141, 0.0137, 0.0168, 0.0176, 0.0175],
           [0.0261, 0.0194, 0.0182, 0.0193, 0.0255, 0.0258, 0.0258]]]]],
       device='cuda:0')
def showAttention1(input_sentence, output_words, attentions,rank=0,path='images/fusion'):
    # Set up figure with colorbar

 
    
    attention=attentions[0][0].sum(0)



    figsize = [100, 100]     # figure size, inches
    
    fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True)
    
    # plot simple raster image on each sub-plot
   
    arr=  [ax]
    for i, axi in enumerate(arr):
        # i runs from 0 to (nrows*ncols-1)
        # axi is equivalent with ax[rowid][colid]

        # get indices of row/column
        rowid = 0
        colid = 0
        # write row/col indices as axes' title for identification
        # axi.set_title("Row:"+str(rowid)+", Col:"+str(colid))
        
        encodes=input_sentence.split(' ')
        axi.set_xticklabels(['']+ encodes , rotation=90,fontsize=8)
        
        decodes=output_words.split(' ')

        axi.set_yticklabels(   decodes+['<EOS>'],fontsize=8)
        
        axi.xaxis.set_major_locator(ticker.MultipleLocator(1))
        axi.yaxis.set_major_locator(ticker.MultipleLocator(1))
        img = attention.cpu().numpy()
        #img = attention.cpu().numpy()
        axi.imshow(img, alpha=0.9, cmap=cm.Blues)

        #fig.colorbar(cax)
    plt.tight_layout(True)
    plt.savefig(path+'/temp{}.png'.format(rank))
    plt.close(fig)
    del(fig)
    #plt.show()
def  showAttention(input_sentence,  attentions,output_words,rank=0,path='images'):
    h, w = 20, 20        # for raster image
   
    nrows, ncols = len(attentions) * attentions[0].size(0), attentions[0].size(1)  # array of sub-plots
    figsize = [200, 200]     # figure size, inches
    
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols)
    
    # plot simple raster image on each sub-plot
   
    arr= ax.flat if len(attentions) * attentions[0].size(1)*attentions[0].size(0)!=1 else [ax]
    for i, axi in enumerate(arr):
        # i runs from 0 to (nrows*ncols-1)
        # axi is equivalent with ax[rowid][colid]

        # get indices of row/column
        rowid = i // ncols
        colid = i % ncols
        # write row/col indices as axes' title for identification
        # axi.set_title("Row:"+str(rowid)+", Col:"+str(colid))
        
        encodes=input_sentence.split()
        axi.set_xticklabels(['']+ encodes , rotation=90,fontsize=2)
        
        decodes=output_words.split()

        axi.set_yticklabels(   decodes,fontsize=2)
        
        axi.xaxis.set_major_locator(ticker.MultipleLocator(1))
        axi.yaxis.set_major_locator(ticker.MultipleLocator(1))
        
        img = attentions[rowid%len(attentions)][rowid%attentions[0].size(0),colid].cpu().numpy()
        axi.imshow(img, alpha=0.9, cmap=cm.Blues)
    
    plt.tight_layout(True)
    #plt.show()
    plt.savefig(path+'/temp{}.png'.format(rank))
    plt.close('all')

output_words='je suis sûr que la présidence .'
input_sentence='i am sure that the presidency .'

attentions=[attentions[0] for i in range(4)]

showAttention(input_sentence,attentions, output_words)
#attentions=F.softmax(attentions, dim=-1)
showAttention1(input_sentence, output_words, attentions,rank=0,path='images/fusion')
#plt.matshow(attentions.cpu().numpy())

#fig.savefig('images/temp.png', dpi=fig.dpi)


def  showPlot(attentions,rank=0,path='images/plot'):
    def found_fit(x):
        return  x  # Found with symfit.
    size=attentions.size(0)
    attentions=attentions.cpu().cpu().numpy()
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
>>>>>>> d315710b13c03dce30f68164f1767337916b1045
    plt.close('all')
