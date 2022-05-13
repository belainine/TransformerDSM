# -*- coding: utf-8 -*-
"""
Created on Sat May  9 01:01:16 2020

@author: belainine
"""
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np
import torch
enc_slf_attn=torch.tensor([[[[0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
           0.0000],
          [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
           0.0000],
          [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
           0.0000],
          [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
           0.0000],
          [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
           0.0000],
          [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
           0.0000],
          [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
           0.0000]],

         [[0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
           0.0000],
          [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
           0.0000],
          [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
           0.0000],
          [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
           0.0000],
          [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
           0.0000],
          [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
           0.0000],
          [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
           0.0000]],

         [[0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
           0.0000],
          [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
           0.0000],
          [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
           0.0000],
          [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
           0.0000],
          [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
           0.0000],
          [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
           0.0000],
          [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
           0.0000]]],


        [[[0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
           0.0000],
          [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
           0.0000],
          [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
           0.0000],
          [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
           0.0000],
          [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
           0.0000],
          [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
           0.0000],
          [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
           0.0000]],

         [[0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
           0.0000],
          [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
           0.0000],
          [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
           0.0000],
          [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
           0.0000],
          [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
           0.0000],
          [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
           0.0000],
          [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
           0.0000]],

         [[0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
           0.0000],
          [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
           0.0000],
          [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
           0.0000],
          [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
           0.0000],
          [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
           0.0000],
          [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
           0.0000],
          [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
           0.0000]]],


        [[[0.0678, 0.0724, 0.0767, 0.0762, 0.0726, 0.0648, 0.0634, 0.0000,
           0.0000],
          [0.0649, 0.0687, 0.0717, 0.0708, 0.0680, 0.0625, 0.0638, 0.0000,
           0.0000],
          [0.0645, 0.0688, 0.0723, 0.0712, 0.0680, 0.0619, 0.0637, 0.0000,
           0.0000],
          [0.0655, 0.0704, 0.0751, 0.0740, 0.0700, 0.0625, 0.0632, 0.0000,
           0.0000],
          [0.0665, 0.0703, 0.0743, 0.0740, 0.0707, 0.0643, 0.0623, 0.0000,
           0.0000],
          [0.0671, 0.0694, 0.0722, 0.0725, 0.0704, 0.0657, 0.0621, 0.0000,
           0.0000],
          [0.0683, 0.0714, 0.0748, 0.0750, 0.0725, 0.0664, 0.0624, 0.0000,
           0.0000]],

         [[0.0694, 0.0647, 0.0574, 0.0531, 0.0546, 0.0643, 0.0687, 0.0000,
           0.0000],
          [0.0695, 0.0650, 0.0579, 0.0536, 0.0551, 0.0646, 0.0687, 0.0000,
           0.0000],
          [0.0696, 0.0652, 0.0585, 0.0546, 0.0561, 0.0648, 0.0685, 0.0000,
           0.0000],
          [0.0687, 0.0649, 0.0596, 0.0571, 0.0582, 0.0646, 0.0675, 0.0000,
           0.0000],
          [0.0667, 0.0641, 0.0612, 0.0606, 0.0611, 0.0639, 0.0656, 0.0000,
           0.0000],
          [0.0664, 0.0648, 0.0636, 0.0643, 0.0646, 0.0646, 0.0647, 0.0000,
           0.0000],
          [0.0686, 0.0657, 0.0621, 0.0609, 0.0618, 0.0654, 0.0667, 0.0000,
           0.0000]],

         [[0.0558, 0.0499, 0.0521, 0.0611, 0.0706, 0.0719, 0.0663, 0.0000,
           0.0000],
          [0.0596, 0.0524, 0.0525, 0.0591, 0.0667, 0.0688, 0.0664, 0.0000,
           0.0000],
          [0.0630, 0.0567, 0.0558, 0.0598, 0.0638, 0.0644, 0.0656, 0.0000,
           0.0000],
          [0.0624, 0.0601, 0.0611, 0.0648, 0.0653, 0.0616, 0.0641, 0.0000,
           0.0000],
          [0.0553, 0.0575, 0.0633, 0.0706, 0.0721, 0.0656, 0.0627, 0.0000,
           0.0000],
          [0.0532, 0.0534, 0.0590, 0.0678, 0.0736, 0.0703, 0.0635, 0.0000,
           0.0000],
          [0.0531, 0.0505, 0.0551, 0.0649, 0.0736, 0.0725, 0.0648, 0.0000,
           0.0000]]],


        [[[0.0570, 0.0624, 0.0631, 0.0616, 0.0552, 0.0501, 0.0487, 0.0521,
           0.0557],
          [0.0577, 0.0635, 0.0640, 0.0626, 0.0570, 0.0535, 0.0530, 0.0575,
           0.0607],
          [0.0572, 0.0634, 0.0639, 0.0624, 0.0567, 0.0534, 0.0532, 0.0580,
           0.0612],
          [0.0566, 0.0626, 0.0632, 0.0617, 0.0558, 0.0523, 0.0519, 0.0560,
           0.0592],
          [0.0584, 0.0615, 0.0619, 0.0610, 0.0570, 0.0539, 0.0530, 0.0543,
           0.0564],
          [0.0602, 0.0611, 0.0614, 0.0610, 0.0585, 0.0555, 0.0541, 0.0538,
           0.0551],
          [0.0593, 0.0612, 0.0616, 0.0609, 0.0571, 0.0528, 0.0511, 0.0515,
           0.0536]],

         [[0.0696, 0.0635, 0.0576, 0.0544, 0.0590, 0.0673, 0.0715, 0.0665,
           0.0583],
          [0.0693, 0.0634, 0.0579, 0.0547, 0.0592, 0.0669, 0.0705, 0.0657,
           0.0581],
          [0.0690, 0.0631, 0.0578, 0.0548, 0.0591, 0.0665, 0.0699, 0.0650,
           0.0575],
          [0.0679, 0.0624, 0.0579, 0.0557, 0.0590, 0.0655, 0.0690, 0.0644,
           0.0574],
          [0.0658, 0.0618, 0.0589, 0.0577, 0.0596, 0.0640, 0.0670, 0.0636,
           0.0583],
          [0.0642, 0.0610, 0.0595, 0.0591, 0.0600, 0.0622, 0.0635, 0.0606,
           0.0570],
          [0.0663, 0.0616, 0.0585, 0.0570, 0.0593, 0.0635, 0.0653, 0.0613,
           0.0560]],

         [[0.0600, 0.0548, 0.0584, 0.0667, 0.0728, 0.0718, 0.0654, 0.0617,
           0.0606],
          [0.0641, 0.0583, 0.0589, 0.0646, 0.0688, 0.0683, 0.0670, 0.0638,
           0.0606],
          [0.0656, 0.0619, 0.0609, 0.0636, 0.0642, 0.0629, 0.0650, 0.0650,
           0.0619],
          [0.0612, 0.0623, 0.0641, 0.0656, 0.0621, 0.0581, 0.0588, 0.0638,
           0.0648],
          [0.0531, 0.0561, 0.0641, 0.0695, 0.0678, 0.0625, 0.0548, 0.0592,
           0.0656],
          [0.0535, 0.0535, 0.0612, 0.0691, 0.0725, 0.0693, 0.0586, 0.0585,
           0.0632],
          [0.0554, 0.0528, 0.0593, 0.0686, 0.0744, 0.0723, 0.0617, 0.0593,
           0.0617]]]], device='cuda:0')
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
def  showUtteranceAttention(output_words,  attentions,rank=0,path='images/utterance'):
    h, w = 20, 20        # for raster image
    utt_length=attentions[0].size()[1]
    input_sentence=' '.join(['utterance_{}'.format(i) for i in range(utt_length)])
    nrows, ncols = len(attentions) , attentions[0].size(1)  # array of sub-plots
    figsize = [250, 250]     # figure size, inches
    
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize,gridspec_kw={'wspace':1, 'hspace':1})
    
    # plot simple raster image on each sub-plot
    im=0
    arr= ax.flat if len(attentions) * attentions[0].size(1)*attentions[0].size(0)!=1 else [ax]
    for i, axi in enumerate(arr):
        # i runs from 0 to (nrows*ncols-1)
        # axi is equivalent with ax[rowid][colid]

        # get indices of row/column
        rowid = i // ncols
        colid = i % ncols
        # write row/col indices as axes' title for identification
        # axi.set_title("Row:"+str(rowid)+", Col:"+str(colid))
        
        encodes=output_words.split()
        axi.set_xticklabels(['']+ encodes , rotation=90,fontsize=25)
        
        decodes=input_sentence.split()

        axi.set_yticklabels( ['']+  decodes,fontsize=25)
        axi.set_title('layer {}, and head {}'.format(rowid,int(colid) ),fontsize=25)
        axi.xaxis.set_major_locator(ticker.MultipleLocator(1))
        axi.yaxis.set_major_locator(ticker.MultipleLocator(1))
        img = attentions[rowid][0,colid].cpu().numpy()
        axi.margins(x=0.5, y=0.5)
        im=axi.imshow(img, alpha=0.9)
        axins = inset_axes(axi,
                   width="100%",  # width = 5% of parent_bbox width
                   height="5%",  # height : 50%
                   #loc="lower left",#'lower left',
                   bbox_to_anchor=(0.0, 0.5, 1, 1),
                   bbox_transform=axi.transAxes,
                   borderpad=0,
                   )
        axins.xaxis.set_ticks_position("bottom")
        cbar=fig.colorbar(im, cax=axins,  orientation='horizontal',ticks=[1, 2, 3])    
        cbar.ax.set_xticklabels(['0', '50', '100'],fontsize=25)
    #plt.show()
    plt.savefig(path+'/temp{}.png'.format(rank))
    plt.close('all')
def  showSelfAttention(input_sentence,  attentions,output_words,rank=0,path='images/self'):
    h, w = 40, 40        # for raster image

    
    nb_utterances= attentions[0].size(1)
    input_sentence=[s.strip() for s in input_sentence.split('__eou__')]
    output_words=[s.strip() for s in output_words.split('__eou__')]
 
    nb_b=attentions[0].size(0)
    n_layers=len(attentions)
    n_head=attentions[0].size(2)
    nrows, ncols = n_layers*nb_b , nb_utterances*n_head  # array of sub-plots
    figsize = [250, 250]     # figure size, inches
    
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize,gridspec_kw={'wspace':1, 'hspace':1})
    
    # plot simple raster image on each sub-plot
    
    arr= ax.flat if len(attentions) *attentions[0].size(0)!=1 else [ax]
    for i, axi in enumerate(arr):
        # i runs from 0 to (nrows*ncols-1)
        # axi is equivalent with ax[rowid][colid]

        # get indices of row/column
        rowid = i // ncols
        colid = i % ncols
        
        # write row/col indices as axes' title for identification
        # axi.set_title("Row:"+str(rowid)+", Col:"+str(colid))
        
        encodes=input_sentence[colid%nb_utterances].split()
        axi.set_xticklabels(['']+ encodes , rotation=90,fontsize=25)
        
        decodes=output_words[0].split()

        axi.set_yticklabels( ['']+  decodes,fontsize=25)
        
        axi.xaxis.set_major_locator(ticker.MultipleLocator(1))
        axi.yaxis.set_major_locator(ticker.MultipleLocator(1))
        
        axi.set_title('layer {}, dim {} and head {}'.format(rowid//nb_b,colid%nb_utterances,int(colid/nb_utterances) ),fontsize=25)


        img = attentions[rowid%n_layers][(rowid)%nb_b,colid%nb_utterances,int(colid/nb_utterances)].cpu().numpy()
        im=axi.imshow(img, alpha=0.9)
        axins = inset_axes(axi,
                   width="100%",  # width = 5% of parent_bbox width
                   height="5%",  # height : 50%
                   #loc="lower left",#'lower left',
                   bbox_to_anchor=(0.0, 0.5, 1, 1),
                   bbox_transform=axi.transAxes,
                   borderpad=0,
                   )
        axins.xaxis.set_ticks_position("bottom")
        cbar=fig.colorbar(im, cax=axins,  orientation='horizontal',ticks=[1, 2, 3])    
        cbar.ax.set_xticklabels(['0', '50', '100'],fontsize=25)
    #plt.tight_layout(True)
    #plt.show()
    plt.savefig(path+'/temp{}.png'.format(rank))
    plt.close('all')
if __name__ == '__main__':
    input_sentence='mr cashman , i am grateful for that information . __eou__ mr cashman , i am grateful for that information . __eou__ mr cashman , i am grateful for that information . __eou__ mr cashman , i am grateful for that information .'
    output_words='monsieur cashman , je vous remercie de cette information . __eou__ monsieur cashman , je vous remercie de cette information . __eou__ monsieur cashman , je vous remercie de cette information .'
    
    
    attentions=[enc_slf_attn.repeat(4,1,1,1,1) for i in range(5)]
    print('attentions',attentions[0].size())
    showSelfAttention(input_sentence,  attentions,output_words,rank=0)
    
    attentions=[attn.sum(4).transpose(1,2) for attn in attentions]
    print('attentions',attentions[0].size())
    input_sentence=' '.join(['utterance' for i in range(attentions[0].size()[1])])
    output_words='monsieur cashman , je vous remercie de cette information .'
    showUtteranceAttention(output_words,  attentions,rank=0,path='images/utterance')