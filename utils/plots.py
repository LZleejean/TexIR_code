'''
@File    :   plots.py
@Time    :   2023/02/27 12:08:12
@Author  :   Zhen Li 
@Contact :   yodlee@mail.nwpu.edu.cn
@Institution: Realsee
@License :   GNU General Public License v2.0
@Desc    :   None
'''



import numpy as np
from PIL import Image
import cv2

import torch
import torchvision

from utils.general import hdr_recover

tonemap_img = lambda x: torch.pow(x, 1./2.2)
clip_img = lambda x: torch.clamp(x, min=0., max=1.)



def plot_irf(path, iters, gt_irf, pred_irf, name='rendering'):
    """write irf results when val env points.

    Args:
        path (_type_): _description_
        gt_irf (torch.float): [h, w, 3]
        pred_irf (torch.float): [h, w, 3]
    """

    # gt_irf = clip_img(tonemap_img(gt_irf))
    # pred_irf = clip_img(tonemap_img((pred_irf)))
    gt_irf = gt_irf #* (2**(-6))
    pred_irf = (pred_irf) #* (2**(-6))
    out = torch.stack((pred_irf, gt_irf), dim=0).permute(0,3,1,2)   # shape: (2, c, h, w)
    out = torchvision.utils.make_grid(out,
                                         scale_each=False,
                                         normalize=False,
                                         nrow=1).permute(1,2,0).numpy()   # shape: (h, w, c)
    
    # write the rgb image via opencv-python
    # cv2.imwrite('{0}/rendering_{1}.png'.format(path, iters), out[:,:,::-1]*255)
    cv2.imwrite('{0}/{1}_{2}.exr'.format(path, name, iters), out[:,:,::-1])
    print('saving render img to {0}/{1}_{2}.png'.format(path, name, iters))

def plot_gbuffer(path, iters, pred_gbuffer_mat, is_tonemapped=True):
    """write gbuffer material results.

    Args:
        path (_type_): _description_
        gt_irf (torch.float): [h, w, 3]
        pred_irf (torch.float): [h, w, 3]
    """
    if is_tonemapped:
        pred_gbuffer_mat = clip_img(tonemap_img(pred_gbuffer_mat)).numpy()   # shape: (h, w, c)
        pred_gbuffer_mat = pred_gbuffer_mat.numpy()
        cv2.imwrite('{0}/gbuffer_{1}.png'.format(path, iters), pred_gbuffer_mat[:,:,::-1]*255)
    else:
        #pred_gbuffer_mat = clip_img((pred_gbuffer_mat)).numpy()   # shape: (h, w, c)
        pred_gbuffer_mat = pred_gbuffer_mat.numpy()
        cv2.imwrite('{0}/gbuffer_{1}.exr'.format(path, iters), pred_gbuffer_mat[:,:,::-1])
    print('saving render img to {0}/gbuffer_{1}.exr'.format(path, iters))

def plot_mat(path, iters, pred_gbuffer_mat, name='mat', is_tonemapped=True):
    """write gbuffer material results.

    Args:
        path (_type_): _description_
        gt_irf (torch.float): [h, w, 3]
        pred_irf (torch.float): [h, w, 3]
    """
    if is_tonemapped:
        pred_gbuffer_mat = clip_img(tonemap_img(pred_gbuffer_mat)).numpy()   # shape: (h, w, c)

        cv2.imwrite('{0}/{1}_{2}.png'.format(path, name, iters), pred_gbuffer_mat[:,:,::-1]*255)
    else:
        #pred_gbuffer_mat = clip_img((pred_gbuffer_mat)).numpy()   # shape: (h, w, c)
        pred_gbuffer_mat = pred_gbuffer_mat.numpy()
        cv2.imwrite('{0}/{1}_{2}.hdr'.format(path, name, iters), pred_gbuffer_mat[:,:,::-1])
    print('saving render img to {0}/{1}_{2}.png'.format(path, name, iters))