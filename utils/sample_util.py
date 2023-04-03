'''
@File    :   sample_util.py
@Time    :   2023/02/27 12:08:01
@Author  :   Zhen Li 
@Contact :   yodlee@mail.nwpu.edu.cn
@Institution: Realsee
@License :   GNU General Public License v2.0
@Desc    :   None
'''


import os
import cv2
import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as tranforms
from urllib3 import Retry


TINY_NUMBER = 1e-6
TINY_TINY_NUMBER = 1e-14

def RadicalInverse(bits):
    #reverse bit
    #高低16位换位置
    bits = (bits << 16) | (bits >> 16)
    #A是5的按位取反
    bits = ((bits & 0x55555555) << 1) | ((bits & 0xAAAAAAAA) >> 1)
    #C是3的按位取反
    bits = ((bits & 0x33333333) << 2) | ((bits & 0xCCCCCCCC) >> 2)
    bits = ((bits & 0x0F0F0F0F) << 4) | ((bits & 0xF0F0F0F0) >> 4)
    bits = ((bits & 0x00FF00FF) << 8) | ((bits & 0xFF00FF00) >> 8)
    return  float(bits) * 2.3283064365386963e-10

def Hammersley(i,N):
    return [float(i)/float(N),RadicalInverse(i)]


def generate_fixed_samples(b, num_sample_dir):
    samples=np.zeros((num_sample_dir,2),dtype=np.float32)
    for i in range(0,num_sample_dir):
        s = Hammersley(i,num_sample_dir)
        samples[i][0] = s[0]
        samples[i][1] = s[1]
    samples = torch.from_numpy(samples).unsqueeze(0).cuda() #size:(batch_size, samples, 2)
    samples = samples.repeat(b, 1, 1).detach()
    # samples[:,:, 0:1] = torch.clamp(samples[:,:,0:1] + torch.rand_like(samples[:,:,0:1])*0.09, 0., 1.)
    shift = torch.rand(b, 1, 2).cuda()
    samples = samples + shift
    index1 = samples > 1.
    samples[index1] = samples[index1]-1.
    index2 = samples < 0.
    samples[index2] = samples[index2] + 1
    samples = torch.clamp(samples, 0+TINY_NUMBER, 1-TINY_NUMBER)    # avoid NAN in roughness backward.
    return samples

# shift = torch.rand(98304, 1, 2).cuda()
def generate_dir(normals, num_sample_dir, samples=None, mode='uniform', roughness=None, pre_mode='Hammersley'):
    """_summary_

    Args:
        normals (torch.float32): [b, 3] (h * w, 3)
        num_sample_dir (int): 1
        mode (str, optional): sampling mode. Defaults to 'uniform'.
        roughness (torch.float32, optional): [b, 1] (h * w, 1). Defaults to None.
        pre_mode (str, optional): pre-sampling mode. Defaults to 'Hammersley'.

    Returns:
        _type_: _description_
    """
    b, c = normals.shape
    normals = normals.unsqueeze(1)
    # compute projection axis
    # x_axis = torch.zeros_like(normals).cuda().expand(b, num_sample_dir, 3)  #size:(batch_size, samples, 3)
    normals = normals.expand(b, num_sample_dir, 3)
    # mask = torch.abs(normals[:,:,0]) > 0.99
    # x_axis[mask, :] = torch.tensor([0., 1., 0.],dtype=torch.float32, device=normals.get_device())
    # x_axis[~mask, :] = torch.tensor([1., 0., 0.],dtype=torch.float32, device=normals.get_device())
    x_axis = torch.where(torch.abs(normals[:,:,0:1]) > 0.99, torch.tensor([0, 1., 0.]).cuda(), torch.tensor([1., 0., 0.]).cuda())

    def norm_axis(x):
        return x / (torch.norm(x, dim=-1, keepdim=True) + TINY_NUMBER)

    normals = norm_axis(normals)
    U = norm_axis(torch.cross(x_axis, normals))
    V = norm_axis(torch.cross( normals, U))

    if pre_mode == "Hammersley":
        samples=np.zeros((num_sample_dir,2),dtype=np.float32)
        for i in range(0,num_sample_dir):
            s = Hammersley(i,num_sample_dir)
            samples[i][0] = s[0]
            samples[i][1] = s[1]
        samples = torch.from_numpy(samples).unsqueeze(0).cuda() #size:(batch_size, samples, 2)
        samples = samples.repeat(b, 1, 1).detach()
        # samples[:,:, 0:1] = torch.clamp(samples[:,:,0:1] + torch.rand_like(samples[:,:,0:1])*0.09, 0., 1.)
        shift = torch.rand(b, 1, 2).cuda()
        samples = samples + shift
        index1 = samples > 1.
        samples[index1] = samples[index1]-1.
        index2 = samples < 0.
        samples[index2] = samples[index2] + 1
        samples = torch.clamp(samples, 0+TINY_NUMBER, 1-TINY_NUMBER)    # avoid NAN in roughness backward.
    else:
        # independent sample
        samples = torch.rand((b, num_sample_dir, 2)).cuda()

    # uniform sample, attention: we generate sampled dir via y as up axis. translate to our coor:
    # phi - np.pi; y = sin((np.pi/2-theta)) =  costheta; y_projected = cos((np.pi/2-theta)) = sintheta
    if mode =='uniform':
        phi = 2 * np.pi * samples[:,:,1:2] - np.pi
        cosTheta = (1.0 - samples[:,:,0:1])
        sinTheta = torch.sqrt(1.0 - cosTheta * cosTheta)
        L = V * (torch.sin(phi) * sinTheta) \
                        + normals * cosTheta \
                        + U * -(torch.cos(phi) * sinTheta)  # [batch, num_samples, 3]
    elif mode =='cosine':
        phi = 2 * np.pi * samples[:,:,1:2] - np.pi
        cosTheta = torch.sqrt(1.0 - samples[:,:,0:1])
        sinTheta = torch.sqrt(1.0 - cosTheta * cosTheta)
        # phi = (torch.rand(1, num_sample_dir, 1).cuda() * 2 * np.pi - np.pi ).repeat(b, 1, 1)
        # Theta = (torch.rand(1, num_sample_dir, 1).cuda() * np.pi/2. ).repeat(b, 1, 1)
        # cosTheta = torch.cos(Theta)
        # sinTheta = torch.sin(Theta)
        L = V * (torch.sin(phi) * sinTheta) \
                        + normals * cosTheta \
                        + U * -(torch.cos(phi) * sinTheta)  # [batch, num_samples, 3]
    elif mode =='importance':
        a = roughness * roughness
        a = a.unsqueeze(1).expand(b, num_sample_dir, 1)

        phi = 2 * np.pi * samples[:,:,1:2] - np.pi
        cosTheta = torch.sqrt( (1.0-samples[:,:,0:1]) /  (1.0 + (a*a-1) * samples[:,:,0:1]) )
        cosTheta = torch.clamp(cosTheta, min=-1.0+TINY_NUMBER,max=1.0-TINY_NUMBER)    # avoid NAN in backward.
        sinTheta = torch.clamp(torch.sqrt(1.0 - cosTheta * cosTheta) , min=-1.0+TINY_NUMBER,max=1.0-TINY_NUMBER)
        L = V * (torch.sin(phi) * sinTheta) \
                        + normals * cosTheta \
                        + U * -(torch.cos(phi) * sinTheta)  # [batch, num_samples, 3]
    

    return L

