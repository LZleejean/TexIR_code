import os
from glob import glob
import torch
import numpy as np
import cv2

def mkdir_ifnotexists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def get_class(kls):
    parts = kls.split('.')
    module = ".".join(parts[:-1])
    m = __import__(module)
    for comp in parts[1:]:
        m = getattr(m, comp)
    return m


def glob_imgs(path):
    imgs = []
    for ext in ['*.png', '*.jpg', '*.JPEG', '*.JPG', '*.exr']:
        imgs.extend(glob(os.path.join(path, ext)))
    return imgs


def split_input(model_input, total_pixels):
    '''
     Split the input to fit Cuda memory for large resolution.
     Can decrease the value of n_pixels in case of cuda out of memory error.
     '''
    # n_pixels = 20000
    n_pixels = 2000
    split = []
    for i, indx in enumerate(torch.split(torch.arange(total_pixels).cuda(), n_pixels, dim=0)):
        data = model_input.copy()
        data['uv'] = torch.index_select(model_input['uv'], 1, indx)
        data['object_mask'] = torch.index_select(model_input['object_mask'], 1, indx)
        split.append(data)
    return split


def merge_output(res, total_pixels, batch_size):
    ''' Merge the split output. '''

    model_outputs = {}
    for entry in res[0]:
        if res[0][entry] is None:
            continue
        if len(res[0][entry].shape) == 1:
            model_outputs[entry] = torch.cat([r[entry].reshape(batch_size, -1, 1) for r in res],
                                             1).reshape(batch_size * total_pixels)
        else:
            model_outputs[entry] = torch.cat([r[entry].reshape(batch_size, -1, r[entry].shape[-1]) for r in res],
                                             1).reshape(batch_size * total_pixels, -1)

    return model_outputs


def hdr_scale(img, base=np.e):
	# if scale_model=="e":
	# 	return torch.log1p(img)
	# tensor_two = torch.full_like(img,2)
	# return tensor_two.min(torch.log10(1+img))-1
    return torch.log(img+1) / np.math.log(base)

def hdr_recover(img, base=np.e):
    return torch.pow(base, img)-1

def mse_to_psnr(mse):
  """Compute PSNR given an MSE (we assume the maximum pixel value is 1)."""
  return -10. / np.log(10.) * torch.log(mse)

def psnr_to_mse(psnr):
  """Compute MSE given a PSNR (we assume the maximum pixel value is 1)."""
  return torch.exp(-0.1 * np.log(10.) * psnr)

def tonemapping(img):
    """_summary_

    Args:
        img (torch.tensor): [b, h, w, c]
    """
    return torch.clamp(img**(1/2.2), 0., 1.)


def get_mip_level(n):
    count = 0
    while not (n & 1 or n==1):
        n = n>>1
        count = count + 1
    return count

def rgb_to_intensity(tensor, dim=-1):
    """_summary_

    Args:
        tensor (torch.float32): shape: [h, w, 3] / [b,h,w,3]

    Returns:
        _type_: _description_
    """
    if dim== -1:
        return 0.29900 * tensor[...,0:1] + 0.58700 * tensor[...,1:2] + 0.11400 * tensor[...,2:3]
    # assume that the shape of input is [b,h,w,3]
    elif dim == 0:
        return 0.29900 * tensor[0:1,...] + 0.58700 * tensor[1:2,...] + 0.11400 * tensor[2:3,...]
    elif dim ==1:
        return 0.29900 * tensor[:,0:1,...] + 0.58700 * tensor[:,1:2,...] + 0.11400 * tensor[:,2:3,...]
    elif dim ==2:
        return 0.29900 * tensor[:,:,0:1,...] + 0.58700 * tensor[:,:,1:2,...] + 0.11400 * tensor[:,:,2:3,...]


def parse_roomseg(path):
    with open(os.path.join(path, 'originOccupancyGrid_f0.meta'), 'r') as f:
        first_line = f.readline()
    
    _, _w, _h, x_min, z_min = first_line.strip().split(" ")

    roomsegs = cv2.imread(os.path.join(path, 'roomSegs_uchar_f0.png'))
    roomsegs = np.asarray(roomsegs, np.float32)
    roomsegs = torch.from_numpy(roomsegs)[:,:,0:1].unsqueeze(0).permute(0, 3, 1, 2)   # shape: [1, 1, h, w]

    return float(_), float(_w), float(_h), float(x_min), float(z_min), roomsegs


def scale_compute(gt, prediction):
    scale, _ = torch.lstsq(gt.flatten().unsqueeze(1), prediction.flatten().unsqueeze(1))
    return scale[0, 0].clone().detach()