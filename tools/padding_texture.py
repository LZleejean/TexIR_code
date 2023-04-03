import cv2
import numpy as np
from scipy import ndimage
import torch
import torch.nn.functional as F
import os

def padding_index_texture(path):
    img = cv2.imread(path,-1)
    img = np.asarray(img, np.float32)
    h, w, c = img.shape

    intensity = img[:,:,0] + img[:,:,1] + img[:,:, 2]

    mask = np.asarray(intensity==0.0, dtype=np.uint8)

    # # 3 is better than 7
    # kernel = np.ones((3,3), np.uint8)
    # mask_large = cv2.dilate(mask, kernel)
    mask_large = mask

    distance, indices = ndimage.distance_transform_edt(mask_large, return_indices=True)
    indices = torch.from_numpy(indices).permute(1,2,0).reshape(-1,2)

    # indexes = np.argwhere(distance==0)
    # indexes = distance == 0

    img_torch = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)

    uv_init = torch.zeros((h*w, 2), dtype=torch.float32)

    permute = [1,0]
    uv_init[mask_large.reshape(-1)] = indices[mask_large.reshape(-1)][:,permute].float()/torch.tensor([w, h]).unsqueeze(0) * 2. -1.

    uv_init = uv_init.reshape(h, w, 2).unsqueeze(0)

    res = F.grid_sample(img_torch, uv_init, mode='nearest')[0].permute(1,2,0).numpy()
    res = res * np.asarray(mask,np.float32)[:,:,np.newaxis] + img * np.asarray(1-mask, np.float32)[:,:,np.newaxis]

    # kernel = np.ones((3,3), np.uint16)
    # res = cv2.dilate(res, kernel)
    res = np.asarray(res, np.uint16)
    cv2.imwrite(path.replace('.png','_padding.png'), res)
    # cv2.imwrite(path.replace("0_irr_texture", 't'), res)
    # denoise
    # cmd = "/home/SecondDisk/Code/opensource/oidn/build/oidnDenoise --hdr {} -o {}".format(path.replace("0_irr_texture", 't'), path.replace("0_irr_texture", '0_irr_texture_denoised_padding_1'))
    # os.system(cmd)

def padding_texture(path):
    img = cv2.imread(path,-1)
    img = np.asarray(img, np.float32)
    h, w, c = img.shape

    intensity = img[:,:,0] + img[:,:,1] + img[:,:, 2]

    mask = np.asarray(intensity==0.0, dtype=np.uint8)

    # # 3 is better than 7
    # kernel = np.ones((3,3), np.uint8)
    # mask_large = cv2.dilate(mask, kernel)
    mask_large = mask

    distance, indices = ndimage.distance_transform_edt(mask_large, return_indices=True)
    indices = torch.from_numpy(indices).permute(1,2,0).reshape(-1,2)

    # indexes = np.argwhere(distance==0)
    # indexes = distance == 0

    img_torch = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)

    uv_init = torch.zeros((h*w, 2), dtype=torch.float32)

    permute = [1,0]
    uv_init[mask_large.reshape(-1)] = indices[mask_large.reshape(-1)][:,permute].float()/torch.tensor([w, h]).unsqueeze(0) * 2. -1.

    uv_init = uv_init.reshape(h, w, 2).unsqueeze(0)

    res = F.grid_sample(img_torch, uv_init, mode='nearest')[0].permute(1,2,0).numpy()
    res = res * np.asarray(mask,np.float32)[:,:,np.newaxis] + img * np.asarray(1-mask, np.float32)[:,:,np.newaxis]

    # kernel = np.ones((3,3), np.uint16)
    # res = cv2.dilate(res, kernel)

    cv2.imwrite(path.replace("0_irr_texture", 't'), res)
    # denoise
    cmd = "/home/SecondDisk/Code/opensource/oidn/build/oidnDenoise --hdr {} -o {}".format(path.replace("0_irr_texture", 't'), path.replace("0_irr_texture", '0_irr_texture_denoised_padding'))
    os.system(cmd)

if __name__=="__main__":
    padding_texture("/home/SecondDisk/test_data/galois_model/inverse/hdrhouse/customHouse/vrproc/hdr_texture/0_irr_texture.hdr")
    # padding_index_texture("/home/SecondDisk/test_data/galois_model/inverse/hdrhouse/customHouse/vrproc/hdr_texture/source/output0000_padding.png")