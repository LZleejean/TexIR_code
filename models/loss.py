'''
@File    :   loss.py
@Time    :   2023/02/27 12:09:16
@Author  :   Zhen Li 
@Contact :   yodlee@mail.nwpu.edu.cn
@Institution: Realsee
@License :   GNU General Public License v2.0
@Desc    :   None
'''


from utils.sample_util import TINY_NUMBER
import torch
from torch import clamp, nn, unsafe_chunk, unsqueeze
from torch.nn import functional as F
import numpy as np
import os
import cv2
import math

from utils.general import hdr_scale, mse_to_psnr
from models.embedder import *

from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
# from lpips_pytorch import LPIPS
# import lpips

class IRFLoss(nn.Module):
    def __init__(self, loss_type='L1'):
        super().__init__()

        if loss_type == 'L1':
            print('Using L1 loss for comparing radiance!')
            self.rgb_loss = nn.L1Loss(reduction='mean')
        elif loss_type == 'L2':
            print('Using L2 loss for comparing radiance!')
            self.rgb_loss = nn.MSELoss(reduction='mean')
        else:
            raise Exception('Unknown loss_type!')
    
    def forward(self, res):
        """compute the loss between prediction and gt.

        Args:
            gt_irf (torch.float): shape: [b, h*w, 3]
            predicted_irf (torch.float): shape: [b, h*w, 3]
        """
        gt_irf = res['gt']
        predicted_irf = res['pred']
        # predicted_jit = res['pred_jit']
        # return self.rgb_loss(hdr_scale(gt_irf), predicted_irf) + self.rgb_loss(predicted_jit, predicted_irf)
        return self.rgb_loss(hdr_scale(gt_irf), predicted_irf)


class RenderLoss(nn.Module):
    def __init__(self, loss_type='L1', w_gradient=0):
        super().__init__()

        if loss_type == 'L1':
            print('Using L1 loss for comparing re-rendered radiance!')
            self.rgb_loss = nn.L1Loss(reduction='mean')
        elif loss_type == 'L2':
            print('Using L2 loss for comparing re-rendered radiance!')
            self.rgb_loss = nn.MSELoss(reduction='mean')
        elif loss_type == 'psnr':
            print('Using PSNR loss for comparing re-rendered radiance!')
            self.rgb_loss = PSNRLoss()
        elif loss_type == 'ssim':
            print('Using ssim loss for comparing re-rendered radiance!')
            self.rgb_loss = SSIMLoss()
        elif loss_type == 'msssim':
            print('Using ms-ssim loss for comparing re-rendered radiance!')
            self.rgb_loss = MSSSIMLoss()
        else:
            raise Exception('Unknown loss_type!')
        # self.gradient_loss = GradientLoss()
        self.gradient_loss = TVLoss(0.01)
        self.w_gradient = w_gradient
        self.seg_loss = SegLoss(1)
    
    def forward(self, gt_img, preds, gt_mask, floor_max_mask, seg_mask, stage=0, room_seg_mask=None):
        """compute the loss between prediction and gt.

        Args:
            gt_img (torch.float): shape: [6, h, w, 3]
            gt_mask (torch.float): shape: [6, h, w, 1]
            predicted_img (torch.float): shape: [6, h, w, 3]
        """
        if stage == 0:
            empty_mask = preds['empty_mask']
            predicted_img = preds['rgb'] * empty_mask
            direct = self.rgb_loss(hdr_scale(predicted_img*gt_mask ), hdr_scale(gt_img*gt_mask ))

            seg_loss = self.seg_loss(preds['albedo'], None, seg_mask, floor_max_mask, mode=0) * 20

            return direct + seg_loss, seg_loss.item()
        elif stage == 1:
            empty_mask = preds['empty_mask']
            predicted_img = preds['rgb'] * empty_mask

            direct = self.rgb_loss(hdr_scale(gt_img.unsqueeze(0)*floor_max_mask*seg_mask), hdr_scale(predicted_img.unsqueeze(0)*floor_max_mask*seg_mask)) * (predicted_img.shape[1]*predicted_img.shape[2])

            seg_loss = self.seg_loss(preds['roughness'], preds['roughness_womipmap'], seg_mask, floor_max_mask, mode=1, valid_mask=empty_mask) #* floor_max_mask.shape[0]

            return direct + seg_loss, seg_loss.item(), 0.
        elif stage == 2:
            empty_mask = preds['empty_mask']
            predicted_img = preds['rgb'] * empty_mask

            direct = self.rgb_loss(hdr_scale(gt_img.unsqueeze(0)*seg_mask), hdr_scale(predicted_img.unsqueeze(0)*seg_mask)) #* (predicted_img.shape[1]*predicted_img.shape[2])
            
            # we do not consider roomseg, so we set the beta small to avoid to produce absolutely consistent results for different roomseg. Now 0.01, 0.1 when roomseg is enable.
            seg_loss = self.seg_loss(preds['roughness'], None, seg_mask, floor_max_mask, mode=2, valid_mask = empty_mask, room_seg_mask=room_seg_mask) * 0.2 #* floor_max_mask.shape[0]

            return direct + seg_loss , seg_loss.item(), 0

class PSNRLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
    
    def forward(self, gt_img, predicted_img):

        return -mse_to_psnr(self.mse(gt_img.permute(0,3,1,2), predicted_img.permute(0,3,1,2)))

class SSIMLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ssim_module = SSIM(data_range=1., size_average=True, channel=3, nonnegative_ssim=True)
    
    def forward(self, gt_img, predicted_img):
        return 1. - self.ssim_module(gt_img.permute(0,3,1,2), predicted_img.permute(0,3,1,2))

class MSSSIMLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ssim_module = MS_SSIM(data_range=1., size_average=True, channel=3)
    
    def forward(self, gt_img, predicted_img):
        return 1. - self.ssim_module(gt_img.permute(0,3,1,2), predicted_img.permute(0,3,1,2))

        

# class MixLoss(nn.Module):
#     def __init__(self, alpha=0.84):
#         super().__init__()
#         self.ms_ssim_module = MS_SSIM(data_range=1., size_average=True, channel=3)
#         self.L1_loss = nn.L1Loss()
#         self.alpha = 0.84
#     def forward(self, gt_img, predicted_img):



class GradientLoss(nn.Module):
    def __init__(self):
        super(GradientLoss, self).__init__()

    def forward(self, pred, gt):
        assert pred.dim() == gt.dim(), "inconsistent dimensions"
        # horizontal difference
        h_gradient = pred[:,:,:,0:-2] - pred[:,:,:,2:]
        h_gradient_gt = gt[:,:,:,0:-2] -  gt[:,:,:,2:]
        h_gradient_loss = (h_gradient - h_gradient_gt).abs()

        # Vertical difference
        v_gradient = pred[:,:,0:-2,:] - pred[:,:,2:,:]
        v_gradient_gt = gt[:,:,0:-2,:] - gt[:,:,2:,:]
        v_gradient_loss = (v_gradient - v_gradient_gt).abs()
        

        gradient_loss = (h_gradient_loss.mean() + v_gradient_loss.mean())/2

        return gradient_loss


class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight
        self.erode = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        

    def forward(self, x, seg_mask):
        """_summary_

        Args:
            img (_type_): [6, c, h, w]
            seg_mask (torch.float32): [47, 6, h, w, 1]

        Returns:
            _type_: _description_
        """

        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        
        classes, face, h, w, c = seg_mask.shape
        seg_mask = seg_mask.permute(0,1,4,2,3).reshape(-1, c, h, w)  # shape: [47*6, 1, h, w]
        
        w_mask = (-self.erode(-seg_mask)[:,:,:,:w_x-1]).reshape(classes, face, 1, h, w-1)
        h_mask = (-self.erode(-seg_mask)[:,:,:h_x-1,:]).reshape(classes, face, 1, h-1, w)

        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]).unsqueeze(0) * h_mask,2).sum()  
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]).unsqueeze(0) * w_mask,2).sum()

        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size/seg_mask.shape[0]

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]

class SegLoss(nn.Module):
    def __init__(self, SegLoss_weight=1):
        super(SegLoss,self).__init__()
        self.SegLoss_weight = SegLoss_weight
        self.l1 = nn.L1Loss()
        self.erode = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    
    def forward(self, img, img_womipmap, seg_mask, floor_max_mask, mode=0, valid_mask=None, room_seg_mask=None):
        """_summary_

        Args:
            img (torch.float32): [6, h, w, 3/1]
            seg_mask (torch.float32): [47, 6, h, w, 1]

        Returns:
            _type_: _description_
        """
        b, h, w, c = img.shape
        classes, bb, hh, ww, cc = seg_mask.shape
        
        seg_mask = seg_mask.reshape(classes, b, h*w, -1)    # shape: [6, h*w, 1]
        # floor_max_mask = -self.erode(-floor_max_mask.reshape(-1, h, w, 1).permute(0,3,1,2)).permute(0,2,3,1).reshape(-1, bb, hh, ww, 1)
        floor_max_mask = floor_max_mask.reshape(classes, b, h*w, -1)    # shape: [6, h*w, 1]
        
        # print(seg_floor_mask.shape)
        img_segs = img.reshape(b, -1, c).unsqueeze(0).expand(classes, -1, -1, -1)    # shape: [47, 6, h*w, 1]
        # smooth seg region with highlight region
        if mode==1:
            img_womipmap = img_womipmap.reshape(b, -1, c).unsqueeze(0).expand(classes, -1, -1, -1).detach()    # shape: [47, 6, h*w, 1]
            valid_mask = valid_mask.reshape(b, -1, c).unsqueeze(0).expand(classes, -1, -1, -1)    # shape: [47, 6, h*w, 1]
            # print(img_segs.shape)
            # mean_img = torch.mean(torch.sum(img_segs*floor_max_mask, dim=2, keepdim=True) / (torch.sum(floor_max_mask, dim=2, keepdim=True)+TINY_NUMBER) , dim=1, keepdim=True)*6.   # shape: [47, 1, 1, 1]
            # mean_img = torch.sum((img_segs*floor_max_mask).reshape(classes, -1, c), dim=1, keepdim=True) / (torch.sum((floor_max_mask).reshape(classes, -1, 1), dim=1, keepdim=True)+TINY_NUMBER).unsqueeze(1)   # shape: [47, 1, 1, 1]
            # mean_img = torch.sum((img_segs.detach()*floor_max_mask).reshape(classes, -1, c), dim=1, keepdim=True).unsqueeze(1) / (torch.sum((floor_max_mask).reshape(classes, -1, 1), dim=1, keepdim=True)+TINY_NUMBER).unsqueeze(1)   # shape: [47, 1, 1, 1]
            
            # find the max and min value of highlight region. TODO
            num_pixel = torch.sum(floor_max_mask.reshape(classes, -1, 1), dim=1, keepdim=True)
            # max_v = torch.quantile((img_womipmap*floor_max_mask*valid_mask).reshape(classes, -1, c), 1, dim=1, keepdim=True, interpolation='higher')
            # min_v = torch.quantile((img_womipmap*torch.clamp(1-floor_max_mask*valid_mask,0,1)*1e5+img_womipmap*floor_max_mask*valid_mask).reshape(classes, -1, c), 10/bb/hh/ww, dim=1, keepdim=True, interpolation='higher')
            # mean_img = torch.clamp(min_v + (torch.clamp(max_v-min_v, 0.)) * 0.2, max=max_v).unsqueeze(1)
            # mean_img = torch.clamp(min_v, max=max_v).unsqueeze(1)

            mean_img = torch.ones((classes, 1, c)).cuda()
            for i in range(classes):
                if num_pixel[i,0,0].item() == 0:
                    mean_img[i] = 0
                    continue
                # 0.4 for all labels of synthetic data; for real data, 0.6 for wall 45, 0.2 for floor 46, 0.4 for others
                if i == 45:
                    delta = 0.4
                elif i == 46:
                    delta = 0.4
                else:
                    delta = 0.4
                target_v = torch.quantile(img_womipmap.reshape(classes, -1, c)[i][floor_max_mask.reshape(classes, -1, c)[i].bool()], delta, dim=0, keepdim=True)
                mean_img[i] = target_v
                if i == 43:
                    mean_img[i] = torch.ones_like(target_v)*0.8
            mean_img = mean_img.unsqueeze(1)
            # print(min_v.item())
            # print(mean_img.shape)
            # loss = torch.sum(torch.abs(img_segs*seg_mask - mean_img*seg_mask), dim=2, keepdim=True) / (torch.sum(seg_mask, dim=2, keepdim=True)+TINY_NUMBER)
            
            # loss = torch.mean(torch.abs(img_segs*seg_mask - mean_img*seg_mask))
            loss = self.l1(img_segs*(seg_mask-floor_max_mask) * (num_pixel/(num_pixel+TINY_NUMBER)).unsqueeze(1), mean_img*(seg_mask-floor_max_mask) * (num_pixel/(num_pixel+TINY_NUMBER)).unsqueeze(1))

            # loss = torch.sum(torch.abs(img_segs*seg_floor_mask - mean_img*seg_floor_mask).reshape(-1, c), dim=0, keepdim=True) / (torch.sum((seg_floor_mask).reshape(-1, 1), dim=0, keepdim=True)+TINY_NUMBER)
        # smooth seg region
        elif mode==0:
            mean_img = torch.sum((img_segs*seg_mask).reshape(classes, -1, c), dim=1, keepdim=True) / (torch.sum((seg_mask).reshape(classes, -1, 1), dim=1, keepdim=True)+TINY_NUMBER)   # shape: [47, 1, 1]
            # loss = torch.mean(torch.abs(img_segs*seg_mask - mean_img.unsqueeze(1)*seg_mask))
            loss = self.l1(img_segs*seg_mask, mean_img.unsqueeze(1)*seg_mask)
        else:
            room_classes = room_seg_mask.shape[0]
            room_mask = room_seg_mask.reshape(room_classes, b, -1, 1)
            

            mean_img = torch.sum((img_segs.unsqueeze(0)*seg_mask.unsqueeze(0)*room_mask.unsqueeze(1)).reshape(room_classes, classes, -1, c), dim=2, keepdim=True) / (torch.sum((seg_mask.unsqueeze(0)*room_mask.unsqueeze(1)).reshape(room_classes, classes, -1, 1), dim=2, keepdim=True)+TINY_NUMBER)   # shape: [class_room, 47, 1, 1]
            # loss = torch.mean(torch.abs(img_segs*seg_mask - mean_img.unsqueeze(1)*seg_mask))
            loss = self.l1(img_segs.unsqueeze(0)*seg_mask.unsqueeze(0)*room_mask.unsqueeze(1), mean_img.unsqueeze(2)*seg_mask.unsqueeze(0)*room_mask.unsqueeze(1))

        return self.SegLoss_weight * loss

class InvLoss(nn.Module):
    def __init__(self, idr_rgb_weight, eikonal_weight, mask_weight, alpha,
                    sg_rgb_weight, kl_weight, latent_smooth_weight, 
                    brdf_multires=10, loss_type='L1'):
        super().__init__()
        self.idr_rgb_weight = idr_rgb_weight
        self.eikonal_weight = eikonal_weight
        self.mask_weight = mask_weight
        self.alpha = alpha

        self.sg_rgb_weight = sg_rgb_weight
        self.kl_weight = kl_weight
        self.latent_smooth_weight = latent_smooth_weight
        self.brdf_multires = brdf_multires
        
        # if loss_type == 'L1':
        #     print('Using L1 loss for comparing images!')
        #     self.img_loss = nn.L1Loss(reduction='sum')
        # elif loss_type == 'L2':
        #     print('Using L2 loss for comparing images!')
        #     self.img_loss = nn.MSELoss(reduction='sum')
        # else:
        #     raise Exception('Unknown loss_type!')

        # above mode-'sum' produce huge error, so we use mean mode intead.
        if loss_type == 'L1':
            print('Using L1 loss for comparing images!')
            self.img_loss = nn.L1Loss(reduction='mean')
        elif loss_type == 'L2':
            print('Using L2 loss for comparing images!')
            self.img_loss = nn.MSELoss(reduction='mean')
        else:
            raise Exception('Unknown loss_type!')
    
    def get_rgb_loss(self,rgb_values, rgb_gt):
        rgb_loss = self.img_loss(hdr_scale(rgb_values), hdr_scale(rgb_gt))
        return rgb_loss

    def get_eikonal_loss(self, grad_theta):
        if grad_theta.shape[0] == 0:
            return torch.tensor(0.0).cuda().float()

        eikonal_loss = ((grad_theta.norm(2, dim=1) - 1) ** 2).mean()
        return eikonal_loss

    def get_mask_loss(self, sdf_output, network_object_mask, object_mask):
        mask = ~(network_object_mask & object_mask)
        if mask.sum() == 0:
            return torch.tensor(0.0).cuda().float()
        sdf_pred = -self.alpha * sdf_output[mask]
        gt = object_mask[mask].float()
        mask_loss = (1 / self.alpha) * F.binary_cross_entropy_with_logits(
            sdf_pred.squeeze(), gt, reduction='sum') / float(object_mask.shape[0])
        re_weight = 2 - network_object_mask.float()[object_mask].mean()
        return mask_loss * re_weight

    def get_latent_smooth_loss(self, model_outputs):
        d_diff = model_outputs['diffuse_albedo']
        d_rough = model_outputs['roughness'][..., 0]
        d_xi_diff = model_outputs['random_xi_diffuse_albedo']
        d_xi_rough = model_outputs['random_xi_roughness'][..., 0]
        loss = nn.L1Loss()(d_diff, d_xi_diff) + nn.L1Loss()(d_rough, d_xi_rough) 
        return loss 
    
    def kl_divergence(self, rho, rho_hat):
        rho_hat = torch.mean(torch.sigmoid(rho_hat), 0)
        rho = torch.tensor([rho] * len(rho_hat)).cuda()
        return torch.mean(rho * torch.log(rho/rho_hat) + (1 - rho) * torch.log((1 - rho)/(1 - rho_hat)))

    def get_kl_loss(self, model, points):
        loss = 0
        embed_fn, _ = get_embedder(self.brdf_multires)
        values = embed_fn(points)

        for i in range(len(model.brdf_encoder_layer)):
            values = model.brdf_encoder_layer[i](values)

        loss += self.kl_divergence(0.05, values)

        return loss

    def forward(self, model_outputs, rgb_gt, mat_model=None, train_idr=False):

        pred_rgb = model_outputs['rgb']
        sg_rgb_loss = self.get_rgb_loss(pred_rgb, rgb_gt)

        latent_smooth_loss = self.get_latent_smooth_loss(model_outputs)
        kl_loss = self.get_kl_loss(mat_model, model_outputs['position'].reshape(-1,3))

        loss = self.sg_rgb_weight * sg_rgb_loss + \
                self.kl_weight * kl_loss + \
                self.latent_smooth_weight * latent_smooth_loss 

        output = {
            'sg_rgb_loss': sg_rgb_loss,
            'kl_loss': kl_loss,
            'latent_smooth_loss': latent_smooth_loss,
            'loss': loss}

        return output



class NeILFLoss(nn.Module):
    def __init__(self, lambertian_weighting=0.0005, smoothness_weighting=0.0005):
        super().__init__()
        self.l1_loss = nn.L1Loss(reduction='mean')
        self.reg_weight = lambertian_weighting
        self.smooth_weight = smoothness_weighting

    def forward(self, model_outputs, rgb_gt, rgb_grad):
        """_summary_

        Args:
            model_outputs (_type_): _description_
            ground_truth (_type_): _description_

        Returns:
            _type_: _description_
        """
        

        # rendered rgb 
        rgb_values = model_outputs['rgb']
        rgb_loss = self.l1_loss(hdr_scale(rgb_values), hdr_scale(rgb_gt))

        # smoothness smoothness
        rgb_grad = rgb_grad.squeeze(-1)
        brdf_grads = model_outputs['brdf_grad']                # [N, h, w, 3]
        smooth_loss = (brdf_grads.norm(dim=-1) * (-rgb_grad).exp()).mean()

        # lambertian assumption
        roughness = model_outputs['roughness']
        # metallic = model_outputs['metallic']
        # reg_loss = ((roughness - 1).abs() * masks.unsqueeze(1)).sum() / mask_sum + \
        #     ((metallic - 0).abs() * masks.unsqueeze(1)).sum() / mask_sum
        reg_loss = (roughness -1).abs().mean()
    
        loss = rgb_loss + self.smooth_weight * smooth_loss + self.reg_weight * reg_loss

        return loss


class NvDiffRecLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mase_loss = nn.MSELoss(reduction='mean')
        self.albedo_smooth_weight = 0.03

    def forward(self, model_outputs, rgb_gt, iteration):
        """_summary_

        Args:
            model_outputs (_type_): _description_
            ground_truth (_type_): _description_

        Returns:
            _type_: _description_
        """
        

        # rendered rgb 
        img_loss = torch.nn.functional.mse_loss(hdr_scale(model_outputs['rgb']), hdr_scale(rgb_gt))

        # Albedo (k_d) smoothnesss regularizer
        reg_loss = torch.mean(model_outputs['kd_grad']) * self.albedo_smooth_weight * min(1.0, iteration / 100)
    
        loss = img_loss + reg_loss

        return loss


