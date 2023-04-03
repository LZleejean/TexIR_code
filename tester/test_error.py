'''
@File    :   test_error.py
@Time    :   2023/02/27 12:10:54
@Author  :   Zhen Li 
@Contact :   yodlee@mail.nwpu.edu.cn
@Institution: Realsee
@License :   GNU General Public License v2.0
@Desc    :   None
'''


import os
import sys
from datetime import datetime
import time
import itertools
from utils.sample_util import TINY_NUMBER

import imageio
import numpy as np
import torch
from torch.nn import functional as F
from pyhocon import ConfigFactory
from tensorboardX import SummaryWriter

import utils.general as utils
import utils.plots as plt
from models.loss import IRFLoss, SSIMLoss
from utils.Cube2Pano import Cube2Pano

class MatErrorRunner():
    def __init__(self,**kwargs):
        torch.set_default_dtype(torch.float32)
        torch.set_num_threads(1)

        self.conf = ConfigFactory.parse_file(kwargs['conf'])
        self.exps_folder_name = kwargs['exps_folder_name']
        
        self.max_niters = kwargs['max_niters']
        self.GPU_INDEX = kwargs['gpu_index']

        self.expname = 'Mat-' + kwargs['expname']
        
        if kwargs['is_continue'] and kwargs['timestamp'] == 'latest':
            print(os.path.join('../',kwargs['exps_folder_name'],self.expname))
            if os.path.exists(os.path.join('../',kwargs['exps_folder_name'],self.expname)):
                timestamps = os.listdir(os.path.join('../',kwargs['exps_folder_name'],self.expname))
                if (len(timestamps)) == 0:
                    is_continue = False
                    timestamp = None
                else:
                    timestamp = sorted(timestamps)[-1]
                    is_continue = True
            else:
                is_continue = False
                timestamp = None
        else:
            timestamp = kwargs['timestamp']
            is_continue = kwargs['is_continue']

        utils.mkdir_ifnotexists(os.path.join('../',self.exps_folder_name))
        self.expdir = os.path.join('../', self.exps_folder_name, self.expname)
        utils.mkdir_ifnotexists(self.expdir)
        
        self.timestamp = timestamp
        print(timestamp)
        utils.mkdir_ifnotexists(os.path.join(self.expdir, self.timestamp))

        self.plots_dir = os.path.join(self.expdir, self.timestamp, 'plots')
        utils.mkdir_ifnotexists(self.plots_dir)

        self.editing_dir = os.path.join(self.plots_dir,'error')
        utils.mkdir_ifnotexists(self.editing_dir)

        # create checkpoints dirs
        self.checkpoints_path = os.path.join(self.expdir, self.timestamp, 'checkpoints')
        utils.mkdir_ifnotexists(self.checkpoints_path)
        self.model_params_subdir = "ModelParameters"
        self.mat_optimizer_params_subdir = "MatOptimizerParameters"
        self.mat_scheduler_params_subdir = "MatSchedulerParameters"
        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.model_params_subdir))
        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.mat_optimizer_params_subdir))
        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.mat_scheduler_params_subdir))

        # fix random seed
        torch.manual_seed(666)
        torch.cuda.manual_seed(666)
        np.random.seed(666)

        if (not self.GPU_INDEX == 'ignore'):
            os.environ["CUDA_VISIBLE_DEVICES"] = '{0}'.format(self.GPU_INDEX)

        print('shell command : {0}'.format(' '.join(sys.argv)))
        
        print('Loading data ...')
        self.train_dataset = utils.get_class(self.conf.get_string('test.dataset_class'))(
                                self.conf.get_string('test.path_mesh_open3d'), self.conf.get_list('test.pano_img_res'), self.conf.get_float('test.hdr_exposure'))
        print('Finish loading data ...')

        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset,
                                                            batch_size=1,
                                                            shuffle=True
                                                            )
        self.plot_dataloader = torch.utils.data.DataLoader(self.train_dataset,
                                                           batch_size=1,
                                                           shuffle=True
                                                           )
        
        self.model = utils.get_class(self.conf.get_string('test.model_class'))(conf=self.conf, \
            cam_position_list = self.train_dataset.cam_position_list, checkpoint_material=self.plots_dir)
        if torch.cuda.is_available():
            self.model.cuda()
        self.model.eval()

        self.mat_loss = utils.get_class(self.conf.get_string('test.irf_loss_class'))(**self.conf.get_config('render_loss'))
    

        geo_dir = os.path.join('../',kwargs['exps_folder_name'], 'IRRF-' + kwargs['expname'])
        if os.path.exists(geo_dir):
            timestamps = os.listdir(geo_dir)
            timestamp = sorted(timestamps)[-1] # using the newest training result
        else:
            print('No IRF pretrain, please train IRF first!')
            exit(0)
        # # reloading IRRF
        # geo_path = os.path.join(geo_dir, timestamp) + '/checkpoints/ModelParameters/latest.pth'
        # print('Reloading IRRF from: ', geo_path)
        # model = torch.load(geo_path)['model_state_dict']
        # ir = {k.split('network.')[1]: v for k, v in model.items() if 'ir_radiance_network' in k}
        # self.model.ir_radiance_network.load_state_dict(ir)
        # for parm in self.model.ir_radiance_network.parameters():
        #     parm.requires_grad = False
        

        self.n_batches = len(self.train_dataloader)

        self.pano_res = self.conf.get_list('test.pano_img_res')
        self.cube_lenth = int(self.pano_res[1]/4)
        self.cube2pano = Cube2Pano(pano_width=self.pano_res[1], pano_height=self.pano_res[0], cube_lenth=self.cube_lenth)
        self.first_val = True
        
        self.ssim_loss = SSIMLoss()
        self.mse_loss = torch.nn.MSELoss()

    
    def plot_to_disk_material(self):
        
        for i in range(len(self.train_dataset.ids)):
            cam_to_world = self.train_dataset.extrinsics_list[i].cuda()
            gt_img = self.train_dataset.images_items[i]['color']
            gt_img = gt_img.permute(0,3,1,2).reshape(1,-1, self.cube_lenth, self.cube_lenth)   # shape: [1, 6*c, cube_len, cube_len]
            gt_img = self.cube2pano.ToPano(gt_img)[0].permute(1,2,0)    # shape: [pano_h, pano_w, 3]
            
            derived_id = self.train_dataset.ids[i]
            cam_position = self.train_dataset.cam_position_list[i].cuda()

            res = self.model(cam_to_world, derived_id, cam_position, 2)
            pred_albedo = res['albedo'].cpu().detach()  # shape: (6, h, w, c)
            pred_albedo = pred_albedo.permute(0,3,1,2).reshape(1,-1,self.cube_lenth,self.cube_lenth)
            pred_albedo = self.cube2pano.ToPano(pred_albedo)[0].permute(1,2,0)    # shape: [pano_h, pano_w, 3]

            pred_r = res['roughness'].cpu().detach().expand(-1,-1,-1,3)  # shape: (6, h, w, c)
            pred_r = pred_r.permute(0,3,1,2).reshape(1,-1,self.cube_lenth,self.cube_lenth)
            pred_r = self.cube2pano.ToPano(pred_r)[0].permute(1,2,0)    # shape: [pano_h, pano_w, 3]
            plt.plot_gbuffer(self.plots_dir, "albedo_{}".format(i), pred_albedo, False)
            plt.plot_gbuffer(self.plots_dir, "roughness_{}".format(i), pred_r, False)


    def plot_to_disk_cube(self):

        mse_error = 0.0
        psnr_error = 0.0
        ssim_error = 0.0

        for i in range(len(self.train_dataset.ids)):
            cam_to_world = self.train_dataset.extrinsics_list[i].cuda()
            gt_img = self.train_dataset.images_items[i]['color']
            gt_img = gt_img.permute(0,3,1,2).reshape(1,-1, self.cube_lenth, self.cube_lenth)   # shape: [1, 6*c, cube_len, cube_len]
            gt_img = self.cube2pano.ToPano(gt_img)[0].permute(1,2,0)    # shape: [pano_h, pano_w, 3]
            
            derived_id = self.train_dataset.ids[i]
            cam_position = self.train_dataset.cam_position_list[i].cuda()

            res = self.model(cam_to_world, derived_id, cam_position, False)
            pred_img = res['rgb'].cpu().detach()  # shape: (6, h, w, c)
            pred_img = pred_img.permute(0,3,1,2).reshape(1,-1,self.cube_lenth,self.cube_lenth)
            pred_img = self.cube2pano.ToPano(pred_img)[0].permute(1,2,0)    # shape: [pano_h, pano_w, 3]

            ssim_error += 1. - self.ssim_loss(utils.tonemapping(gt_img.unsqueeze(0)), utils.tonemapping(pred_img.unsqueeze(0))).item()
            mse_error += self.mse_loss(utils.tonemapping(gt_img.unsqueeze(0)), utils.tonemapping(pred_img.unsqueeze(0))).item()
            psnr_error += utils.mse_to_psnr(torch.tensor(mse_error)).item()

            plt.plot_mat(self.editing_dir, 0, pred_img, "rendering_{}".format(i), False)
        
        print("re-rendering error: mse: {}, psnr: {}, ssim: {}".format(mse_error/len(self.train_dataset.ids), \
            psnr_error/len(self.train_dataset.ids), (ssim_error)/len(self.train_dataset.ids)))
    
        
    def run(self):
        print("testing...")
        self.plot_to_disk_cube()