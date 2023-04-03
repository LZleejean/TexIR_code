'''
@File    :   test_editing.py
@Time    :   2023/02/27 12:10:49
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
from scipy.interpolate import interp1d

import imageio
import numpy as np
import torch
from torch.nn import functional as F
from pyhocon import ConfigFactory
from tensorboardX import SummaryWriter

import utils.general as utils
import utils.plots as plt
from models.loss import IRFLoss
from utils.Cube2Pano import Cube2Pano

class MatEditingRunner():
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

        self.editing_dir = os.path.join(self.plots_dir,'editing-varying')
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
    

        # geo_dir = os.path.join('../',kwargs['exps_folder_name'], 'IRRF-' + kwargs['expname'])
        # if os.path.exists(geo_dir):
        #     timestamps = os.listdir(geo_dir)
        #     timestamp = sorted(timestamps)[-1] # using the newest training result
        # else:
        #     print('No IRF pretrain, please train IRF first!')
        #     exit(0)
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
        self.floor_max_mask = {}
        self.seg_mask = {}
        self.seg_tag = torch.from_numpy(np.array( list(range(0, 49)), np.float32) )
        # self.seg_tag = torch.tensor([46.], dtype=torch.float32)
        self.room_seg_mask = {}
        self.room_meta_scale, self.room_meta_w, self.room_meta_h, self.room_meta_xmin, self.room_meta_zmin, self.room_img= utils.parse_roomseg(\
            os.path.join(os.path.dirname(os.path.dirname(self.conf.get_string('test.path_mesh_open3d'))), 'roomseg'))

    
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


    def change_color(self, frame_per_color, colors, channel=3):
        new_colors = []
        new_colors.append(colors[0])
        for i in range(1, len(colors)):
            if channel==3:
                x = np.array([0, 1])
                r = np.array([colors[i-1][0], colors[i][0]])
                f1 = interp1d(x, r)
                new_r = f1(np.linspace(0, 1, frame_per_color))

                g = np.array([colors[i-1][1], colors[i][1]])
                f2 = interp1d(x, g)
                new_g = f2(np.linspace(0, 1, frame_per_color))

                b = np.array([colors[i-1][2], colors[i][2]])
                f1 = interp1d(x, b)
                new_b = f1(np.linspace(0, 1, frame_per_color))

                for j in range(frame_per_color):
                    # new_colors.append('{},{},{}'.format(new_r[j], new_g[j], new_b[j]))
                    new_colors.append(np.array([new_r[j], new_g[j], new_b[j]], dtype=np.float32))
            elif channel==1:
                x = np.array([0, 1])
                r = np.array([colors[i-1][0], colors[i][0]])
                f1 = interp1d(x, r)
                new_r = f1(np.linspace(0, 1, frame_per_color))

                for j in range(frame_per_color):
                    # new_colors.append('{},{},{}'.format(new_r[j], new_g[j], new_b[j]))
                    new_colors.append(np.array([new_r[j]], dtype=np.float32))
            else:
                print('the channel of interpolate colors must be equal to 1 or 3.')
        return new_colors


    def plot_to_disk_cube(self):

        # index = torch.randint(0, len(self.train_dataset.ids),(1,))
        # index = -1
        for i in range(len(self.train_dataset.ids)):
            cam_to_world = self.train_dataset.extrinsics_list[i].cuda()
            gt_img = self.train_dataset.images_items[i]['color']
            gt_img = gt_img.permute(0,3,1,2).reshape(1,-1, self.cube_lenth, self.cube_lenth)   # shape: [1, 6*c, cube_len, cube_len]
            gt_img = self.cube2pano.ToPano(gt_img)[0].permute(1,2,0)    # shape: [pano_h, pano_w, 3]
            
            derived_id = self.train_dataset.ids[i]
            cam_position = self.train_dataset.cam_position_list[i].cuda()

            res = self.model(cam_to_world, derived_id, cam_position, True)
            pred_img = res['rgb'].cpu().detach()  # shape: (6, h, w, c)
            pred_img = pred_img.permute(0,3,1,2).reshape(1,-1,self.cube_lenth,self.cube_lenth)
            pred_img = self.cube2pano.ToPano(pred_img)[0].permute(1,2,0)    # shape: [pano_h, pano_w, 3]

            pred_r = res['normal'].cpu().detach().expand(-1,-1,-1,3)  # shape: (6, h, w, c)
            pred_r = pred_r.permute(0,3,1,2).reshape(1,-1,self.cube_lenth,self.cube_lenth)
            pred_r = self.cube2pano.ToPano(pred_r)[0].permute(1,2,0)    # shape: [pano_h, pano_w, 3]
            # plt.plot_gbuffer(self.plots_dir, "{}_{}".format(i, self.cur_iter), pred_r, False)

            plt.plot_mat(self.editing_dir, 0, pred_img, "editing_{}".format(i), False)

    def plot_to_disk_varying(self):
        # for varying one pano

        # # for hdrhouse 9
        # albedo_floors = np.array([
        #     [0.52, 0.30, 0.08],
        #     [0.52, 0.00, 0.08],
        #     [0.12, 0.00, 0.58],
        #     [0.12, 0.50, 0.08],
        #     [0.52, 0.30, 0.08]
        # ],dtype=np.float32)
        # albedo_walls = np.array([
        #     [0.81, 0.60, 0.63],
        #     [0.1, 0.00, 0.63],
        #     [0.12, 0.60, 0.0],
        #     [0.81, 0.60, 0.08],
        #     [0.81, 0.60, 0.63]
        # ],dtype=np.float32)

        # for hdrhouse 8
        albedo_floors = np.array([
            [0.56, 0.93, 0.56],
            [0.52, 0.00, 0.08],
            [0.12, 0.00, 0.58],
            [0.12, 0.50, 0.08],
            [0.56, 0.93, 0.56]
        ],dtype=np.float32)
        albedo_walls = np.array([
            [0.48, 0.63, 0.73],
            [0.1, 0.00, 0.63],
            [0.12, 0.60, 0.0],
            [0.81, 0.60, 0.08],
            [0.48, 0.63, 0.73]
        ],dtype=np.float32)

        roughness_floors = np.array([
            [0.01],
            [0.2],
            [0.4],
            [0.6],
            [0.8]
        ],dtype=np.float32)
        frame_per_color = 5
        new_albedo_floors = self.change_color(frame_per_color, albedo_floors)
        new_albedo_walls = self.change_color(frame_per_color, albedo_walls)
        new_roughness_floor = self.change_color(frame_per_color, roughness_floors, 1)
        
        # varying_material = []
        # for i in range(len(new_albedo_floors)):
        #     one_material = {
        #         'albedo_floor': new_albedo_floors[i],
        #         'albedo_wall': new_albedo_walls[i],
        #         'roughness_floor': new_roughness_floor[i]
        #     }
        #     varying_material.append(one_material)

        i = 0
        cam_to_world = self.train_dataset.extrinsics_list[i].cuda()
        gt_img = self.train_dataset.images_items[i]['color']
        gt_img = gt_img.permute(0,3,1,2).reshape(1,-1, self.cube_lenth, self.cube_lenth)   # shape: [1, 6*c, cube_len, cube_len]
        gt_img = self.cube2pano.ToPano(gt_img)[0].permute(1,2,0)    # shape: [pano_h, pano_w, 3]
        
        derived_id = self.train_dataset.ids[i]
        cam_position = self.train_dataset.cam_position_list[i].cuda()
        
        for j in range(len(new_albedo_floors)+len(new_roughness_floor)):
            
            if j < len(new_albedo_floors):
                # continue
                res = self.model(cam_to_world, derived_id, cam_position, True, new_albedo_floors[j], new_albedo_walls[j])
            else:
                new_j = j - len(new_albedo_floors)
                res = self.model(cam_to_world, derived_id, cam_position, True, None, None, new_roughness_floor[new_j])
            pred_img = res['rgb'].cpu().detach()  # shape: (6, h, w, c)
            pred_img = pred_img.permute(0,3,1,2).reshape(1,-1,self.cube_lenth,self.cube_lenth)
            pred_img = self.cube2pano.ToPano(pred_img)[0].permute(1,2,0)    # shape: [pano_h, pano_w, 3]

            pred_r = res['normal'].cpu().detach().expand(-1,-1,-1,3)  # shape: (6, h, w, c)
            pred_r = pred_r.permute(0,3,1,2).reshape(1,-1,self.cube_lenth,self.cube_lenth)
            pred_r = self.cube2pano.ToPano(pred_r)[0].permute(1,2,0)    # shape: [pano_h, pano_w, 3]
            # plt.plot_gbuffer(self.plots_dir, "{}_{}".format(i, self.cur_iter), pred_r, False)

            plt.plot_mat(self.editing_dir, 0, pred_img, "editing_{}".format(j), True)
        
    
        
    def run(self):
        print("testing...")
        # self.plot_to_disk_cube()
        self.plot_to_disk_varying()