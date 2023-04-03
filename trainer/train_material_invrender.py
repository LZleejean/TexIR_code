'''
@File    :   train_material_invrender.py
@Time    :   2023/02/27 12:11:42
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

class MatInvTrainRunner():
    def __init__(self,**kwargs):
        torch.set_default_dtype(torch.float32)
        torch.set_num_threads(1)

        self.conf = ConfigFactory.parse_file(kwargs['conf'])
        self.exps_folder_name = kwargs['exps_folder_name']
        self.train_batch_size = self.conf.get_int('train.batch_size')
        self.nepochs = self.conf.get_int('train.mat_epoch')
        
        self.max_niters = kwargs['max_niters']
        self.GPU_INDEX = kwargs['gpu_index']

        self.expname = 'Mat-' + kwargs['expname']
        
        if kwargs['is_continue'] and kwargs['timestamp'] == 'latest':
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
        self.timestamp = '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.now())
        utils.mkdir_ifnotexists(os.path.join(self.expdir, self.timestamp))

        self.plots_dir = os.path.join(self.expdir, self.timestamp, 'plots')
        utils.mkdir_ifnotexists(self.plots_dir)

        # create checkpoints dirs
        self.checkpoints_path = os.path.join(self.expdir, self.timestamp, 'checkpoints')
        utils.mkdir_ifnotexists(self.checkpoints_path)
        self.model_params_subdir = "ModelParameters"
        self.mat_optimizer_params_subdir = "MatOptimizerParameters"
        self.mat_scheduler_params_subdir = "MatSchedulerParameters"
        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.model_params_subdir))
        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.mat_optimizer_params_subdir))
        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.mat_scheduler_params_subdir))

        print('Write tensorboard to: ', os.path.join(self.expdir, self.timestamp))
        self.writer = SummaryWriter(os.path.join(self.expdir, self.timestamp))

        os.system("""cp -r {0} "{1}" """.format(kwargs['conf'], os.path.join(self.expdir, self.timestamp, 'runconf.conf')))

        # fix random seed
        torch.manual_seed(666)
        torch.cuda.manual_seed(666)
        np.random.seed(666)

        if (not self.GPU_INDEX == 'ignore'):
            os.environ["CUDA_VISIBLE_DEVICES"] = '{0}'.format(self.GPU_INDEX)

        print('shell command : {0}'.format(' '.join(sys.argv)))
        
        print('Loading data ...')
        self.train_dataset = utils.get_class(self.conf.get_string('train.dataset_class'))(
                                self.conf.get_string('train.path_mesh_open3d'), self.conf.get_list('train.pano_img_res'), self.conf.get_float('train.hdr_exposure'))
        print('Finish loading data ...')

        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset,
                                                            batch_size=1,
                                                            shuffle=True
                                                            )
        self.plot_dataloader = torch.utils.data.DataLoader(self.train_dataset,
                                                           batch_size=1,
                                                           shuffle=True
                                                           )
        
        self.model = utils.get_class(self.conf.get_string('train.model_class'))(conf=self.conf, \
            ids=self.train_dataset.ids, extrinsics = self.train_dataset.extrinsics_list, optim_cam=self.conf.get_bool('train.optim_cam'))
        if torch.cuda.is_available():
            self.model.cuda()

        self.mat_loss = utils.get_class(self.conf.get_string('train.irf_loss_class'))(**self.conf.get_config('loss'))
        if self.conf.get_bool('train.optim_cam'):
            self.mat_optimizer = torch.optim.Adam([
                {'params': self.model.materials.parameters(), 'lr': self.conf.get_float('train.mat_learning_rate')},
                {'params': self.model.param_extrinsics.parameters(), 'lr': self.conf.get_float('train.mat_learning_rate')*0.1}
            ])
        else:
            self.mat_optimizer = torch.optim.Adam(self.model.parameters(),
                                                    lr=self.conf.get_float('train.mat_learning_rate'))
            # self.mat_optimizer = torch.optim.Adam([self.model.materials_a, self.model.materials_r],
            #                                         lr=self.conf.get_float('train.mat_learning_rate'))
        self.mat_scheduler = torch.optim.lr_scheduler.StepLR(self.mat_optimizer,
                                                self.conf.get_int('train.mat_sched_step', default=100),
                                                gamma=self.conf.get_float('train.mat_sched_factor', default=0.0))


        self.start_epoch = 0
        if is_continue:
            old_checkpnts_dir = os.path.join(self.expdir, timestamp, 'checkpoints')

            print('Loading pretrained model: ', os.path.join(
                    old_checkpnts_dir, self.model_params_subdir, str(kwargs['checkpoint']) + ".pth"))

            saved_model_state = torch.load(
                os.path.join(old_checkpnts_dir, self.model_params_subdir, str(kwargs['checkpoint']) + ".pth"))
            self.model.load_state_dict(saved_model_state["model_state_dict"])
            self.start_epoch = saved_model_state['epoch']

            data = torch.load(
                os.path.join(old_checkpnts_dir, self.mat_optimizer_params_subdir, str(kwargs['checkpoint']) + ".pth"))
            self.illum_optimizer.load_state_dict(data["optimizer_state_dict"])

            data = torch.load(
                os.path.join(old_checkpnts_dir, self.mat_scheduler_params_subdir, str(kwargs['checkpoint']) + ".pth"))
            self.illum_scheduler.load_state_dict(data["scheduler_state_dict"])


        # # reloading pretrained radiance field
        # geo_dir = os.path.join('../',kwargs['exps_folder_name'], 'IRRF-' + kwargs['expname'])
        # if os.path.exists(geo_dir):
        #     timestamps = os.listdir(geo_dir)
        #     timestamp = sorted(timestamps)[-1] # using the newest training result
        # else:
        #     print('No IRF pretrain, please train IRF first!')
        #     exit(0)

        # # reloading IRF
        # geo_path = os.path.join(geo_dir, timestamp) + '/checkpoints/ModelParameters/latest.pth'
        # print('Reloading IRF from: ', geo_path)
        # model = torch.load(geo_path)['model_state_dict']
        # ir = {k.split('network.')[1]: v for k, v in model.items() if 'incident_radiance_network' in k}
        # self.model.incident_radiance_network.load_state_dict(ir)
        # for parm in self.model.incident_radiance_network.parameters():
        #     parm.requires_grad = False

        # # reloading IRRF
        # geo_path = os.path.join(geo_dir, timestamp) + '/checkpoints/ModelParameters/latest.pth'
        # print('Reloading IRRF from: ', geo_path)
        # model = torch.load(geo_path)['model_state_dict']
        # ir = {k.split('network.')[1]: v for k, v in model.items() if 'ir_radiance_network' in k}
        # self.model.ir_radiance_network.load_state_dict(ir)
        # for parm in self.model.ir_radiance_network.parameters():
        #     parm.requires_grad = False
        
        self.n_batches = len(self.train_dataloader)
        self.plot_freq = self.conf.get_int('train.plot_freq')
        self.ckpt_freq = self.conf.get_int('train.ckpt_freq')
        self.pano_res = self.conf.get_list('train.pano_img_res')
        self.cube_lenth = int(self.pano_res[1]/4)
        self.cube2pano = Cube2Pano(pano_width=self.pano_res[1], pano_height=self.pano_res[0], cube_lenth=self.cube_lenth)
        self.first_val = True

        self.ssim_loss = SSIMLoss()
        self.mse_loss = torch.nn.MSELoss()



    def save_checkpoints(self, epoch):
        # torch.save(
        #     {"epoch": epoch, "model_state_dict": self.model.state_dict()},
        #     os.path.join(self.checkpoints_path, self.model_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "model_state_dict": self.model.state_dict()},
            os.path.join(self.checkpoints_path, self.model_params_subdir, "latest.pth"))

        # torch.save(
        #     {"epoch": epoch, "optimizer_state_dict": self.mat_optimizer.state_dict()},
        #     os.path.join(self.checkpoints_path, self.mat_optimizer_params_subdir, str(epoch) + ".pth"))
        # torch.save(
        #     {"epoch": epoch, "optimizer_state_dict": self.mat_optimizer.state_dict()},
        #     os.path.join(self.checkpoints_path, self.mat_optimizer_params_subdir, "latest.pth"))

        # torch.save(
        #     {"epoch": epoch, "scheduler_state_dict": self.mat_scheduler.state_dict()},
        #     os.path.join(self.checkpoints_path, self.mat_scheduler_params_subdir, str(epoch) + ".pth"))
        # torch.save(
        #     {"epoch": epoch, "scheduler_state_dict": self.mat_scheduler.state_dict()},
        #     os.path.join(self.checkpoints_path, self.mat_scheduler_params_subdir, "latest.pth"))
    
    def plot_to_disk_material(self):
        self.model.eval()
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

        self.model.train()

    def plot_to_disk_cube(self, stage=0):
        self.model.eval()
        # index = torch.randint(0, len(self.train_dataset.ids),(1,))
        index = -1

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

            res = self.model(cam_to_world, derived_id, cam_position, stage)
            pred_img = res['rgb'].cpu().detach()  # shape: (6, h, w, c)
            pred_img = pred_img.permute(0,3,1,2).reshape(1,-1,self.cube_lenth,self.cube_lenth)
            pred_img = self.cube2pano.ToPano(pred_img)[0].permute(1,2,0)    # shape: [pano_h, pano_w, 3]

            pred_r = res['normal'].cpu().detach().expand(-1,-1,-1,3)  # shape: (6, h, w, c)
            pred_r = pred_r.permute(0,3,1,2).reshape(1,-1,self.cube_lenth,self.cube_lenth)
            pred_r = self.cube2pano.ToPano(pred_r)[0].permute(1,2,0)    # shape: [pano_h, pano_w, 3]
            # plt.plot_gbuffer(self.plots_dir, "{}_{}".format(i, self.cur_iter), pred_r, False)
            plt.plot_irf(
                self.plots_dir,
                "{}_{}".format(i, self.cur_iter),
                gt_img,
                pred_img
            )

            ssim_error += 1. - self.ssim_loss(utils.tonemapping(gt_img.unsqueeze(0)), utils.tonemapping(pred_img.unsqueeze(0))).item()
            mse_error += self.mse_loss(utils.tonemapping(gt_img.unsqueeze(0)), utils.tonemapping(pred_img.unsqueeze(0))).item()
            psnr_error += utils.mse_to_psnr(torch.tensor(mse_error)).item()

        print("re-rendering error: mse: {}, psnr: {}, ssim: {}".format(mse_error/len(self.train_dataset.ids), \
            psnr_error/len(self.train_dataset.ids), (ssim_error)/len(self.train_dataset.ids)))
        # for index in range(len(self.model.materials_albedo)):
        #     plt.plot_mat(self.plots_dir, self.cur_iter, self.model.materials_albedo[index].cpu().detach(), "mat_albedo{}".format(index), False)
        # for index in range(len(self.model.materials_roughness)):
        #     plt.plot_mat(self.plots_dir, self.cur_iter, self.model.materials_roughness[index].cpu().detach(), "mat_roughness{}".format(index), False)

        # plt.plot_mat(self.plots_dir, self.cur_iter, self.model.materials_a.cpu().detach()[:,:,0:3], "mat_albedo{}".format(index), False)
        # plt.plot_mat(self.plots_dir, self.cur_iter, self.model.materials_r.cpu().detach()[:,:,0:1], "mat_roughness{}".format(index), False)

        self.model.train()
        self.first_val = False
    
        
    def run(self):
        print("training...")
        self.cur_iter = self.start_epoch * len(self.train_dataloader)

        for epoch in range(self.start_epoch, self.nepochs + 1):

            for data_index, (gt_item) in enumerate(self.train_dataloader):
                one_batch_start_time = time.time()
                self.model.train()

                gt_color = gt_item['color'].float().cuda()
                h, w, c = gt_color.shape[-3:]
                gt_color = gt_color.reshape(-1, h, w, c)

                gt_mask = gt_item['mask'].float().cuda()
                gt_mask = gt_mask.reshape(-1, h, w, 1)

                gt_extrinsic = gt_item['cam_to_world'].float().cuda()
                id = gt_item['id']
                cam_position = gt_item['cam_position'].float().cuda()
                
                preds = self.model(gt_extrinsic[0], id[0], cam_position[0], 0)
                
                loss_output = self.mat_loss(preds, gt_color, self.model.material_network)

                loss = loss_output['loss']
                rgb_loss = loss_output['sg_rgb_loss']
                kl_loss = loss_output['kl_loss']
                # update mat
                self.mat_optimizer.zero_grad() 
                loss.backward()
                self.mat_optimizer.step()

                if epoch % 1 == 0:
                    print('{0} [{1}] ({2}/{3}): img_loss_stage0 ({6}) = {4}, id = {7}, batch cost time : {5:.4f}s'
                            .format(self.expname, epoch, data_index, self.n_batches, loss.item(), time.time()-one_batch_start_time,\
                                self.conf.get_string("render_loss.loss_type"), id[0]))
                    # print('rgb_loss = {:.8f}, kl_loss = {:.8f}'.format(rgb_loss.item(),kl_loss.item()))
                    self.writer.add_scalar('img_loss_{}_stage0'.format( self.conf.get_string("render_loss.loss_type") ), loss.item(), self.cur_iter)

                self.cur_iter += 1
            self.mat_scheduler.step()
        

        # set high spp for rendering noise-free images.
        self.model.sample_l[1] = 256
        self.plot_to_disk_cube(stage=0)
        # save material maps for each derived.
        self.plot_to_disk_material()