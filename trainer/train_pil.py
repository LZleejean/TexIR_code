'''
@File    :   train_pil.py
@Time    :   2023/02/27 12:12:20
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

import imageio
import numpy as np
import torch
from pyhocon import ConfigFactory
from tensorboardX import SummaryWriter

import utils.general as utils
import utils.plots as plt
from models.loss import IRFLoss

class PILTrainRunner():
    def __init__(self,**kwargs):
        torch.set_default_dtype(torch.float32)
        torch.set_num_threads(1)

        self.conf = ConfigFactory.parse_file(kwargs['conf'])
        self.exps_folder_name = kwargs['exps_folder_name']
        self.train_batch_size = self.conf.get_int('train.batch_size')
        self.val_batch_size = self.conf.get_int('val.batch_size')
        self.nepochs = self.conf.get_int('train.irf_epoch')
        self.max_niters = kwargs['max_niters']
        self.GPU_INDEX = kwargs['gpu_index']
        self.is_hdr_texture = self.conf.get_bool('train.is_hdr_texture') 

        self.expname = 'PIL-' + kwargs['expname']
        
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
        self.irf_optimizer_params_subdir = "IRFOptimizerParameters"
        self.irf_scheduler_params_subdir = "IRFSchedulerParameters"
        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.model_params_subdir))
        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.irf_optimizer_params_subdir))
        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.irf_scheduler_params_subdir))


        print('Write tensorboard to: ', os.path.join(self.expdir, self.timestamp))
        self.writer = SummaryWriter(os.path.join(self.expdir, self.timestamp))

        os.system("""cp -r {0} "{1}" """.format(kwargs['conf'], os.path.join(self.expdir, self.timestamp, 'runconf.conf')))

        if (not self.GPU_INDEX == 'ignore'):
            os.environ["CUDA_VISIBLE_DEVICES"] = '{0}'.format(self.GPU_INDEX)

        print('shell command : {0}'.format(' '.join(sys.argv)))
        
        print('Loading training data ...')
        self.train_dataset = utils.get_class(self.conf.get_string('train.dataset_class'))(
                                self.conf.get_string('train.path_mesh_open3d'), self.conf.get_int('train.samples_point_mesh'))
        print("train data len: {}".format(self.train_dataset.__len__()))
        self.AABB = self.train_dataset.get_AABB()
        print('Finish loading training data ...')
        print('Loading val data ...')
        self.val_dataset = utils.get_class(self.conf.get_string('val.dataset_class'))(
                                self.conf.get_string('train.path_mesh_open3d'), self.conf.get_list('val.env_res'))
        print("val data len: {}".format(self.val_dataset.__len__()))
        print('Finish loading val data ...')
        

        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset,
                                                            batch_size=self.train_batch_size,
                                                            shuffle=True
                                                            )
        self.plot_dataloader = torch.utils.data.DataLoader(self.val_dataset,
                                                           batch_size=self.val_batch_size,
                                                           shuffle=False
                                                           )
        
        self.model = utils.get_class(self.conf.get_string('train.model_class'))(conf=self.conf, AABB=self.AABB, is_hdr_texture=self.is_hdr_texture)

        if torch.cuda.is_available():
            self.model.cuda()

        self.irf_loss = utils.get_class(self.conf.get_string('train.irf_loss_class'))(**self.conf.get_config('irf_loss'))
        self.irf_optimizer = torch.optim.Adam(self.model.ir_radiance_network.parameters(),
                                                lr=self.conf.get_float('train.irf_learning_rate'))
        # self.irf_optimizer = torch.optim.Adam([
        #     {'params': self.model.incident_radiance_network.parameters(), 'lr': self.conf.get_float('train.irf_learning_rate'), 'eps': 1e-15},
        #     {'params': self.model.incident_radiance_network.embeder_param, 'lr': self.conf.get_float('train.irf_learning_rate'), 'eps': 1e-15, 'weight_decay': 1e-6}
        # ])
        # self.irf_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.irf_optimizer,
        #                                         self.conf.get_list('train.irf_sched_milestones', default=[]),
        #                                         gamma=self.conf.get_float('train.irf_sched_factor', default=0.0))
        self.irf_scheduler = torch.optim.lr_scheduler.StepLR(self.irf_optimizer,
                                                self.conf.get_int('train.irf_sched_step', default=1000),
                                                gamma=self.conf.get_float('train.irf_sched_factor', default=0.0))


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
                os.path.join(old_checkpnts_dir, self.irf_optimizer_params_subdir, str(kwargs['checkpoint']) + ".pth"))
            self.irf_optimizer.load_state_dict(data["optimizer_state_dict"])

            data = torch.load(
                os.path.join(old_checkpnts_dir, self.irf_scheduler_params_subdir, str(kwargs['checkpoint']) + ".pth"))
            self.irf_scheduler.load_state_dict(data["scheduler_state_dict"])


        self.n_batches = len(self.train_dataloader)
        self.plot_freq = self.conf.get_int('train.plot_freq')
        self.ckpt_freq = self.conf.get_int('train.ckpt_freq')
        self.val_gt = None
        self.first_val = True

        self.train_resolution = self.conf.get_list('train.env_res', default=[8,16])  # shape : (height, width)
        self.val_resolution = self.conf.get_list('train.val_sample_res', default=[8,16])  # shape : (height, width)


    def save_checkpoints(self, epoch):
        # torch.save(
        #     {"epoch": epoch, "model_state_dict": self.model.state_dict()},
        #     os.path.join(self.checkpoints_path, self.model_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "model_state_dict": self.model.state_dict()},
            os.path.join(self.checkpoints_path, self.model_params_subdir, "latest.pth"))

        # torch.save(
        #     {"epoch": epoch, "optimizer_state_dict": self.irf_optimizer.state_dict()},
        #     os.path.join(self.checkpoints_path, self.irf_optimizer_params_subdir, str(epoch) + ".pth"))
        # torch.save(
            # {"epoch": epoch, "optimizer_state_dict": self.irf_optimizer.state_dict()},
            # os.path.join(self.checkpoints_path, self.irf_optimizer_params_subdir, "latest.pth"))

        # torch.save(
        #     {"epoch": epoch, "scheduler_state_dict": self.irf_scheduler.state_dict()},
        #     os.path.join(self.checkpoints_path, self.irf_scheduler_params_subdir, str(epoch) + ".pth"))
        # torch.save(
        #     {"epoch": epoch, "scheduler_state_dict": self.irf_scheduler.state_dict()},
        #     os.path.join(self.checkpoints_path, self.irf_scheduler_params_subdir, "latest.pth"))
        


    def plot_to_disk(self):
        self.model.eval()
        
        env_res = self.conf.get_list('val.env_res')
        self.val_dataset.arrange_buffers()
        
        len_val = len(self.plot_dataloader)
        print("val data num_batch: {}".format(len_val))

        if self.first_val:
            self.val_gt = torch.zeros(env_res[0]*env_res[1], 3)
        pred_ir = torch.zeros(env_res[0]*env_res[1], 3)
        for data_index, (one_sample) in enumerate(self.plot_dataloader):

            if(data_index % len_val == int(len_val/2)):
                print("val : {}/{} finished!".format(data_index, len_val))
            points = one_sample['point'].float().cuda()
            normals = one_sample['normal'].float().cuda()

            if self.first_val:
                batch_gt_ir, batch_predicted_ir = self.model(points, normals, self.val_resolution)
                b, c = batch_gt_ir.shape

                self.val_gt[data_index*b:(data_index+1)*b] = batch_gt_ir.cpu().detach()
                pred_ir[data_index*b:(data_index+1)*b] = batch_predicted_ir.cpu().detach()
            else:
                batch_predicted_ir = self.model(points, normals, self.val_resolution, True)
                b, c = batch_predicted_ir.shape

                pred_ir[data_index*b:(data_index+1)*b] = batch_predicted_ir.cpu().detach()

        gt_ir = self.val_gt.reshape(env_res[0], env_res[1], 3)
        pred_ir = utils.hdr_recover(pred_ir.reshape(env_res[0], env_res[1], 3))

        plt.plot_irf(
            self.plots_dir,
            self.cur_iter,
            gt_ir,
            pred_ir
        )

        self.model.train()
        self.first_val = False

    def run(self):
        print("training...")
        self.cur_iter = self.start_epoch * len(self.train_dataloader)

        for epoch in range(self.start_epoch, self.nepochs + 1):
            self.train_dataset.change_points()

            # if self.cur_iter > self.max_niters:
            #     self.save_checkpoints(epoch)
            #     self.plot_to_disk()
            #     print('Training has reached max number of iterations: {}; exiting...'.format(self.cur_iter))
            #     exit(0)

            for data_index, (one_sample) in enumerate(self.train_dataloader):
                one_batch_start_time = time.time()
                self.model.train()

                if self.cur_iter % self.ckpt_freq == 0 and not self.cur_iter == 0:
                    self.save_checkpoints(epoch)

                # if self.cur_iter % self.plot_freq == 0 and not self.cur_iter == 0:
                if self.cur_iter % self.plot_freq == 0:
                    self.plot_to_disk()

                points = one_sample['point'].float().cuda()
                normals = one_sample['normal'].float().cuda()
                gt_ir, predicted_ir = self.model(points, normals, self.train_resolution)
                radiance_loss = self.irf_loss(gt_ir, predicted_ir)

                # update irf
                self.irf_optimizer.zero_grad() 
                radiance_loss.backward()
                self.irf_optimizer.step()


                if self.cur_iter % 50 == 0:
                    print('{0} [{1}] ({2}/{3}): radiance_loss = {4}, batch cost time : {5:.4f}s'
                            .format(self.expname, epoch, data_index, self.n_batches, 
                                    radiance_loss.item(), time.time()-one_batch_start_time ))
                    self.writer.add_scalar('radiance_loss', radiance_loss.item(), self.cur_iter)

                self.cur_iter += 1
            self.irf_scheduler.step()

