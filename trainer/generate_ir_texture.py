'''
@File    :   generate_ir_texture.py
@Time    :   2023/02/27 12:11:17
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

import cv2
import numpy as np
import torch
from torch.nn import functional as F
from pyhocon import ConfigFactory
from tensorboardX import SummaryWriter

import utils.general as utils
import utils.plots as plt
from models.loss import IRFLoss
from utils.Cube2Pano import Cube2Pano

class IrrTextureRunner():
    def __init__(self,**kwargs):
        torch.set_default_dtype(torch.float32)
        torch.set_num_threads(1)

        self.conf = ConfigFactory.parse_file(kwargs['conf'])
        self.exps_folder_name = kwargs['exps_folder_name']
        self.train_batch_size = self.conf.get_int('train.batch_size')
        self.nepochs = self.conf.get_int('train.mat_epoch')
        
        self.max_niters = kwargs['max_niters']
        self.GPU_INDEX = kwargs['gpu_index']

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
        
        self.model = utils.get_class(self.conf.get_string('train.model_class'))(conf=self.conf, \
            ids=self.train_dataset.ids, extrinsics = self.train_dataset.extrinsics_list, optim_cam=self.conf.get_bool('train.optim_cam'))
        if torch.cuda.is_available():
            self.model.cuda()

        self.start_epoch = 0
        # self.room_meta_scale, self.room_meta_w, self.room_meta_h, self.room_meta_xmin, self.room_meta_zmin, self.room_img= utils.parse_roomseg(\
        #     os.path.join(os.path.dirname(os.path.dirname(self.conf.get_string('train.path_mesh_open3d'))), 'roomseg'))


        
    def run(self):
        print("generating...")
        irr_texture = self.model()
        target_texture_path = self.conf.get_string('train.path_mesh_open3d').replace('out1.obj', '0_irr_texture.hdr')

        irr_texture_numpy = irr_texture.cpu().numpy()[:,:,::-1]
        print(irr_texture_numpy.shape)
        cv2.imwrite(target_texture_path, irr_texture_numpy)
