'''
@File    :   dataset.py
@Time    :   2023/02/27 12:08:25
@Author  :   Zhen Li 
@Contact :   yodlee@mail.nwpu.edu.cn
@Institution: Realsee
@License :   GNU General Public License v2.0
@Desc    :   None
'''


import os
import cv2
import numpy as np
import json
import time
import random

import torch
from torch import nn
from torch._C import dtype
import torch.nn.functional as F
from torch.utils import data
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.utils as vutils
from torch.autograd import Variable

import open3d as o3d

import pyredner
pyredner.set_print_timing(False)
# Use GPU if available
pyredner.set_use_gpu(torch.cuda.is_available())
# pyredner.set_use_gpu(False)

from utils.Pano2Cube import Pano2Cube

class MeshPoint(Dataset):
    def __init__(self, path_mesh, num_sample, delta=1e-2) -> None:
        super().__init__()
        self.path_mesh = path_mesh
        self.num_sample = num_sample
        self.delta = delta
        
        self.points, self.normals = None, None
        self.AABB = None
        self.mesh = self.read_mesh()


    def __len__(self):
        return self.num_sample
    
    def read_mesh(self):
        trianglemesh = o3d.io.read_triangle_mesh(self.path_mesh)
        # update correct vn
        trianglemesh.compute_vertex_normals()

        vertices = np.asarray(trianglemesh.vertices)
        # vertices = vertices * np.expand_dims(np.array([-1., -1., 1.]), axis=0)

        self.AABB = np.stack([np.min(vertices, axis=0), np.max(vertices, axis=0)], axis=0)  # shape: (2, 3)

        # trianglemesh.vertices = o3d.utility.Vector3dVector(vertices)
        # normals = np.asarray(trianglemesh.vertex_normals)
        # normals = normals * np.expand_dims(np.array([-1., -1., 1.]), axis=0)
        # trianglemesh.vertex_normals = o3d.utility.Vector3dVector(normals)
        
        return trianglemesh

    def sample_mesh(self, mesh):
        pcd = mesh.sample_points_uniformly(number_of_points=self.num_sample)
        p = np.asarray(pcd.points)
        n = np.asarray(pcd.normals)

        p = p + n * self.delta
        # pcd.points = o3d.utility.Vector3dVector(p)  # shape: (num_sample, 3)
        return p, n
    
    def change_points(self):
        self.points, self.normals = self.sample_mesh(self.mesh)
    
    def get_AABB(self):
        return self.AABB
    
    def __getitem__(self, index):

        one_sample = {
            'point': torch.from_numpy(self.points[index]),
            'normal': torch.from_numpy(self.normals[index])
        }
        return one_sample



class ImageMeshPoint(Dataset):
    def __init__(self, path_mesh, resolution=[128, 256], delta=1e-2) -> None:
        super().__init__()
        self.path_mesh = path_mesh
        self.path_root = os.path.join(os.path.dirname(os.path.dirname(path_mesh)))
        self.resolution = resolution
        self.delta = delta

        # fixed val image : 1646101011, 1646101217
        self.derived_id = 1657181612

        self.extrinsics_list, cam_position_list = self.read_extrinsic()
        self.buffers = self.render_mesh()
        self.points, self.normals = None, None
        

        # color = self.buffers[:,:,6:9].cpu().numpy()
        # cv2.imwrite("../results/test_color.jpg", color[:,:,::-1]*255)
        # postions = self.buffers[:,:,0:3].cpu().numpy()+100
        # cv2.imwrite("../results/test_position.jpg", postions[:,:,::-1])
        # normals = self.buffers[:,:,3:6].cpu().numpy()
        # cv2.imwrite("../results/test_normal.exr", normals[:,:,::-1])

        # exit(-1)

    def __len__(self):
        return self.resolution[0] * self.resolution[1]

    def render_mesh(self):
        """ fixed val image : 1646101011, 1646101217
        """
        #### redner
        material_map, mesh_list, light_map = pyredner.load_obj(self.path_mesh, obj_group=True)
        # for _, mesh in mesh_list:
        #     mesh.vertices = mesh.vertices * torch.tensor([-1.0,-1.0,1.0]).unsqueeze(0).cuda()
        #     mesh.normals = mesh.normals * torch.tensor([-1.0,-1.0,1.0]).unsqueeze(0).cuda()

        # Setup materials
        material_id_map = {}
        materials = []
        count = 0
        for key, value in material_map.items():
            material_id_map[key] = count
            count += 1
            materials.append(value)

        # Setup geometries
        shapes = []
        for mtl_name, mesh in mesh_list:
            shapes.append(pyredner.Shape(\
                vertices = mesh.vertices,
                indices = mesh.indices,
                uvs = mesh.uvs,
                uv_indices = mesh.uv_indices,
                normals = mesh.normals,
                material_id = material_id_map[mtl_name]))

        # if self.derived_id == 1646101011:
        #     test_extrinsic = torch.tensor([
        #         [0.0624092, 0.0284072, 0.997646, -0.375868],
        #         [-0.00555302, 0.999589, -0.0281151, -0.00280015],
        #         [-0.998035, -0.00378528, 0.0625423, 0.0361362],
        #         [0, 0, 0, 1]
        #     ])
        # elif self.derived_id == 1646101217:
        #     test_extrinsic = torch.tensor([
        #         [-0.138782, 0.0233641, 0.990047, 2.47513],
        #         [-0.00960273, 0.999643, -0.0249369, -0.00600408],
        #         [-0.990276, -0.0129679, -0.138508, -0.438975],
        #         [0, 0, 0, 1]
        #     ])
        # elif self.derived_id == 1657181612:
        #     test_extrinsic = torch.tensor([
        #         [0.999796, -0.00263614, -0.0200223, -8.27731],
        #         [0.0027613, 0.999977, 0.00622319, -0.0240314],
        #         [0.0200058, -0.00627716, 0.99978, 1.49406],
        #         [0, 0, 0, 1]
        #     ])
        test_extrinsic = self.extrinsics_list[0]

        # for translate axis
        # test_extrinsic[:,-1] = test_extrinsic[:, -1] * torch.tensor([-1.0, -1.0, 1.0, 1.0]).unsqueeze(0)
        # do not have to translate
        U = test_extrinsic[0:3,0].clone() * -1
        W = test_extrinsic[0:3,2].clone() 
        V = torch.cross(W, U)
        test_extrinsic[0:3,0] = -W
        test_extrinsic[0:3,1] = V
        test_extrinsic[0:3,2] = U


        camera = pyredner.Camera(cam_to_world=test_extrinsic, 
            camera_type=pyredner.camera_type.panorama, 
            resolution=(self.resolution[0], self.resolution[1]),
            clip_near = 1e-2, # needs to > 0
            fisheye = False
            )
        scene = pyredner.Scene(camera, shapes, materials)
        buffers = pyredner.render_g_buffer(scene=scene, channels=[pyredner.channels.position, pyredner.channels.geometry_normal], num_samples=[8,8])
        
        return buffers

    def read_extrinsic(self):

        with open(os.path.join(self.path_root, 'info', 'final_extrinsics.txt'),'r') as f:
            lines = f.readlines()

        # fit cyclops's final extrinsics.
        lines = [i.replace(' \n', '\n') for i in lines]
        extrinsics = np.loadtxt(lines[1:], delimiter=' ')   # shape: (N*4, 4)


        extrinsics_list = []
        cam_position_list = []
        for i in range(int(lines[0].strip())):
            # transform into redner's setting of camera.
            tmp_matrix = torch.from_numpy( extrinsics[i*4:(i+1)*4,:] )
            U = tmp_matrix[0:3,0].clone() * -1
            W = tmp_matrix[0:3,2].clone() 
            V = torch.cross(W, U)
            tmp_matrix[0:3,0] = -W
            tmp_matrix[0:3,1] = V
            tmp_matrix[0:3,2] = U

            extrinsics_list.append(tmp_matrix.float())
            cam_position_list.append(tmp_matrix[0:3, -1].float())
        return extrinsics_list, cam_position_list
    
    def arrange_buffers(self):
        buffers = self.buffers.reshape(self.resolution[0] * self.resolution[1], -1)
        points = buffers[:,0:3]
        normals = buffers[:, 3:6]
        p = points + normals * self.delta
        self.points = p
        self.normals = normals
    
    def __getitem__(self, index):

        one_sample = {
            'point': self.points[index],
            'normal': self.normals[index]
        }
        return one_sample



class ImageDerived(Dataset):
    def __init__(self, path_mesh, resolution=[1000, 2000], hdr_exposure=5.0) -> None:
        super().__init__()
        self.path_mesh = path_mesh
        self.path_root = os.path.join(os.path.dirname(os.path.dirname(path_mesh))) # "../reproject/derived"
        self.resolution = resolution
        self.hdr_exposure = hdr_exposure

        self.ids = self.read_id()
        self.extrinsics_list, self.cam_position_list= self.read_extrinsic()
        self.images_items = self.read_images(self.ids)

        self.sample_index = None


    def __len__(self):
        assert len(self.ids)==len(self.extrinsics_list) and len(self.ids)==len(self.images_items)
        return len(self.ids)

    def __getitem__(self, index):
        
        item = {
            'color': self.images_items[index]['color'],
            'mask': self.images_items[index]['mask'],
            'cam_to_world': self.extrinsics_list[index],
            'id': self.ids[index],
            'cam_position': self.cam_position_list[index]
        }
        return item
    
    def read_id(self):
        """return all derived ids.

        Returns:
            [list(str)]: a list with all derived ids.
        """

        with open(os.path.join(self.path_root, 'info', 'aligned.txt'),'r') as f:
            lines = f.readlines()
        
        for i in range(len(lines)):
            lines[i] = lines[i].strip()
        return lines
    
    def read_extrinsic(self):

        with open(os.path.join(self.path_root, 'info', 'final_extrinsics.txt'),'r') as f:
            lines = f.readlines()

        # fit cyclops's final extrinsics.
        lines = [i.replace(' \n', '\n') for i in lines]
        extrinsics = np.loadtxt(lines[1:], delimiter=' ')   # shape: (N*4, 4)


        extrinsics_list = []
        cam_position_list = []
        for i in range(int(lines[0].strip())):
            # transform into redner's setting of camera.
            tmp_matrix = torch.from_numpy( extrinsics[i*4:(i+1)*4,:] )
            U = tmp_matrix[0:3,0].clone() * -1
            W = tmp_matrix[0:3,2].clone() 
            V = torch.cross(W, U)
            tmp_matrix[0:3,0] = -W
            tmp_matrix[0:3,1] = V
            tmp_matrix[0:3,2] = U

            extrinsics_list.append(tmp_matrix.float())
            cam_position_list.append(tmp_matrix[0:3, -1].float())
        return extrinsics_list, cam_position_list
    
    def read_images(self, ids):
        derived_path = os.path.join(self.path_root, 'derived')
        imgs_list = []
        for id in ids:
            target_path = os.path.join(derived_path, id, 'panoImage_orig.jpg')

            img = cv2.imread(target_path,-1)
            mask = img[:,:,3:4]

            target_path = target_path.replace('derived','hdr').replace('panoImage_orig.jpg','ccm.hdr')
            img = cv2.imread(target_path,-1)
            color = img[:,:,0:3]
            intensity = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
            mask_light_source = np.asarray(intensity > 0.1, np.uint8)*255
            # mask = mask - mask_light_source[:,:,np.newaxis]

            # color = cv2.GaussianBlur(color, (7,7), 2)
            
            color = cv2.resize(color, (self.resolution[1], self.resolution[0]), interpolation=cv2.INTER_CUBIC)
            # color = np.asarray(color, np.float32)/255.0
            color = np.clip(color, a_min=0., a_max=np.finfo(np.float32).max)
            color = color * (2**self.hdr_exposure)
            
            # exclude outliers
            mask = cv2.erode(mask, np.ones((7,7), np.uint8))
            mask = cv2.resize(mask,(self.resolution[1], self.resolution[0]), interpolation=cv2.INTER_NEAREST )
            mask = np.asarray(mask, np.float32)/255.0   # attention: actual shape: (h, w)
            

            color = color[:,:,::-1]
            # color = color **(2.2)   # srgb -> linear
            # shape: (h, w, c)
            item = {
                'color': torch.from_numpy(color.copy()),
                'mask': torch.from_numpy(mask).unsqueeze(-1)
            }
            imgs_list.append(item)
        return imgs_list


class ImageCubeDerived(Dataset):
    def __init__(self, path_mesh, resolution=[1000, 2000], hdr_exposure=5.0) -> None:
        super().__init__()
        self.path_mesh = path_mesh
        self.path_root = os.path.join(os.path.dirname(os.path.dirname(path_mesh))) # "../reproject/derived"
        self.resolution = resolution
        self.cube_res = int(resolution[1]/4)
        self.hdr_exposure = hdr_exposure
        
        self.pano2cube = Pano2Cube(1, 4000, 8000, self.cube_res, 6)

        self.ids = self.read_id()
        self.extrinsics_list, self.cam_position_list = self.read_extrinsic()
        self.images_items = self.read_images(self.ids)
        

        self.sample_index = None


    def __len__(self):
        assert len(self.ids)==len(self.extrinsics_list) and len(self.ids)==len(self.images_items)
        return len(self.ids)

    def __getitem__(self, index):
        
        item = {
            'color': self.images_items[index]['color'],
            'mask': self.images_items[index]['mask'],
            'segs': self.images_items[index]['segs'],
            'cam_to_world': self.extrinsics_list[index],    # Attention! this key actually is mvp, for ensuring consistent with other model.
            'id': self.ids[index],
            'cam_position': self.cam_position_list[index],
            'rgb_grad': self.images_items[index]['rgb_grad']
        }
        return item
    
    def read_id(self):
        """return all derived ids.

        Returns:
            [list(str)]: a list with all derived ids.
        """

        with open(os.path.join(self.path_root, 'info', 'aligned.txt'),'r') as f:
            lines = f.readlines()
        # del lines[-2]
        for i in range(len(lines)):
            lines[i] = lines[i].strip()
        return lines
    
    def read_extrinsic(self):

        with open(os.path.join(self.path_root, 'info', 'final_extrinsics.txt'),'r') as f:
            lines = f.readlines()
        # del lines[-8:-4]
        # fit cyclops's final extrinsics.
        lines = [i.replace(' \n', '\n') for i in lines]
        extrinsics = np.loadtxt(lines[1:], delimiter=' ')   # shape: (N*4, 4)


        extrinsics_list = []
        cam_position_list = []
        for i in range(int(extrinsics.shape[0]/4)):
            # transform into redner's setting of camera.
            test_extrinsic = torch.from_numpy( extrinsics[i*4:(i+1)*4,:] ).float()
            cam_position_list.append(test_extrinsic[0:3, -1])
            
            # face 0
            test_face0 = test_extrinsic.clone()
            Right = test_face0[0:3, 0].clone()
            Front = test_face0[0:3, 2].clone()
            test_face0[0:3, 2] = -Right
            test_face0[0:3, 0] = Front
            test_face0 = torch.inverse(test_face0)
            # default is face 1
            test_face1 = torch.inverse(test_extrinsic)
            # face 2
            test_face2 = test_extrinsic.clone()
            Right = test_face2[0:3, 0].clone()
            Front = test_face2[0:3, 2].clone()
            test_face2[0:3, 2] = Right
            test_face2[0:3, 0] = -Front
            test_face2 = torch.inverse(test_face2)
            # face 3
            test_face3 = test_extrinsic.clone()
            Right = test_face3[0:3, 0].clone()
            Front = test_face3[0:3, 2].clone()
            test_face3[0:3, 2] = -Front
            test_face3[0:3, 0] = -Right
            test_face3 = torch.inverse(test_face3)
            # face 4
            test_face4 = test_extrinsic.clone()
            Front = test_face4[0:3, 2].clone()
            Right = test_face4[0:3, 0].clone()
            Up = torch.cross(Right, Front)
            test_face4[0:3, 2] = Up
            test_face4[0:3, 1] = Front  # don't know hot to explain this direction
            test_face4 = torch.inverse(test_face4)
            # face 5
            test_face5 = test_extrinsic.clone()
            Front = test_face5[0:3, 2].clone()
            Right = test_face5[0:3, 0].clone()
            Up = torch.cross(Right, Front)
            test_face5[0:3, 2] = -Up
            test_face5[0:3, 1] = -Front  # don't know hot to explain this direction
            test_face5 = torch.inverse(test_face5)

            test_w2c = torch.stack([test_face0, test_face1, test_face2, test_face3, test_face4, test_face5], dim=0)

            proj = torch.from_numpy(self.projection())
            proj = proj.expand(6, 4, 4)
            # mvp = torch.matmul(proj, test_w2c)
            mvp = torch.einsum('ijk,ikl->ijl', proj, test_w2c)
            mvp = torch.transpose(mvp, 1, 2)


            extrinsics_list.append(mvp.float())
        return extrinsics_list, cam_position_list
    
    def read_images(self, ids):
        derived_path = os.path.join(self.path_root, 'derived')
        imgs_list = []
        for id in ids:

            target_path = os.path.join(derived_path, id, 'panoImage_orig.jpg')

            img = cv2.imread(target_path,-1)
            h,w,c = img.shape
            mask = img[:,:,3:4]

            target_path = target_path.replace('derived','hdr').replace('panoImage_orig.jpg','ccm.hdr')
            img = cv2.imread(target_path,-1)
            color = img[:,:,0:3]
            intensity = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
            mask_light_source = np.asarray(intensity > 0.5, np.uint8)*255
            # mask = mask - mask_light_source[:,:,np.newaxis]

            color = np.clip(color, a_min=0., a_max=np.finfo(np.float32).max)
            color = color[:,:,::-1]
            color = color * (2**self.hdr_exposure)
            
            rgb_gray = cv2.cvtColor(color, cv2.COLOR_RGB2GRAY)                              # [H, W]
            rgb_grad_x = cv2.Sobel(rgb_gray, cv2.CV_32F, 1, 0, ksize=3)         # [H, W]
            rgb_grad_y = cv2.Sobel(rgb_gray, cv2.CV_32F, 0, 1, ksize=3)         # [H, W]
            rgb_grad = cv2.magnitude(rgb_grad_x, rgb_grad_y)                                    # [H, W]

            # exclude outliers
            mask = cv2.erode(mask, np.ones((5,5), np.uint8))
            mask = np.asarray(mask, np.float32)/255.0   # attention: actual shape: (h, w)
            
            target_path = target_path.replace('ccm.hdr', 'panoImage_gray.png').replace('hdr', 'derived')
            segs = cv2.imread(target_path, -1)
            segs = cv2.resize(segs, (w,h), interpolation=cv2.INTER_NEAREST)
            segs = np.asarray(segs, np.float32)



            
            img = np.concatenate([color, mask[...,None], segs[...,None], rgb_grad[...,None]], axis=-1)  # shape: (h, w, c)
            
            h,w,c = img.shape
            img = transforms.ToTensor()(img)    # shape: (c, h, w)
            img = torch.reshape(img,(1,c,h,w))
            img = self.pano2cube.Tocube(img, mode='nearest')    # shape: (1, 6*c, cube_res, cube_res)

            color_list = []
            mask_list = []
            segs_list = []
            rgbgrad_list = []
            for i in range(6):
                face = img[0, i*c:(i+1)*c, :, :]
                color = face[0:3, :, :].permute(1,2,0)  # shape: (cube_res, cube_res, 3)
                color_list.append(color)
                mask = face[3:4, :, :].permute(1,2,0)  # shape: (cube_res, cube_res, 1)
                mask_list.append(mask)
                segs = face[4:5, :, :].permute(1,2,0)  # shape: (cube_res, cube_res, 1)
                segs_list.append(segs)

                grad = face[5:6, :, :].permute(1,2,0)  # shape: (cube_res, cube_res, 1)
                rgbgrad_list.append(grad)

            # shape: (6, cube_res, cube_res, c)
            item = {
                'color': torch.stack(color_list, dim=0),
                'mask': torch.stack(mask_list, dim=0),
                'segs': torch.stack(segs_list, dim=0),
                'rgb_grad': torch.stack(rgbgrad_list, dim=0)
            }
            imgs_list.append(item)
        return imgs_list
    
    # attention, our front is +z, so the projection matrix is below
    def projection(self, fov=90., n=1e-4, f=100):
        fov = fov * np.pi / 180.
        return np.array([[1/np.math.tan(fov/2.),    0,            0,              0],
                        [  0,  1/np.math.tan(fov/2.),            0,              0],
                        [  0,    0, (f+n)/(f-n), -(2*f*n)/(f-n)],
                        [  0,    0,           1,              0]]).astype(np.float32)


class ImageCubeNovel(Dataset):
    def __init__(self, path_mesh, resolution=[1000, 2000], hdr_exposure=5.0) -> None:
        super().__init__()
        self.path_mesh = path_mesh
        self.path_root = os.path.join(os.path.dirname(os.path.dirname(path_mesh))) # "../reproject/derived"
        self.resolution = resolution
        self.cube_res = int(resolution[1]/4)
        self.hdr_exposure = hdr_exposure
        
        self.pano2cube = Pano2Cube(1, 4000, 8000, self.cube_res, 5)


        self.extrinsics_list, self.cam_position_list = self.read_extrinsic()

        self.sample_index = None


    def __len__(self):
        return len(self.extrinsics_list)

    def __getitem__(self, index):
        
        item = {
            'cam_to_world': self.extrinsics_list[index],    # Attention! this key actually is mvp, for ensuring consistent with other model.
            'cam_position': self.cam_position_list[index]
        }
        return item
    
    def read_extrinsic(self):

        with open(os.path.join(self.path_root, 'info', 'final_extrinsics.txt'),'r') as f:
            lines = f.readlines()
        # del lines[-8:-4]
        # fit cyclops's final extrinsics.
        lines = [i.replace(' \n', '\n') for i in lines]
        extrinsics = np.loadtxt(lines[1:], delimiter=' ')   # shape: (N*4, 4)


        extrinsics_list = []
        cam_position_list = []
        
        # change start_index and direction to generate novel view
        start_index = 2
        direction = torch.tensor([1.0, 0.0, 0.0])

        test_extrinsic_start = torch.from_numpy( extrinsics[start_index*4:(start_index+1)*4,:] ).float()
        test_extrinsic_start[0:3, -1] = test_extrinsic_start[0:3, -1] + torch.tensor([-0.2, 0.0, -0.6])

        num = 60
        # move 4 m to +x axis
        step = 6/num
        for i in range(num):
            test_extrinsic = test_extrinsic_start.clone()
            test_extrinsic[0:3, -1] = test_extrinsic[0:3, -1] + direction * step * i
            cam_position_list.append(test_extrinsic[0:3, -1])

            # face 0
            test_face0 = test_extrinsic.clone()
            Right = test_face0[0:3, 0].clone()
            Front = test_face0[0:3, 2].clone()
            test_face0[0:3, 2] = -Right
            test_face0[0:3, 0] = Front
            test_face0 = torch.inverse(test_face0)
            # default is face 1
            test_face1 = torch.inverse(test_extrinsic)
            # face 2
            test_face2 = test_extrinsic.clone()
            Right = test_face2[0:3, 0].clone()
            Front = test_face2[0:3, 2].clone()
            test_face2[0:3, 2] = Right
            test_face2[0:3, 0] = -Front
            test_face2 = torch.inverse(test_face2)
            # face 3
            test_face3 = test_extrinsic.clone()
            Right = test_face3[0:3, 0].clone()
            Front = test_face3[0:3, 2].clone()
            test_face3[0:3, 2] = -Front
            test_face3[0:3, 0] = -Right
            test_face3 = torch.inverse(test_face3)
            # face 4
            test_face4 = test_extrinsic.clone()
            Front = test_face4[0:3, 2].clone()
            Right = test_face4[0:3, 0].clone()
            Up = torch.cross(Right, Front)
            test_face4[0:3, 2] = Up
            test_face4[0:3, 1] = Front  # don't know hot to explain this direction
            test_face4 = torch.inverse(test_face4)
            # face 5
            test_face5 = test_extrinsic.clone()
            Front = test_face5[0:3, 2].clone()
            Right = test_face5[0:3, 0].clone()
            Up = torch.cross(Right, Front)
            test_face5[0:3, 2] = -Up
            test_face5[0:3, 1] = -Front  # don't know hot to explain this direction
            test_face5 = torch.inverse(test_face5)

            test_w2c = torch.stack([test_face0, test_face1, test_face2, test_face3, test_face4, test_face5], dim=0)

            proj = torch.from_numpy(self.projection())
            proj = proj.expand(6, 4, 4)
            # mvp = torch.matmul(proj, test_w2c)
            mvp = torch.einsum('ijk,ikl->ijl', proj, test_w2c)
            mvp = torch.transpose(mvp, 1, 2)


            extrinsics_list.append(mvp.float())
        return extrinsics_list, cam_position_list
    
    # attention, our front is +z, so the projection matrix is below
    def projection(self, fov=90., n=1e-4, f=100):
        fov = fov * np.pi / 180.
        return np.array([[1/np.math.tan(fov/2.),    0,            0,              0],
                        [  0,  1/np.math.tan(fov/2.),            0,              0],
                        [  0,    0, (f+n)/(f-n), -(2*f*n)/(f-n)],
                        [  0,    0,           1,              0]]).astype(np.float32)


class ImageCubeSyn(Dataset):
    def __init__(self, path_mesh, resolution=[1000, 2000], hdr_exposure=1.0) -> None:
        super().__init__()
        self.path_mesh = path_mesh
        self.path_root = os.path.join(os.path.dirname(os.path.dirname(path_mesh))) # "../reproject/derived"
        self.resolution = resolution
        self.cube_res = int(resolution[1]/4)
        self.hdr_exposure = hdr_exposure
        
        self.pano2cube = Pano2Cube(1, 512, 1024, self.cube_res, 6)

        self.ids = self.read_id()
        self.extrinsics_list, self.cam_position_list = self.read_extrinsic()
        self.images_items = self.read_images(self.ids)

        self.novel_ids = self.read_id('novel.txt')
        self.novel_extrinsics_list, self.novel_cam_position_list = self.read_extrinsic('novel_extrinsics.txt')
        self.novel_images_items = self.read_images(self.novel_ids)
        

        self.sample_index = None


    def __len__(self):
        assert len(self.ids)==len(self.extrinsics_list) and len(self.ids)==len(self.images_items)
        return len(self.ids)

    def __getitem__(self, index):
        
        item = {
            'color': self.images_items[index]['color'],
            'mask': self.images_items[index]['mask'],
            'segs': self.images_items[index]['segs'],
            'cam_to_world': self.extrinsics_list[index],    # Attention! this key actually is mvp, for ensuring consistent with other model.
            'id': self.ids[index],
            'cam_position': self.cam_position_list[index],
            'rgb_grad': self.images_items[index]['rgb_grad'],
            'gt_albedo': self.images_items[index]['gt_albedo'],
            'gt_roughness': self.images_items[index]['gt_roughness']
        }
        return item
    
    def read_id(self, txt_name='aligned.txt'):
        """return all derived ids.

        Returns:
            [list(str)]: a list with all derived ids.
        """

        with open(os.path.join(self.path_root, 'info', txt_name),'r') as f:
            lines = f.readlines()
        # del lines[-2]
        for i in range(len(lines)):
            lines[i] = lines[i].strip()
        return lines
    
    def read_extrinsic(self, txt_name='final_extrinsics.txt'):

        with open(os.path.join(self.path_root, 'info', txt_name),'r') as f:
            lines = f.readlines()
        # del lines[-8:-4]
        # fit cyclops's final extrinsics.
        lines = [i.replace(' \n', '\n') for i in lines]
        extrinsics = np.loadtxt(lines[1:], delimiter=' ')   # shape: (N*4, 4)


        extrinsics_list = []
        cam_position_list = []
        for i in range(int(extrinsics.shape[0]/4)):
            # transform into redner's setting of camera.
            test_extrinsic = torch.from_numpy( extrinsics[i*4:(i+1)*4,:] ).float()
            cam_position_list.append(test_extrinsic[0:3, -1])
            
            # face 0
            test_face0 = test_extrinsic.clone()
            Right = test_face0[0:3, 0].clone()
            Front = test_face0[0:3, 2].clone()
            test_face0[0:3, 2] = -Right
            test_face0[0:3, 0] = Front
            test_face0 = torch.inverse(test_face0)
            # default is face 1
            test_face1 = torch.inverse(test_extrinsic)
            # face 2
            test_face2 = test_extrinsic.clone()
            Right = test_face2[0:3, 0].clone()
            Front = test_face2[0:3, 2].clone()
            test_face2[0:3, 2] = Right
            test_face2[0:3, 0] = -Front
            test_face2 = torch.inverse(test_face2)
            # face 3
            test_face3 = test_extrinsic.clone()
            Right = test_face3[0:3, 0].clone()
            Front = test_face3[0:3, 2].clone()
            test_face3[0:3, 2] = -Front
            test_face3[0:3, 0] = -Right
            test_face3 = torch.inverse(test_face3)
            # face 4
            test_face4 = test_extrinsic.clone()
            Front = test_face4[0:3, 2].clone()
            Right = test_face4[0:3, 0].clone()
            Up = torch.cross(Right, Front)
            test_face4[0:3, 2] = Up
            test_face4[0:3, 1] = Front  # don't know how to explain this direction
            test_face4 = torch.inverse(test_face4)
            # face 5
            test_face5 = test_extrinsic.clone()
            Front = test_face5[0:3, 2].clone()
            Right = test_face5[0:3, 0].clone()
            Up = torch.cross(Right, Front)
            test_face5[0:3, 2] = -Up
            test_face5[0:3, 1] = -Front  # don't know how to explain this direction
            test_face5 = torch.inverse(test_face5)

            test_w2c = torch.stack([test_face0, test_face1, test_face2, test_face3, test_face4, test_face5], dim=0)

            proj = torch.from_numpy(self.projection())
            proj = proj.expand(6, 4, 4)
            # mvp = torch.matmul(proj, test_w2c)
            mvp = torch.einsum('ijk,ikl->ijl', proj, test_w2c)
            mvp = torch.transpose(mvp, 1, 2)


            extrinsics_list.append(mvp.float())
        return extrinsics_list, cam_position_list
    
    def read_images(self, ids):
        derived_path = os.path.join(self.path_root, 'derived')
        imgs_list = []
        for id in ids:

            target_path = os.path.join(derived_path, id, 'panoImage_orig.jpg')

            img = cv2.imread(target_path,-1)
            h,w,c = img.shape
            mask = img[:,:,3:4]

            target_path = target_path.replace('derived','hdr').replace('panoImage_orig.jpg','ccm.hdr')
            img = cv2.imread(target_path,-1)
            color = img[:,:,0:3]
            intensity = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
            mask_light_source = np.asarray(intensity > 0.5, np.uint8)*255
            # mask = mask - mask_light_source[:,:,np.newaxis]

            color = np.clip(color, a_min=0., a_max=np.finfo(np.float32).max)
            color = color[:,:,::-1]
            color = color * (2**self.hdr_exposure)
            
            rgb_gray = cv2.cvtColor(color, cv2.COLOR_RGB2GRAY)                              # [H, W]
            rgb_grad_x = cv2.Sobel(rgb_gray, cv2.CV_32F, 1, 0, ksize=3)         # [H, W]
            rgb_grad_y = cv2.Sobel(rgb_gray, cv2.CV_32F, 0, 1, ksize=3)         # [H, W]
            rgb_grad = cv2.magnitude(rgb_grad_x, rgb_grad_y)                                    # [H, W]

            # exclude outliers
            mask = cv2.erode(mask, np.ones((5,5), np.uint8))
            mask = np.asarray(mask, np.float32)/255.0   # attention: actual shape: (h, w)
            
            target_path = target_path.replace('ccm.hdr', 'panoImage_gray.png').replace('hdr', 'derived')
            segs = cv2.imread(target_path, -1)
            segs = cv2.resize(segs, (w,h), interpolation=cv2.INTER_NEAREST)
            segs = np.asarray(segs, np.float32)
            # add noise for seg. 16*16(512/32), 256*256(512/2)
            mask_lenth =  int(w/32)
            mask_x = random.randint(0, w - mask_lenth) # 缺失部分左下角的x坐标
            mask_y = random.randint(0, h - mask_lenth)  #缺失部分左上角的y坐标
            #segs[mask_y:mask_y+mask_lenth, mask_x:mask_x+mask_lenth] = 0.0

            segs_pano = torch.from_numpy(cv2.resize(segs, (self.cube_res*4, self.cube_res*2), interpolation=cv2.INTER_NEAREST)[...,None])
            
            img = np.concatenate([color, mask[...,None], segs[...,None], rgb_grad[...,None]], axis=-1)  # shape: (h, w, c)
            
            h,w,c = img.shape
            img = transforms.ToTensor()(img)    # shape: (c, h, w)
            img = torch.reshape(img,(1,c,h,w))
            img = self.pano2cube.Tocube(img, mode='nearest')    # shape: (1, 6*c, cube_res, cube_res)

            albedo_path = target_path.replace('panoImage_gray.png','albedo.png')
            albedo = cv2.imread(albedo_path, -1)[:,:,::-1]
            albedo = cv2.resize(albedo, (self.cube_res*4, self.cube_res*2))
            albedo = np.asarray(albedo, np.float32)/255.0
            # srgb->linear
            albedo = albedo**(2.2)
            albedo = torch.from_numpy(albedo)

            roughness_path = target_path.replace('panoImage_gray.png','roughness.png')
            roughness = cv2.imread(roughness_path, -1)
            roughness = cv2.resize(roughness, (self.cube_res*4, self.cube_res*2))
            roughness = np.asarray(roughness, np.float32)/255.0
            roughness = torch.from_numpy(roughness)

            color_list = []
            mask_list = []
            segs_list = []
            rgbgrad_list = []
            for i in range(6):
                face = img[0, i*c:(i+1)*c, :, :]
                color = face[0:3, :, :].permute(1,2,0)  # shape: (cube_res, cube_res, 3)
                color_list.append(color)
                mask = face[3:4, :, :].permute(1,2,0)  # shape: (cube_res, cube_res, 1)
                mask_list.append(mask)
                segs = face[4:5, :, :].permute(1,2,0)  # shape: (cube_res, cube_res, 1)
                segs_list.append(segs)

                grad = face[5:6, :, :].permute(1,2,0)  # shape: (cube_res, cube_res, 1)
                rgbgrad_list.append(grad)

            # shape: (6, cube_res, cube_res, c)
            item = {
                'color': torch.stack(color_list, dim=0),
                'mask': torch.stack(mask_list, dim=0),
                'segs': torch.stack(segs_list, dim=0),
                'rgb_grad': torch.stack(rgbgrad_list, dim=0),
                'gt_albedo': albedo,   # [h, w, 3]
                'gt_roughness': roughness,
                'segs_pano': segs_pano
            }
            imgs_list.append(item)
        return imgs_list
    
    # attention, our front is +z, so the projection matrix is below
    def projection(self, fov=90., n=1e-4, f=100):
        fov = fov * np.pi / 180.
        return np.array([[1/np.math.tan(fov/2.),    0,            0,              0],
                        [  0,  1/np.math.tan(fov/2.),            0,              0],
                        [  0,    0, (f+n)/(f-n), -(2*f*n)/(f-n)],
                        [  0,    0,           1,              0]]).astype(np.float32)



