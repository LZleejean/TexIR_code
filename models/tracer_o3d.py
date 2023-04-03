'''
@File    :   tracer_o3d.py
@Time    :   2023/02/27 12:10:36
@Author  :   Zhen Li 
@Contact :   yodlee@mail.nwpu.edu.cn
@Institution: Realsee
@License :   GNU General Public License v2.0
@Desc    :   None
'''


import cv2
import numpy as np
import open3d as o3d

import torch
import torch.nn as nn
import torch.nn.functional as F
from pyhocon import ConfigFactory

import pyredner
pyredner.set_print_timing(False)
# Use GPU if available
pyredner.set_use_gpu(torch.cuda.is_available())

from models.incidentNet import IRNetwork, NeILFMLP
from utils.sample_util import *

class TracerO3d(nn.Module):
    def __init__(self, conf, AABB=None, is_hdr_texture=False):
        super().__init__()
        # self.incident_radiance_network = IRNetwork(**conf.get_config('models.incident_radiance_network'), AABB=AABB)
        self.incident_radiance_network = NeILFMLP()

        # self.resolution = conf.get_list('train.env_res', default=[8,16])  # shape : (height, width)
        # self.resolution[0] = int(self.resolution[0]*8)  # like 8spp

        # self.num_sample_dir = int(self.resolution[0]*self.resolution[1]) #conf.get_int('train.num_sample_dir', default=128)

        self.path_traced_mesh = conf.get_string('train.path_mesh_open3d')


        # init ray casting scene
        trianglemesh = o3d.io.read_triangle_mesh(self.path_traced_mesh)     # o3d tracer must use the mesh with one texture map.
        trianglemesh.compute_vertex_normals()
        # vertices = np.asarray(trianglemesh.vertices)
        # vertices = vertices * np.expand_dims(np.array([-1., -1., 1.]), axis=0)
        # trianglemesh.vertices = o3d.utility.Vector3dVector(vertices)
        # normals = np.asarray(trianglemesh.vertex_normals)
        # normals = normals * np.expand_dims(np.array([-1., -1., 1.]), axis=0)
        # trianglemesh.vertex_normals = o3d.utility.Vector3dVector(normals)
        # read extra hdr texture because open3d cannot read .hdr/.exr
        if is_hdr_texture:
            texture = cv2.imread(self.path_traced_mesh.replace("out1.obj","hdr_texture.hdr"), -1)[:,:,::-1]
            texture = cv2.flip(texture, 0)
            texture = np.asarray(texture, np.float32)
            self.texture = (torch.from_numpy(texture).permute(2, 0, 1).unsqueeze(0).float()) # shape: (1, H, W, 3)
            self.texture = self.texture * (2**conf.get_float('train.hdr_exposure'))
        else:
            texture = np.asarray(trianglemesh.textures[1])
            self.texture = (torch.from_numpy(texture).permute(2, 0, 1).unsqueeze(0).float()/255.)**(2.2) # shape: (1, H, W, 3)
        
        
        self.triangle_uvs = np.asarray(trianglemesh.triangle_uvs)

        mesh = o3d.t.geometry.TriangleMesh.from_legacy(trianglemesh)
        # Create a scene and add the triangle mesh.
        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(mesh)


        
    def forward(self, points, normals, resolution, isnot_first_val=False):
        """

        Args:
            points (torch.float32): shape: [b, 3] 
            normals (torch.float32, optional): shape: [b, 3] . Defaults to None.
            isnot_first_val (bool): if set true, do not trace gt ir and only predict ir.

        Returns:
            gt_ir (torch.float32): shape: [b, h*w, 3]
            predicted_ir (torch.float32): shape: [b, h*w, 3]
        """
        b, c = points.shape

        directions = self.generate_dir(normals, resolution) # shape: (b, num_sample, 1, 3)
        rays = torch.cat([points.unsqueeze(1).unsqueeze(1).expand_as(directions), directions], dim=-1)
        if not isnot_first_val:

            ray = o3d.core.Tensor(rays.cpu().numpy(), dtype=o3d.core.Dtype.Float32) # attention: RaycastingScene only support cpu
            
            intersections = self.scene.cast_rays(ray)
            hit = intersections['t_hit'].numpy()    # shape: (b, num_sample, 1)

            # mask = np.isfinite(hit)     # shape: (b, num_sample, 1)
            mask = np.logical_and(hit > 1e-4, np.isfinite(hit))

            prim_id = intersections['primitive_ids'].numpy()    # shape: (b, num_sample, 1)
            prim_uvs = intersections['primitive_uvs'].numpy()    # shape: (b, num_sample, 1, 2)
            prim_uvs = np.clip(prim_uvs, 0., 1.)

            prim_id[~mask] = 0

            tmp = np.stack([prim_id*3+0, prim_id*3+1, prim_id*3+2], axis=0)    # shape: (3, b, num_sample, 1)
            tmp = tmp.reshape(-1)
            index = self.triangle_uvs[tmp]
            index = index.reshape(3, b, resolution[0],resolution[1], 2) # shape: (3, b, num_sample, 1, 2)
            grid = index[0,:,:,:,:] * (1-prim_uvs[:,:,:,0:1]-prim_uvs[:,:,:,1:2]) + index[1,:,:,:,:] * prim_uvs[:,:,:,0:1] + index[2,:,:,:,:] * prim_uvs[:,:,:,1:2] # shape: (b, num_sample, 1, 2)
            grid = torch.from_numpy(grid).float() # shape: (b, num_sample, 1, 2)
            grid[:,:,:,0:1] = grid[:,:,:,0:1]*2-1
            grid[:,:,:,1:2] = -(1-grid[:,:,:,1:2]*2)
            
            gt_ir = F.grid_sample(self.texture.expand([b]+list(self.texture.shape[1:])), grid, mode='bilinear', padding_mode="border",align_corners=False).permute(0,2,3,1) # shape: (b, num_sample, 1, 3)
            ner_mask = ~mask
            gt_ir[ner_mask,:] = 0


            predicted_mask_ir = self.incident_radiance_network(torch.cat([rays[:,:,:,0:3].reshape(-1, 3), rays[:,:,:,3:6].reshape(-1, 3)], dim=-1)) # shape: (b*num_sample*1, 3)
            predicted_mask_ir = predicted_mask_ir.reshape(b, resolution[0],resolution[1], 3)
            predicted_mask_ir[~mask,:] = 0

            res = {
                'gt':  gt_ir.reshape(b, -1, 3).cuda() ,
                'pred': predicted_mask_ir.reshape(b, -1, 3)
            }

            return res
        else:
            predicted_mask_ir = self.incident_radiance_network(torch.cat([rays[:,:,:,0:3].reshape(-1, 3), rays[:,:,:,3:6].reshape(-1, 3)], dim=-1)) # shape: (b*num_sample*1, 3)
            predicted_mask_ir = predicted_mask_ir.reshape(b, resolution[0],resolution[1], 3)
            res = {
                'pred': predicted_mask_ir.reshape(b, -1, 3)
            }
            return res
        
    
    def RadicalInverse(self,bits):
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
    
    def Hammersley(self,i,N):
        return [float(i)/float(N),self.RadicalInverse(i)]

    def generate_dir(self, normals, resolution):
        b, c = normals.shape
        normals = normals.unsqueeze(1).unsqueeze(1).expand(b, resolution[0],resolution[1], 3)
        # compute projection axis
        x_axis = torch.zeros_like(normals).cuda()  #size:(batch_size, samples, 1, 3)
        mask = torch.abs(normals[:,:,:,0]) > 0.99
        x_axis[mask, :] = torch.tensor([0., 1., 0.],dtype=torch.float32, device=normals.get_device())
        x_axis[~mask, :] = torch.tensor([1., 0., 0.],dtype=torch.float32, device=normals.get_device())

        def norm_axis(x):
            return x / (torch.norm(x, dim=-1, keepdim=True) + TINY_NUMBER)

        normals = norm_axis(normals)
        U = norm_axis(torch.cross(x_axis, normals))
        V = norm_axis(torch.cross( normals, U))

        num_sample_dir = resolution[0]*resolution[1]
        samples=np.zeros((num_sample_dir,2),dtype=np.float32)
        for i in range(0,num_sample_dir):
            s = Hammersley(i,num_sample_dir)
            samples[i][0] = s[0]
            samples[i][1] = s[1]
        samples = torch.from_numpy(samples).unsqueeze(0).unsqueeze(-2).cuda() #size:(batch_size, samples, 1, 2)
        samples = samples.repeat(b, 1, 1, 1).detach()
        # samples[:,:, 0:1] = torch.clamp(samples[:,:,0:1] + torch.rand_like(samples[:,:,0:1])*0.09, 0., 1.)
        shift = torch.rand(b, 1, 1, 2).cuda()
        samples = samples + shift
        index1 = samples > 1.
        samples[index1] = samples[index1]-1.
        index2 = samples < 0.
        samples[index2] = samples[index2] + 1
        samples = torch.clamp(samples, 0+TINY_NUMBER, 1-TINY_NUMBER)    # avoid NAN in roughness backward.
        samples = samples.expand(b, num_sample_dir, 1, 2).reshape(b, resolution[0], resolution[1], 2)
        
        # ############ test sample for ordered variable, attention: the cosTheta is uniformly generated instead elevation. so the res pano will have distortion in elevation.
        # azimuth = torch.linspace(0.,1.,256).cuda()
        # elevation = torch.linspace(0.,1.,128).cuda()
        # elevation,azimuth = torch.meshgrid([elevation,azimuth])
        # samples = torch.stack([elevation,azimuth], dim=-1).unsqueeze(0) #size:(batch_size, h, w, 2)

        # uniform sample, attention: we generate sampled dir via y as up axis. translate to our coor:
        # phi - np.pi; y = sin((np.pi/2-theta)) =  costheta; y_projected = cos((np.pi/2-theta)) = sintheta
        phi = 2 * np.pi * samples[:,:,:,1:2] - np.pi
        cosTheta = (1.0 - samples[:,:,:,0:1])
        sinTheta = torch.sqrt(1.0 - cosTheta * cosTheta)
        L = V * (torch.sin(phi) * sinTheta) \
                        + normals * cosTheta \
                        + U * -(torch.cos(phi) * sinTheta)  # [batch, num_samples, 1, 3]

        return L



    
    


if __name__=="__main__":
    conf = ConfigFactory.parse_file('./configs/default.conf')
    tracer = TracerO3d(conf)
    test_points = torch.tensor([-0.295, 0.104, -1.523]).unsqueeze(0)
    test_points = (test_points + 0.1 * torch.tensor([-1., 0., 0.]).unsqueeze(0)).cuda()
    radiance = tracer(test_points, torch.tensor([-1., 0., 0.]).unsqueeze(0).cuda())
    env_res = conf.get_list('train.env_res', default=[8,16])  # shape : (height, width)
    radiance = radiance[0].reshape(env_res[0], env_res[1], 3)
    print(radiance.shape)
    cv2.imwrite("../results/test_house/env_hemi_open3d.jpg", radiance.cpu().numpy()[:,:,::-1]*255.0)

