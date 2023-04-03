'''
@File    :   tracer_o3d_irt.py
@Time    :   2023/02/27 12:10:20
@Author  :   Zhen Li 
@Contact :   yodlee@mail.nwpu.edu.cn
@Institution: Realsee
@License :   GNU General Public License v2.0
@Desc    :   None
'''


import cv2
import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from pyhocon import ConfigFactory
import open3d as o3d

import nvdiffrast.torch as dr

import pyredner
pyredner.set_use_gpu(False)

from utils.general import get_mip_level, rgb_to_intensity


from utils.sample_util import   TINY_NUMBER, generate_dir, generate_fixed_samples
from utils.general import hdr_recover
from utils.Cube2Pano import Cube2Pano


class TracerO3d(nn.Module):
    def __init__(self, conf, ids, extrinsics, optim_cam=False, gt_irf=True):
        super().__init__()
        
        self.resolution = conf.get_list('train.env_res', default=[8,16])  # shape : (height, width)

        self.path_traced_mesh = conf.get_string('train.path_mesh_open3d')
        self.pano_res = conf.get_list('train.pano_img_res', default=[1000,2000])  # shape : (height, width)
        
        self.sample_l = conf.get_list('train.sample_light', default=[64,64])  # number of samples : (diffuse, specular)
        self.sample_type = conf.get_list('models.render.sample_type', default=['uniform','importance'])
        self.optim_cam = optim_cam
        self.ids = ids
        self.extrinsics = extrinsics
        self.conf = conf
        # assume that there only exists one mesh and one texture
        self.object_list = self.generate_obj(self.path_traced_mesh)[0]
        self.cube_res = 256
        self.cube2pano = Cube2Pano(pano_width=1024, pano_height=512, cube_lenth=self.cube_res, cube_channel=6, is_cuda=True)
        

        self.triangles = (nn.Parameter(self.object_list.indices, requires_grad=False))
        self.uvs = (nn.Parameter(self.object_list.uvs, requires_grad=False))
        self.uv_indices = (nn.Parameter(self.object_list.uv_indices, requires_grad=False))
        self.w = torch.ones([self.object_list.vertices.shape[0], 1])
        self.vertices = (nn.Parameter(torch.cat([self.object_list.vertices, self.w], dim=-1).repeat(6, 1, 1), requires_grad=False)) # shape: [5, n, 4]
        self.normals = (nn.Parameter(self.object_list.normals, requires_grad=False))    # shape: [n, 3]
        
        texture = self.object_list.material.diffuse_reflectance.texels
        # self.max_mip_level = int(np.log2(texture.shape[0]))
        self.max_mip_level = (get_mip_level(texture.shape[0]))
        
        # self.samples_diff = nn.Parameter(generate_fixed_samples(self.cube_res*self.cube_res*6, self.sample_l[0]), requires_grad=False)
        # self.samples_spec = nn.Parameter(generate_fixed_samples(self.cube_res*self.cube_res*6, self.sample_l[1]), requires_grad=False)
        
        self.glctx = dr.RasterizeGLContext()

        if gt_irf:
            self.path_traced_mesh = conf.get_string('train.path_mesh_open3d')
            # init ray casting scene
            trianglemesh = o3d.io.read_triangle_mesh(self.path_traced_mesh)     # o3d tracer must use the mesh with one texture map.
            # trianglemesh.compute_vertex_normals()
            texture = cv2.imread(self.path_traced_mesh.replace("out1.obj","hdr_texture.hdr"), -1)[:,:,::-1]
            texture = cv2.flip(texture, 0)
            texture = np.asarray(texture, np.float32)
            self.texture = (torch.from_numpy(texture).permute(2, 0, 1).unsqueeze(0).float()) # shape: (1, 3, H, W)
            self.texture = self.texture * (2**conf.get_float('train.hdr_exposure'))
            # texture = np.asarray(trianglemesh.textures[1])
            # self.texture = (torch.from_numpy(texture).permute(2, 0, 1).unsqueeze(0).float()/255.)**(2.2) # shape: (1, 3, H, W)

            self.triangle_uvs = np.asarray(trianglemesh.triangle_uvs)
            mesh = o3d.t.geometry.TriangleMesh.from_legacy(trianglemesh)
            # Create a scene and add the triangle mesh.
            self.scene = o3d.t.geometry.RaycastingScene()
            self.scene.add_triangles(mesh)
        
        index_texture = cv2.imread(self.path_traced_mesh.replace("out1.obj","0.png"), -1)
        # kernel = np.ones((11,11), np.uint16)
        # index_texture = cv2.dilate(index_texture, kernel)
        # resize the size of ir texture into 1024*1024
        self.index_texture = cv2.resize(index_texture, (1024,1024), cv2.INTER_NEAREST)
        


    def generate_positions(self):
        self.position_normal_list = []
        for i in range(len(self.ids)):
            clip_vertexes = torch.einsum('ijk,ikl->ijl', self.vertices, self.extrinsics[i].cuda())
        
            rast_out, rast_out_db = dr.rasterize(self.glctx, clip_vertexes, self.triangles, resolution=[self.cube_res, self.cube_res])
            features = torch.cat([self.vertices[:,:,0:3], self.normals.unsqueeze(0).expand(6, -1, -1)], dim=-1)
            g_buffers, _ = dr.interpolate(features.contiguous(), rast_out, self.triangles) # [6, h, w, 6]
            # disable antialias, because the artifacts of edges will appear between different objects. 
            g_buffers = torch.where(rast_out[..., 3:] > 0, g_buffers, torch.tensor([1.,0.,0., 1.,0.,0.]).cuda()) # give a fix position and normal.

            g_buffers[...,0:3] = g_buffers[...,0:3] + 1e-2*g_buffers[...,3:]
            res = self.cube2pano.ToPano(g_buffers.permute(0,3,1,2).reshape(1, -1, self.cube_res, self.cube_res))[0].permute(1,2,0)  # shape (h, w, 3)
            self.position_normal_list.append(res)

    
    def calcute_position_normal_texture(self):
        position_map = np.zeros(self.index_texture.shape, dtype=np.float32)
        normal_map = np.zeros(self.index_texture.shape, dtype=np.float32)

        hdrIdList = np.unique(self.index_texture[:, :, 2])
        for hdrId in hdrIdList:

            # if hdrId == 0:
            #     continue

            rows, cols = np.where(self.index_texture[:, :, 2] == hdrId)
            hdr = self.position_normal_list[hdrId].cpu().numpy()
            height, width, bands = hdr.shape

            # source width 6720, source height 3360
            hdrCol = (self.index_texture[rows, cols, 1] / 50000 * width).astype(np.int)
            hdrCol = np.clip(hdrCol, 0, width - 1)
            hdrRow = (self.index_texture[rows, cols, 0] / 50000 * height).astype(np.int)
            hdrRow = np.clip(hdrRow, 0, height - 1)
            position_map[rows, cols, :] = hdr[hdrRow.astype(np.int), hdrCol.astype(np.int), 0:3]
            normal_map[rows, cols, :] = hdr[hdrRow.astype(np.int), hdrCol.astype(np.int), 3:6]
            
            h_seams, w_seams = np.where((self.index_texture[:,:,0]+self.index_texture[:,:,1]+self.index_texture[:,:,2]) == 0)
            position_map[h_seams, w_seams] = np.array([0,0,0],dtype=np.float32)[np.newaxis,:]
            normal_map[h_seams, w_seams] = np.array([0,0,0],dtype=np.float32)[np.newaxis,:]
        
        self.position_texture = torch.from_numpy(position_map).cuda()
        self.normal_texture = torch.from_numpy(normal_map).cuda()

        
    def forward(self):
        self.generate_positions()
        self.ir_texture = torch.from_numpy(np.zeros_like(self.index_texture, dtype=np.float32))
        self.calcute_position_normal_texture()

        # target_texture_path = self.conf.get_string('train.path_mesh_open3d').replace('out1.obj', '0_normal_texture.exr')

        # irr_texture_numpy = self.normal_texture.cpu().numpy()[:,:,::-1]
        # cv2.imwrite(target_texture_path, irr_texture_numpy)
        print("Finish precomputing model!")

        batch = 512
        position_texture = self.position_texture.reshape(-1, 3)
        normal_texture = self.normal_texture.reshape(-1, 3)

        
        assert position_texture.shape[0]%512 ==0
        step = int(position_texture.shape[0]/512)
        irr_texture = torch.zeros_like(position_texture).cuda()

        for i in range(step):
            position = position_texture[int(i*batch):int((i+1)*batch), :]
            normal = normal_texture[int(i*batch):int((i+1)*batch), :]   # shape: [batch, 3]
            light_dir_diff = generate_dir(normal, self.sample_l[0], None, self.sample_type[0], None)    # shape: [batch, num_samples, 3]
            gt_ir = self.query_irf(position.unsqueeze(1).expand_as(light_dir_diff), light_dir_diff.unsqueeze(-2), self.sample_l[0])
            ndl = torch.clamp(torch.sum( normal.unsqueeze(1) * light_dir_diff.reshape(batch, -1, 3), dim=-1, keepdim=True), 0.0, 1.0)    # shape: [b, n_sample, 1]
            gt_irr = torch.sum(gt_ir.reshape(batch, -1, 3).cuda() * ndl, dim=1) * 2 * np.pi / self.sample_l[0]
            
            irr_texture[int(i*batch):int((i+1)*batch), :] = gt_irr
        irr_texture = irr_texture.reshape(list(self.position_texture.shape))
        
        irr_texture_numpy = irr_texture.cpu().numpy()
        h_seams, w_seams = np.where((self.index_texture[:,:,0]+self.index_texture[:,:,1]+self.index_texture[:,:,2]) == 0)
        irr_texture_numpy[h_seams, w_seams] = np.array([0,0,0],dtype=np.float32)[np.newaxis,:]
        
        return torch.from_numpy(irr_texture_numpy).cuda()
    

    def generate_obj(self, path_mesh):
        object_list = pyredner.load_obj(path_mesh, obj_group=True, return_objects=True)
        
        # for object in object_list:
        #     object.normals = pyredner.compute_vertex_normal(object.vertices, object.indices)
        
        return object_list
    
    def render(self, normal: torch.Tensor, albedo: torch.Tensor, roughness: torch.Tensor, points: torch.Tensor, cam_position: torch.Tensor):
        """render final color according to g buffers and IRF.

        Args:
            normal (torch.float32): [6, cube_len, cube_len, 3]
            albedo (torch.float32): [6, cube_len, cube_len, 3]
            roughness (torch.float32): [6, cube_len, cube_len, 1]
            points (torch.float32): [6, cube_len, cube_len, 3]
            cam_position (torch.float32): [3]
        """

        face, h, w, c = normal.shape
        normal = normal.reshape(-1, 3)
        albedo = albedo.reshape(-1, 3)
        roughness = roughness.reshape(-1, 1)
        points = points.reshape(-1, 3)
        view = F.normalize(cam_position.unsqueeze(0) - points, eps=1e-4)

        # light_dir_diff = generate_dir(normal, self.sample_l[0], None, mode=self.sample_type[0])     # shape: [env_h*env_w, n_sample, 3]
        # # with torch.no_grad():
        # #     diffuse_lighting = hdr_recover(self.incident_radiance_network(points.unsqueeze(1).expand_as(light_dir_diff), light_dir_diff))#* (2**7)
        # diffuse_lighting = self.query_irf(points.unsqueeze(1).expand_as(light_dir_diff), light_dir_diff.unsqueeze(-2), self.sample_l[0])
        
        # diffuse = self.diffuse_reflectance(diffuse_lighting, light_dir_diff, normal, albedo, self.sample_type[0]) / self.sample_l[0]
        # diffuse = torch.sum(diffuse_lighting, dim=1) / self.sample_l[0]
        
        with torch.no_grad():
            irr = hdr_recover(self.ir_radiance_network(points)) # shape : [b, 3]
        diffuse = irr * albedo / np.pi

        h_dir_specular = generate_dir(normal, self.sample_l[1], None, self.sample_type[1], roughness)
        vdh = torch.clamp(torch.sum( h_dir_specular * view.unsqueeze(1), dim=-1,keepdim=True), 0.0, 1.0)    # shape: [b, n_sample, 1]
        light_dir_spec = 2  * vdh * h_dir_specular - view.unsqueeze(1)
        # with torch.no_grad():
        #     specular_lighting = hdr_recover(self.incident_radiance_network(points.unsqueeze(1).expand_as(h_dir_specular), light_dir_spec))#* (2**7)
        specular_lighting = self.query_irf(points.unsqueeze(1).expand_as(light_dir_spec), light_dir_spec.unsqueeze(-2).detach(), self.sample_l[1])
        
        specular = self.specular_reflectance(specular_lighting, h_dir_specular, normal, view, light_dir_spec, roughness) / self.sample_l[1]

        res ={
            'rgb':  ( diffuse +specular ).reshape(face, h, w, 3),
            'albedo': albedo.reshape(face, h, w, 3),
            'normal': normal.reshape(face, h, w, 3).detach(),
            'position': (points+2e-2*normal).reshape(face, h, w, 3).detach()
        }
        return res



    def query_irf(self, points, directions, num_sample):
        batch, n_sample, c = points.shape
        rays = torch.cat([points.unsqueeze(-2).expand_as(directions), directions], dim=-1)
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
        index = index.reshape(3, batch, num_sample,1, 2) # shape: (3, b, num_sample, 1, 2)
        grid = index[0,:,:,:,:] * (1-prim_uvs[:,:,:,0:1]-prim_uvs[:,:,:,1:2]) + index[1,:,:,:,:] * prim_uvs[:,:,:,0:1] + index[2,:,:,:,:] * prim_uvs[:,:,:,1:2] # shape: (b, num_sample, 1, 2)
        grid = torch.from_numpy(grid).float() # shape: (b, num_sample, 1, 2)
        grid[:,:,:,0:1] = grid[:,:,:,0:1]*2-1
        grid[:,:,:,1:2] = -(1-grid[:,:,:,1:2]*2)
        
        gt_ir = F.grid_sample(self.texture.expand([batch]+list(self.texture.shape[1:])), grid, mode='bilinear', padding_mode="border",align_corners=False).permute(0,2,3,1) # shape: (b, num_sample, 1, 3)
        ner_mask = ~mask
        gt_ir[ner_mask,:] = 0
        gt_ir = gt_ir.reshape(batch, num_sample, 3)
        return gt_ir