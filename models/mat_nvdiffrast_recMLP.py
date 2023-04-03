'''
@File    :   mat_nvdiffrast_recMLP.py
@Time    :   2023/02/27 12:09:47
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


from models.incidentNet import MLPTexture3D
from utils.sample_util import   TINY_NUMBER, generate_dir, generate_fixed_samples
from utils.general import hdr_recover


class MaterialModel(nn.Module):
    def __init__(self, conf, ids, extrinsics, optim_cam=False, gt_irf=True, gt_irrt=True):
        super().__init__()
        # self.incident_radiance_network = IRNetwork(**conf.get_config('models.incident_radiance_network'))
        # self.ir_radiance_network = MatNetwork(**conf.get_config('models.irrf_network'))
        

        self.resolution = conf.get_list('train.env_res', default=[8,16])  # shape : (height, width)

        self.path_traced_mesh = conf.get_string('train.path_mesh_open3d')
        self.pano_res = conf.get_list('train.pano_img_res', default=[1000,2000])  # shape : (height, width)
        self.cube_res = int(self.pano_res[1]/4)
        self.sample_l = conf.get_list('train.sample_light', default=[64,64])  # number of samples : (diffuse, specular)
        self.sample_type = conf.get_list('models.render.sample_type', default=['uniform','importance'])
        self.optim_cam = optim_cam
        self.ids = ids
        self.extrinsics = extrinsics
        self.conf = conf
        # assume that there only exists one mesh and one texture
        self.object_list = self.generate_obj(self.path_traced_mesh)[0]
        self.AABB = torch.stack([ torch.min(self.object_list.vertices, dim=0)[0], torch.max(self.object_list.vertices, dim=0)[0] ], dim=0)  # shape: (2, 3)
        # self.AABB = self.AABB.unsqueeze(1).unsqueeze(1).expand(-1, self.cube_res, self.cube_res, 3)
        self.material_network = MLPTexture3D(self.AABB)


        self.triangles = (nn.Parameter(self.object_list.indices, requires_grad=False))
        self.uvs = (nn.Parameter(self.object_list.uvs, requires_grad=False))
        self.uv_indices = (nn.Parameter(self.object_list.uv_indices, requires_grad=False))
        self.w = torch.ones([self.object_list.vertices.shape[0], 1])
        self.vertices = (nn.Parameter(torch.cat([self.object_list.vertices, self.w], dim=-1).repeat(6, 1, 1), requires_grad=False)) # shape: [5, n, 4]
        self.normals = (nn.Parameter(self.object_list.normals, requires_grad=False))    # shape: [n, 3]
        
        texture = self.object_list.material.diffuse_reflectance.texels

        # self.max_mip_level = int(np.log2(texture.shape[0]))
        self.max_mip_level = (get_mip_level(texture.shape[0]))

        self.gt_irrt = gt_irrt
        if gt_irrt:
            texture = cv2.imread(self.path_traced_mesh.replace("out1.obj","irt.hdr"), -1)[:,:,::-1]

            texture = np.asarray(texture, np.float32).copy()
            self.irrt = nn.Parameter(torch.from_numpy(texture), requires_grad=False)
        
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
        

        

        
    def forward(self, mvp, id, cam_position, stage=1):
        """ assume that the batch_size is 1.

        Args:
            mvp (torch.float32): shape: [6, 4, 4].
            cam_position (torch.float32): shape: [3]. 

        Returns:
            color (torch.float32): shape: [6, cube_res, cube_res, 3]
        """
        
        
        clip_vertexes = torch.einsum('ijk,ikl->ijl', self.vertices, mvp)
        
        rast_out, rast_out_db = dr.rasterize(self.glctx, clip_vertexes, self.triangles, resolution=[self.cube_res, self.cube_res])
        features = torch.cat([self.vertices[:,:,0:3], self.normals.unsqueeze(0).expand(6, -1, -1)], dim=-1)
        g_buffers, _ = dr.interpolate(features.contiguous(), rast_out, self.triangles) # [6, h, w, 6]
        # disable antialias, because the artifacts of edges will appear between different objects. 
        g_buffers = torch.where(rast_out[..., 3:] > 0, g_buffers, torch.tensor([1.,0.,0., 1.,0.,0.]).cuda()) # give a fix position and normal.
        mask = (rast_out[..., 3:] > 0).float()

        # get irr from irrt
        texc, texd = dr.interpolate(self.uvs[None, ...], rast_out, self.uv_indices, rast_db=rast_out_db, diff_attrs='all')
        irr = dr.texture(self.irrt[None, ...], texc, texd, filter_mode='linear-mipmap-linear', max_mip_level=self.max_mip_level)
        
        materials = self.material_network.sample(g_buffers[:,:,:,0:3])
        albedo = materials[...,0:3]
        roughness = materials[...,3:4]

        materials_jit = self.material_network.sample(g_buffers[:,:,:,0:3] + torch.normal(mean=0, std=0.01, size=g_buffers[:,:,:,0:3].shape, device="cuda"))

        kd_grad    = torch.sum(torch.abs(materials_jit[...,0:3] - albedo), dim=-1, keepdim=True) / 3    #shape：[b, h, w, 1]

        res = self.render(g_buffers[:,:,:,3:6].detach(), albedo, roughness, g_buffers[:,:,:,0:3].detach()+1e-2*g_buffers[:,:,:,3:6].detach(), cam_position, irr)
        
        res.update({
            'roughness': roughness,
            'kd_grad': kd_grad
        })

        return res
    

    def generate_obj(self, path_mesh):
        object_list = pyredner.load_obj(path_mesh, obj_group=True, return_objects=True)
        
        # for object in object_list:
        #     object.normals = pyredner.compute_vertex_normal(object.vertices, object.indices)
        
        return object_list
    
    def render(self, normal: torch.Tensor, albedo: torch.Tensor, roughness: torch.Tensor, points: torch.Tensor, cam_position: torch.Tensor, irr):
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
        irr = irr.reshape(-1, 3)
        view = F.normalize(cam_position.unsqueeze(0) - points, eps=1e-4)

        # with torch.no_grad():
        #     irr = hdr_recover(self.ir_radiance_network(points)) # shape : [b, 3]
        diffuse = irr * albedo / np.pi

        h_dir_specular = generate_dir(normal, self.sample_l[1], None, self.sample_type[1], roughness)
        vdh = torch.clamp(torch.sum( h_dir_specular * view.unsqueeze(1), dim=-1,keepdim=True), 0.0, 1.0)    # shape: [b, n_sample, 1]
        light_dir_spec = 2  * vdh * h_dir_specular - view.unsqueeze(1)
        specular_lighting = self.query_irf(points.unsqueeze(1).expand_as(light_dir_spec), light_dir_spec.unsqueeze(-2).detach(), self.sample_l[1])
        
        specular = self.specular_reflectance(specular_lighting, h_dir_specular, normal, view, light_dir_spec, roughness) / self.sample_l[1]

        res ={
            'rgb':  ( diffuse +specular ).reshape(face, h, w, 3),
            'albedo': albedo.reshape(face, h, w, 3),
            'normal': normal.reshape(face, h, w, 3).detach(),
            'position': (points).reshape(face, h, w, 3).detach()
        }
        return res

    
    def diffuse_reflectance(self, lighting, l, n, albedo, sample_type='uniform'):
        ndl = torch.clamp(torch.sum( n.unsqueeze(1) * l, dim=-1,keepdim=True), 0.0, 1.0)    # shape: [b, n_sample, 1]
        brdf = albedo.unsqueeze(1) / np.pi

        if sample_type=='cosine':
            return torch.sum( lighting * brdf * np.pi , dim=1)
        return torch.sum( lighting * brdf * ndl * 2*np.pi , dim=1)

    def specular_reflectance(self, lighting, h, n, v, l, roughness, albedo=None):
        
        vdh = torch.clamp(torch.sum( h * v.unsqueeze(1), dim=-1,keepdim=True), 0.0, 1.0)    # shape: [b, n_sample, 1]
        ndl = torch.clamp(torch.sum( n.unsqueeze(1) * l, dim=-1,keepdim=True), 0.0, 1.0)    # shape: [b, n_sample, 1]
        ndh = torch.clamp(torch.sum( n.unsqueeze(1) * h, dim=-1,keepdim=True), 0.0, 1.0)    # shape: [b, n_sample, 1]
        ndv = torch.clamp(torch.sum( n.unsqueeze(1) * v.unsqueeze(1), dim=-1,keepdim=True), 0.0, 1.0)    # shape: [b, n_sample, 1]
        vdl = torch.clamp(torch.sum( v.unsqueeze(1) * l, dim=-1,keepdim=True), 0.0, 1.0)    # shape: [b, n_sample, 1]

        f = 0.04 + 0.96 * torch.pow(2.0,((-5.55472*vdh-6.98316)*vdh))

        k = (roughness.unsqueeze(1) + 1) * (roughness.unsqueeze(1) + 1) / 8
        g1_ndv = ndv / torch.clamp( ndv *(1-k) + k , min=TINY_NUMBER)
        g1_ndl = ndl / torch.clamp( ndl *(1-k) + k , min=TINY_NUMBER)
        g = g1_ndl * g1_ndv

        # brdf: f * d * g / (4*ndl*ndv)
        brdf = f * g / torch.clamp(4 * ndl * ndv, min=TINY_NUMBER)

        # pdf : D*ndh / (4*vdh)
        # equation: L = lighing * brdf * ndl
        return torch.sum( lighting * brdf * ndl * 4 * vdh / torch.clamp(ndh, TINY_NUMBER), dim=1)


    def query_irf(self, points, directions, num_sample):
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
        index = index.reshape(3, self.cube_res*self.cube_res*6, num_sample,1, 2) # shape: (3, b, num_sample, 1, 2)
        grid = index[0,:,:,:,:] * (1-prim_uvs[:,:,:,0:1]-prim_uvs[:,:,:,1:2]) + index[1,:,:,:,:] * prim_uvs[:,:,:,0:1] + index[2,:,:,:,:] * prim_uvs[:,:,:,1:2] # shape: (b, num_sample, 1, 2)
        grid = torch.from_numpy(grid).float() # shape: (b, num_sample, 1, 2)
        grid[:,:,:,0:1] = grid[:,:,:,0:1]*2-1
        grid[:,:,:,1:2] = -(1-grid[:,:,:,1:2]*2)
        
        gt_ir = F.grid_sample(self.texture.expand([self.cube_res*self.cube_res*6]+list(self.texture.shape[1:])), grid, mode='bilinear', padding_mode="border",align_corners=False).permute(0,2,3,1) # shape: (b, num_sample, 1, 3)
        ner_mask = ~mask
        gt_ir[ner_mask,:] = 0
        gt_ir = gt_ir.reshape(self.cube_res*self.cube_res*6, num_sample, 3)
        return gt_ir.cuda()