'''
@File    :   test_redner.py
@Time    :   2023/02/27 12:10:10
@Author  :   Zhen Li 
@Contact :   yodlee@mail.nwpu.edu.cn
@Institution: Realsee
@License :   GNU General Public License v2.0
@Desc    :   None
'''


import functools
import cv2
import numpy as np
import os
import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
from pyhocon import ConfigFactory
import open3d as o3d

import pyredner
pyredner.set_use_gpu(False)

from utils.general import get_mip_level, rgb_to_intensity


from models.incidentNet import IRNetwork, MatNetwork
from utils.sample_util import   TINY_NUMBER, generate_dir, generate_fixed_samples
from utils.general import hdr_recover


class MaterialModel(nn.Module):
    def __init__(self, conf, cam_position_list, checkpoint_material, gt_irf=True, relighting=False):
        super().__init__()
        # self.incident_radiance_network = IRNetwork(**conf.get_config('models.incident_radiance_network'))
        self.ir_radiance_network = MatNetwork(**conf.get_config('models.irrf_network'))

        self.path_traced_mesh = conf.get_string('test.path_mesh_open3d')
        self.pano_res = conf.get_list('test.pano_img_res', default=[1000,2000])  # shape : (height, width)
        self.cube_res = int(self.pano_res[1]/4)
        self.sample_l = conf.get_list('test.sample_light', default=[64,64])  # number of samples : (diffuse, specular)
        self.sample_type = conf.get_list('models.render.sample_type', default=['uniform','importance'])

        self.conf = conf
        self.checkpoint_material = checkpoint_material
        self.cam_position_list = cam_position_list
        # assume that there only exists one mesh and one texture
        self.object_list = self.generate_obj(self.path_traced_mesh)[0]
        

        self.triangles = (nn.Parameter(self.object_list.indices, requires_grad=False))
        self.uvs = (nn.Parameter(self.object_list.uvs, requires_grad=False))
        self.uv_indices = (nn.Parameter(self.object_list.uv_indices, requires_grad=False))
        self.w = torch.ones([self.object_list.vertices.shape[0], 1])
        self.vertices = (nn.Parameter(torch.cat([self.object_list.vertices, self.w], dim=-1).repeat(6, 1, 1), requires_grad=False)) # shape: [5, n, 4]
        self.normals = (nn.Parameter(self.object_list.normals, requires_grad=False))    # shape: [n, 3]
        
        texture = self.object_list.material.diffuse_reflectance.texels
        # self.materials = (nn.Parameter(torch.ones(list(self.object_list.material.diffuse_reflectance.texels.shape[:-1]) + [4]) * 0.5, requires_grad=True))
        albedo_path = self.sort_res(self.checkpoint_material)
        roughness_path = albedo_path.replace('albedo','roughness')
        materials_a = cv2.imread(albedo_path, -1)[:,:,::-1]
        materials_a = np.asarray(materials_a, np.float32).copy()
        self.materials_a = (nn.Parameter(torch.from_numpy(materials_a), requires_grad=False))
        materials_r = cv2.imread(roughness_path, -1)[:,:,0:1]
        materials_r = np.asarray(materials_r, np.float32).copy()
        self.materials_r = (nn.Parameter(torch.from_numpy(materials_r), requires_grad=False))

        
        self.object_list.material.diffuse_reflectance = pyredner.Texture(self.materials_a)
        # the BRDF model we recovered is different with redner, thus we have to set specular reflectance.
        self.object_list.material.specular_reflectance = pyredner.Texture(torch.ones_like(self.materials_a)*1.)
        self.object_list.material.roughness = pyredner.Texture(self.materials_r/20)
        self.object_list = [self.object_list]
        self.add_light_source()

        texture_seg = cv2.imread(self.path_traced_mesh.replace("out1.obj","0_seg_gray.png"), -1)[:,:,0:1]
        texture_seg = np.asarray(texture_seg, np.float32).copy()
        self.texture_seg = (nn.Parameter(torch.from_numpy(texture_seg), requires_grad=False))

        # self.max_mip_level = int(np.log2(texture.shape[0]))
        self.max_mip_level = (get_mip_level(texture.shape[0]))
        
        # self.samples_diff = nn.Parameter(generate_fixed_samples(self.cube_res*self.cube_res*6, self.sample_l[0]), requires_grad=False)
        # self.samples_spec = nn.Parameter(generate_fixed_samples(self.cube_res*self.cube_res*6, self.sample_l[1]), requires_grad=False)
        
        self.relighting = relighting

        if gt_irf:
            self.path_traced_mesh = conf.get_string('test.path_mesh_open3d')
            # init ray casting scene
            trianglemesh = o3d.io.read_triangle_mesh(self.path_traced_mesh)     # o3d tracer must use the mesh with one texture map.
            # trianglemesh.compute_vertex_normals()
            texture = cv2.imread(self.path_traced_mesh.replace("out1.obj","hdr_texture.hdr"), -1)[:,:,::-1]
            texture = cv2.flip(texture, 0)
            texture = np.asarray(texture, np.float32)
            self.texture = (torch.from_numpy(texture).permute(2, 0, 1).unsqueeze(0).float()) # shape: (1, 3, H, W)
            self.texture = self.texture * (2**conf.get_float('test.hdr_exposure'))
            # texture = np.asarray(trianglemesh.textures[1])
            # self.texture = (torch.from_numpy(texture).permute(2, 0, 1).unsqueeze(0).float()/255.)**(2.2) # shape: (1, 3, H, W)

            if self.relighting:
                source = self.texture
                intensity = rgb_to_intensity(self.texture * (2**-conf.get_float('test.hdr_exposure')), dim=1).permute(0,2,3,1)
                self.texture = torch.where(intensity[...,0:1]>0.5, torch.tensor([2.14, 1.38, 0.2])* (2**conf.get_float('test.hdr_exposure')) , source.permute(0,2,3,1)).permute(0,3,1,2)

            self.triangle_uvs = np.asarray(trianglemesh.triangle_uvs)
            mesh = o3d.t.geometry.TriangleMesh.from_legacy(trianglemesh)
            # Create a scene and add the triangle mesh.
            self.scene = o3d.t.geometry.RaycastingScene()
            self.scene.add_triangles(mesh)
        
    
    def add_light_source(self):

        size=torch.tensor([0.5,0.5])
        intensity=torch.tensor([2.14, 1.38, 0.2])*6
        for cam_position in self.cam_position_list:
            position = torch.tensor([0., -0.5, 0.]) + cam_position
            light_source = pyredner.generate_quad_light(position, cam_position, size, intensity)
            self.object_list.append(light_source)


    def sort_res(self, checkpoint_material):
        def compare(A, B):
            epochA = os.path.splitext(A)[0].split('_')[-1]
            epochB = os.path.splitext(B)[0].split('_')[-1]
            if int(epochA) > int(epochB):
                return -1
            return 1
        
        all_path = glob.glob('{}/mat_albedo-1_*'.format(checkpoint_material))
        all_path.sort(key=functools.cmp_to_key(compare))

        return all_path[0]

        
    def forward(self, cam_to_world, id, cam_position, editing=True):
        """ assume that the batch_size is 1.

        Args:
            mvp (torch.float32): shape: [6, 4, 4].
            cam_position (torch.float32): shape: [3]. 

        Returns:
            color (torch.float32): shape: [h, w, 3]
        """
        
        camera = pyredner.Camera(cam_to_world=cam_to_world, 
                camera_type=pyredner.camera_type.panorama, 
                resolution=(self.pano_res[0],self.pano_res[1]),
                clip_near = 1e-2, # needs to > 0
                fisheye = False
                )
        scene = pyredner.Scene(camera=camera, objects=self.object_list)
        # imgs = pyredner.render_g_buffer(scene=scene, channels=[pyredner.channels.diffuse_reflectance], num_samples=[1,1]) # shape: (env_h, env_w, 3)
        # albedo: [0:3], roughness: [3:4], position: [4:7], normal: [7:10]
        rgb = pyredner.render_pathtracing(scene=scene, \
            max_bounces=3, num_samples=[64,1])
        
        res = {"rgb": rgb}

        return res
    

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

        if self.relighting:
            light_dir_diff = generate_dir(normal, self.sample_l[0], None, mode=self.sample_type[0])     # shape: [env_h*env_w, n_sample, 3]
            # with torch.no_grad():
            #     diffuse_lighting = hdr_recover(self.incident_radiance_network(points.unsqueeze(1).expand_as(light_dir_diff), light_dir_diff))#* (2**7)
            diffuse_lighting = self.query_irf(points.unsqueeze(1).expand_as(light_dir_diff), light_dir_diff.unsqueeze(-2), self.sample_l[0])
            
            diffuse = self.diffuse_reflectance(diffuse_lighting, light_dir_diff, normal, albedo, self.sample_type[0]) / self.sample_l[0]
            # diffuse = torch.sum(diffuse_lighting, dim=1) / self.sample_l[0]
        else:
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