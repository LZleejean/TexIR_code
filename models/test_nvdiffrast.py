'''
@File    :   test_nvdiffrast.py
@Time    :   2023/02/27 12:10:04
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

import nvdiffrast.torch as dr

import pyredner
pyredner.set_use_gpu(False)

from utils.general import get_mip_level, rgb_to_intensity


from models.incidentNet import IRNetwork, MatNetwork
from utils.sample_util import   TINY_NUMBER, generate_dir, generate_fixed_samples
from utils.general import hdr_recover


class MaterialModel(nn.Module):
    def __init__(self, conf, cam_position_list, checkpoint_material, gt_irf=True, relighting=False, gt_irrt=True):
        super().__init__()
        # self.incident_radiance_network = IRNetwork(**conf.get_config('models.incident_radiance_network'))
        # self.ir_radiance_network = MatNetwork(**conf.get_config('models.irrf_network'))

        self.path_traced_mesh = conf.get_string('test.path_mesh_open3d')
        self.pano_res = conf.get_list('test.pano_img_res', default=[1000,2000])  # shape : (height, width)
        self.cube_res = int(self.pano_res[1]/4)
        self.sample_l = conf.get_list('test.sample_light', default=[64,64])  # number of samples : (diffuse, specular)
        self.sample_type = conf.get_list('models.render.sample_type', default=['uniform','importance'])

        self.conf = conf
        self.checkpoint_material = checkpoint_material
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

        self.gt_irrt = gt_irrt
        if gt_irrt:
            texture = cv2.imread(self.path_traced_mesh.replace("out1.obj","irt.hdr"), -1)[:,:,::-1]

            texture = np.asarray(texture, np.float32).copy()
            self.irrt = nn.Parameter(torch.from_numpy(texture), requires_grad=False)

        texture_seg = cv2.imread(self.path_traced_mesh.replace("out1.obj","0_seg_gray.png"), -1)[:,:,0:1]
        texture_seg = np.asarray(texture_seg, np.float32).copy()
        self.texture_seg = (nn.Parameter(torch.from_numpy(texture_seg), requires_grad=False))

        # self.max_mip_level = int(np.log2(texture.shape[0]))
        self.max_mip_level = (get_mip_level(texture.shape[0]))
        
        # self.samples_diff = nn.Parameter(generate_fixed_samples(self.cube_res*self.cube_res*6, self.sample_l[0]), requires_grad=False)
        # self.samples_spec = nn.Parameter(generate_fixed_samples(self.cube_res*self.cube_res*6, self.sample_l[1]), requires_grad=False)
        
        self.glctx = dr.RasterizeGLContext()

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

        
    def forward(self, mvp, id, cam_position, editing=True, albedo_floor=None, albedo_wall=None, roughness_floor=None):
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
        
        texc, texd = dr.interpolate(self.uvs[None, ...], rast_out, self.uv_indices, rast_db=rast_out_db, diff_attrs='all')
        # enable mipmap
        # materials = dr.texture(torch.cat([self.materials_a, self.materials_r], dim=-1)[None, ...], texc, texd, filter_mode='linear-mipmap-linear', max_mip_level=self.max_mip_level)
        albedo = dr.texture(self.materials_a[None, ...], texc, texd, filter_mode='linear-mipmap-linear', max_mip_level=self.max_mip_level)
        seg = dr.texture(self.texture_seg[None, ...], texc, texd, filter_mode='linear')
        
        roughness = dr.texture(self.materials_r[None, ...], texc, texd, filter_mode='linear-mipmap-linear', max_mip_level=self.max_mip_level)
        # disable mipmap
        # materials = dr.texture(torch.cat([self.materials_a, self.materials_r], dim=-1)[None, ...], texc, texd, filter_mode='linear', boundary_mode='zero',max_mip_level=0)
        # get irr from irrt
        irr = dr.texture(self.irrt[None, ...], texc, texd, filter_mode='linear-mipmap-linear', max_mip_level=self.max_mip_level)

        if editing:
            # # for hdrhouse 1
            # # wall into red
            # albedo = torch.where(seg==45.0, torch.tensor([0.52, 0.00, 0.00]).cuda(), albedo)
            # # floor into rough
            # roughness = torch.where(seg==46.0, torch.tensor([0.3]).cuda(), roughness)

            # # for hdrhouse 2
            # # floor into grey
            # albedo = torch.where(seg==46.0, torch.tensor([0.41, 0.38, 0.35]).cuda(), albedo)
            # # sofa into red
            # albedo = torch.where(seg==16.0, torch.tensor([0.52, 0.00, 0.00]).cuda(), albedo)

            # # for hdrhouse 3
            # # floor into green
            # albedo = torch.where(seg==46.0, torch.tensor([0.00, 0.38, 0.00]).cuda(), albedo)
            # # bed into blue
            # albedo = torch.where(seg==6.0, torch.tensor([0.48, 0.63, 0.73]).cuda(), albedo)

            # # for hdrhouse 4
            # # wall into glossy
            # roughness = torch.where(seg==45.0, torch.tensor([0.1]).cuda(), roughness)
            # # floor into glossy
            # roughness = torch.where(seg==46.0, torch.tensor([0.1]).cuda(), roughness)

            # # for hdrhouse 5
            # # table into red
            # albedo = torch.where(seg==22.0, torch.tensor([150,0,0]).cuda()/255., albedo)
            # # floor into skyblue
            # albedo = torch.where(seg==46.0, torch.tensor([135,206,235]).cuda()/255., albedo)

            # # for hdrhouse 6
            # # table into darkorange
            # albedo = torch.where(seg==22.0, torch.tensor([255,140,0]).cuda()/255., albedo)
            # # floor into glossy
            # roughness = torch.where(seg==46.0, torch.tensor([0.1]).cuda(), roughness)

            # # for hdrhouse 7
            # # wall into gold
            # albedo = torch.where(seg==45.0, torch.tensor([255,215,0]).cuda()/255., albedo)
            # # floor into lightpink
            # albedo = torch.where(seg==46.0, torch.tensor([255,182,193]).cuda()/255., albedo)

            # # for hdrhouse 8
            # # wall into blue
            # albedo = torch.where(seg==45.0, torch.tensor([0.48, 0.63, 0.73]).cuda(), albedo)
            # # floor into lightgreen
            # albedo = torch.where(seg==46.0, torch.tensor([144,238,144]).cuda()/255., albedo)
            # # roughness = torch.where(seg==46.0, torch.tensor([0.1]).cuda(), roughness)

            # # for hdrhouse 9
            # # floor into wood
            # albedo = torch.where(seg==46.0, torch.tensor([0.52, 0.30, 0.08]).cuda(), albedo)
            # # wall into pink
            # albedo = torch.where(seg==45.0, torch.tensor([0.81, 0.60, 0.63]).cuda(), albedo)
            
            if albedo_floor is not None:
                # for hdrhouse panoid 6
                albedo = torch.where(seg==46.0, torch.tensor(albedo_floor).cuda(), albedo)
                albedo = torch.where(seg==45.0, torch.tensor(albedo_wall).cuda(), albedo)
            if roughness_floor is not None:
                roughness = torch.where(seg==46.0, torch.tensor(roughness_floor).cuda(), roughness)

            # # for hdrhouse 11
            # # sofa into lightgreen
            # albedo = torch.where(seg==16.0, torch.tensor([144,238,144]).cuda()/255., albedo)
            # # floor into lightcoral
            # albedo = torch.where(seg==46.0, torch.tensor([240,128,128]).cuda()/255., albedo)

            # # for syn
            # # wall into magenta
            # albedo = torch.where(seg==45.0, torch.tensor([255,0,255]).cuda()/255., albedo)
            # # floor into darkcyan
            # albedo = torch.where(seg==46.0, torch.tensor([0,139,139]).cuda()/255., albedo)

            

        res = self.render(g_buffers[:,:,:,3:6].detach(), albedo, roughness, g_buffers[:,:,:,0:3].detach()+1e-2*g_buffers[:,:,:,3:6].detach(), cam_position, irr)

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

        if self.relighting:
            light_dir_diff = generate_dir(normal, self.sample_l[0], None, mode=self.sample_type[0])     # shape: [env_h*env_w, n_sample, 3]
            # with torch.no_grad():
            #     diffuse_lighting = hdr_recover(self.incident_radiance_network(points.unsqueeze(1).expand_as(light_dir_diff), light_dir_diff))#* (2**7)
            diffuse_lighting = self.query_irf(points.unsqueeze(1).expand_as(light_dir_diff), light_dir_diff.unsqueeze(-2), self.sample_l[0])
            
            diffuse = self.diffuse_reflectance(diffuse_lighting, light_dir_diff, normal, albedo, self.sample_type[0]) / self.sample_l[0]
            # diffuse = torch.sum(diffuse_lighting, dim=1) / self.sample_l[0]
        else:
            # with torch.no_grad():
            #     irr = hdr_recover(self.ir_radiance_network(points)) # shape : [b, 3]
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