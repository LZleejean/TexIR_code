'''
@File    :   mat_mlp.py
@Time    :   2023/02/27 12:09:22
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

from models.incidentNet import IRNetwork, MatNetwork
from utils.sample_util import   TINY_NUMBER, generate_dir


class MaterialMLP(nn.Module):
    def __init__(self, conf, ids, extrinsics, optim_cam=False, gt_irf=True):
        super().__init__()
        # self.incident_radiance_network = IRNetwork(**conf.get_config('models.incident_radiance_network'))
        self.material_network = MatNetwork(**conf.get_config('models.material_network'))

        self.path_traced_mesh = conf.get_string('train.path_mesh_open3d')
        self.pano_res = conf.get_list('train.pano_img_res', default=[256,512])  # shape : (height, width)
        self.sample_l = conf.get_list('train.sample_light', default=[64,64])  # number of samples : (diffuse, specular)
        self.optim_cam = optim_cam
        self.ids = ids
        self.extrinsics = extrinsics

        self.object_list = self.generate_obj(self.path_traced_mesh)
        


        if optim_cam:
            self.param_extrinsics = {}
            for i in range(len(ids)):
                self.param_extrinsics.update({
                    self.ids[i]: nn.Parameter(self.extrinsics[i], requires_grad=True)
                })
            self.param_extrinsics = nn.ParameterDict(self.param_extrinsics)
        
        if gt_irf:
            self.path_traced_mesh = conf.get_string('train.path_mesh_open3d')
            # init ray casting scene
            trianglemesh = o3d.io.read_triangle_mesh(self.path_traced_mesh)     # o3d tracer must use the mesh with one texture map.
            trianglemesh.compute_vertex_normals()
            texture = cv2.imread(self.path_traced_mesh.replace("1.obj","_hdr_ccm.hdr"), -1)[:,:,::-1]
            texture = cv2.flip(texture, 0)
            texture = np.asarray(texture, np.float32)
            self.texture = (torch.from_numpy(texture).permute(2, 0, 1).unsqueeze(0).float()) # shape: (1, H, W, 3)
            self.texture = self.texture * (2**7)

            self.triangle_uvs = np.asarray(trianglemesh.triangle_uvs)
            mesh = o3d.t.geometry.TriangleMesh.from_legacy(trianglemesh)
            # Create a scene and add the triangle mesh.
            self.scene = o3d.t.geometry.RaycastingScene()
            self.scene.add_triangles(mesh)
            self.mat = None

        

        
    def forward(self, cam_to_world, id):
        """ assume that the batch_size is 1.

        Args:
            cam_to_world (torch.float32): shape: [4, 4].

        Returns:
            predicted_ir (torch.float32): shape: [1, h, w, 3]
        """

        if self.optim_cam:
            camera = pyredner.Camera(cam_to_world=self.param_extrinsics[id], 
                camera_type=pyredner.camera_type.panorama, 
                resolution=(self.pano_res[0],self.pano_res[1]),
                clip_near = 1e-2, # needs to > 0
                fisheye = False
                )
        else:
            camera = pyredner.Camera(cam_to_world=cam_to_world, 
                camera_type=pyredner.camera_type.panorama, 
                resolution=(self.pano_res[0],self.pano_res[1]),
                clip_near = 1e-2, # needs to > 0
                fisheye = False
                )
        
        scene = pyredner.Scene(camera=camera, objects=self.object_list)
        # imgs = pyredner.render_g_buffer(scene=scene, channels=[pyredner.channels.diffuse_reflectance], num_samples=[1,1]) # shape: (env_h, env_w, 3)
        # albedo: [0:3], roughness: [3:4], position: [4:7], normal: [7:10]
        g_buffers = pyredner.render_g_buffer(scene=scene, \
            channels=[
                pyredner.channels.position, 
                pyredner.channels.geometry_normal], num_samples=[2,2])
        positions = g_buffers[:,:,0:3].detach()+1e-2*g_buffers[:,:,3:6].detach()
        mat = self.material_network(positions)
        self.mat = mat
        img = self.render(g_buffers[:,:,3:6].detach(), mat[:,:,0:3], mat[:,:,3:4], positions, cam_to_world[0:3, -1])
        return img.unsqueeze(0)
    

    def generate_obj(self, path_mesh):
        object_list = pyredner.load_obj(path_mesh, obj_group=True, return_objects=True)
        
        # for object in object_list:
        #     object.vertices = object.vertices * torch.tensor([-1.0,-1.0,1.0]).unsqueeze(0).cuda()
        #     object.normals = object.normals * torch.tensor([-1.0,-1.0,1.0]).unsqueeze(0).cuda()
        
        return object_list
    
    def render(self, normal: torch.Tensor, albedo: torch.Tensor, roughness: torch.Tensor, points: torch.Tensor, cam_position: torch.Tensor):
        """render final color according to g buffers and IRF.

        Args:
            normal (torch.float32): [env_h, env_w, 3]
            albedo (torch.float32): [env_h, env_w, 3]
            roughness (torch.float32): [env_h, env_w, 1]
            points (torch.float32): [env_h, env_w, 3]
            cam_position (torch.float32): [3]
        """

        env_h, env_w, c = normal.shape
        normal = normal.reshape(-1, 3)
        albedo = albedo.reshape(-1, 3)
        roughness = roughness.reshape(-1, 1)
        points = points.reshape(-1, 3)
        view = F.normalize(cam_position.unsqueeze(0) - points, eps=1e-4)

        light_dir_diff = generate_dir(normal, self.sample_l[0])     # shape: [env_h*env_w, n_sample, 3]
        # with torch.no_grad():
        #     diffuse_lighting = hdr_recover(self.incident_radiance_network(points.unsqueeze(1).expand_as(light_dir_diff), light_dir_diff))
        diffuse_lighting = self.query_irf(points.unsqueeze(1).expand_as(light_dir_diff), light_dir_diff.unsqueeze(-2), self.sample_l[0])
        
        diffuse = self.diffuse_reflectance(diffuse_lighting, light_dir_diff, normal, albedo) / self.sample_l[0]
        # diffuse = torch.sum(diffuse_lighting, dim=1) / self.sample_l[0]

        h_dir_specular = generate_dir(normal, self.sample_l[1], 'importance', roughness)
        vdh = torch.clamp(torch.sum( h_dir_specular * view.unsqueeze(1), dim=-1,keepdim=True), 0.0, 1.0)    # shape: [b, n_sample, 1]
        light_dir_spec = 2  * vdh * h_dir_specular - view.unsqueeze(1)
        # with torch.no_grad():
        #     specular_lighting = hdr_recover(self.incident_radiance_network(points.unsqueeze(1).expand_as(h_dir_specular), light_dir_spec))
        specular_lighting = self.query_irf(points.unsqueeze(1).expand_as(light_dir_spec), light_dir_spec.unsqueeze(-2).detach(), self.sample_l[1])
        
        specular = self.specular_reflectance(specular_lighting, h_dir_specular, normal, view, light_dir_spec, roughness) / self.sample_l[1]

        return (diffuse + specular).reshape(env_h, env_w, 3)

    
    def diffuse_reflectance(self, lighting, l, n, albedo):
        ndl = torch.clamp(torch.sum( n.unsqueeze(1) * l, dim=-1,keepdim=True), 0.0, 1.0)    # shape: [b, n_sample, 1]
        brdf = albedo.unsqueeze(1) / np.pi

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
        mask = np.logical_or(hit < 1e-4, np.isfinite(hit))

        prim_id = intersections['primitive_ids'].numpy()    # shape: (b, num_sample, 1)
        prim_uvs = intersections['primitive_uvs'].numpy()    # shape: (b, num_sample, 1, 2)
        prim_uvs = np.clip(prim_uvs, 0., 1.)

        prim_id[~mask] = 0

        tmp = np.stack([prim_id*3+0, prim_id*3+1, prim_id*3+2], axis=0)    # shape: (3, b, num_sample, 1)
        tmp = tmp.reshape(-1)
        index = self.triangle_uvs[tmp]
        index = index.reshape(3, self.pano_res[0]*self.pano_res[1], num_sample,1, 2) # shape: (3, b, num_sample, 1, 2)
        grid = index[0,:,:,:,:] * (1-prim_uvs[:,:,:,0:1]-prim_uvs[:,:,:,1:2]) + index[1,:,:,:,:] * prim_uvs[:,:,:,0:1] + index[2,:,:,:,:] * prim_uvs[:,:,:,1:2] # shape: (b, num_sample, 1, 2)
        grid = torch.from_numpy(grid).float() # shape: (b, num_sample, 1, 2)
        grid[:,:,:,0:1] = grid[:,:,:,0:1]*2-1
        grid[:,:,:,1:2] = -(1-grid[:,:,:,1:2]*2)
        
        gt_ir = F.grid_sample(self.texture.expand([self.pano_res[0]*self.pano_res[1]]+list(self.texture.shape[1:])), grid, mode='bilinear', padding_mode="border",align_corners=False).permute(0,2,3,1) # shape: (b, num_sample, 1, 3)
        ner_mask = ~mask
        gt_ir[ner_mask,:] = 0
        gt_ir = gt_ir.reshape(self.pano_res[0]*self.pano_res[1], num_sample, 3)
        return gt_ir.cuda()


if __name__=="__main__":
    conf = ConfigFactory.parse_file('./configs/default.conf')
    mm = MaterialMLP(conf).cuda()
    radiance = mm().cpu().detach()
    print(radiance.shape)
    cv2.imwrite("../results/test_house/pano.jpg", radiance[0].numpy()[:,:,::-1]*255.0)

