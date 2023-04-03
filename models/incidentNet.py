import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import pyredner
pyredner.set_print_timing(False)

from siren_pytorch import SirenNet
import tinycudann as tcnn

from models.embedder import *



class IRNetwork(nn.Module):
    def __init__(self, points_multires=10, dirs_multires=4, dims=[128, 128, 128, 128], AABB=None):
        super().__init__()
        
        p_input_dim = 3
        self.p_embed_fn = None
        if points_multires > 0:
            self.p_embed_fn, p_input_dim = get_embedder(points_multires)
            # self.embeder_param, self.p_embed_fn, p_input_dim = get_hashgrid_embedder('sdf', AABB)
            # self.p_embed_fn, p_input_dim = get_frequency_embedder('material', AABB)
        
        dir_input_dim = 3
        self.dir_embed_fn = None
        if dirs_multires > 0:
            self.dir_embed_fn, dir_input_dim = get_embedder(dirs_multires)
            # embed_fn, input_ch = get_frequency_embedder('sdf', AABB)

        self.actv_fn = nn.ReLU(inplace=False)

        vis_layer = []
        dim = p_input_dim + dir_input_dim
        for i in range(len(dims)):
            vis_layer.append(nn.Linear(dim, dims[i]))
            vis_layer.append(self.actv_fn)
            dim = dims[i]
        vis_layer.append(nn.Linear(dim, 3))
        self.vis_layer = nn.Sequential(*vis_layer)

    def forward(self, points, view_dirs):
        if self.p_embed_fn is not None:
            points = self.p_embed_fn(points)
        if self.dir_embed_fn is not None:
            view_dirs = self.dir_embed_fn(view_dirs)

        vis = self.vis_layer(torch.cat([points, view_dirs], -1))
        # res = torch.clamp( vis , min=0.0)

        return vis

class IRSGNetwork(nn.Module):
    def __init__(self, points_multires=10, dims=[128, 128, 128, 128], AABB=None, num_lgt_sgs=24):
        super().__init__()
        
        p_input_dim = 3
        self.p_embed_fn = None
        if points_multires > 0:
            self.p_embed_fn, p_input_dim = get_embedder(points_multires)
            # self.embeder_param, self.p_embed_fn, p_input_dim = get_hashgrid_embedder('sdf', AABB)
            # self.p_embed_fn, p_input_dim = get_frequency_embedder('material', AABB)
        

        self.actv_fn = nn.ReLU(inplace=False)
        self.num_lgt_sgs = num_lgt_sgs

        sg_layer = []
        dim = p_input_dim
        for i in range(len(dims)):
            sg_layer.append(nn.Linear(dim, dims[i]))
            sg_layer.append(self.actv_fn)
            dim = dims[i]
        sg_layer.append(nn.Linear(dim, self.num_lgt_sgs * 6))
        self.sg_layer = nn.Sequential(*sg_layer)

    def forward(self, points):
        if self.p_embed_fn is not None:
            points = self.p_embed_fn(points)

        batch_size = points.shape[0]
        output = self.sg_layer(points).reshape(batch_size, self.num_lgt_sgs, 6)

        lgt_lobes = torch.sigmoid(output[..., :2])
        theta, phi = lgt_lobes[..., :1] * 2 * np.pi, lgt_lobes[..., 1:2] * 2 * np.pi
        
        lgt_lobes = torch.cat(
            [torch.cos(theta) * torch.sin(phi), torch.sin(theta) * torch.sin(phi), torch.cos(phi)], dim=-1)
        
        lambda_mu = output[..., 2:]
        # lambda_mu[..., :1] = torch.sigmoid(lambda_mu[..., :1]) * 30 + 0.1
        tmp1 = torch.sigmoid(lambda_mu[..., :1]) * 30 + 0.1
        # lambda_mu[..., 1:] = self.actv_fn(lambda_mu[..., 1:])
        tmp2 = self.actv_fn(lambda_mu[..., 1:])

        # lgt_sgs = torch.cat([lgt_lobes, lambda_mu], axis=-1)
        lgt_sgs = torch.cat([lgt_lobes, tmp1, tmp2], axis=-1)

        return lgt_sgs

class MatNetwork(nn.Module):
    def __init__(self, points_multires=10, p_input_dim = 3, p_out_dim = 4, dims=[128, 128, 128, 128], AABB=None):
        super().__init__()
        
        self.p_embed_fn = None
        if points_multires > 0:
            self.p_embed_fn, p_input_dim = get_embedder(points_multires)
            # self.embeder_param, self.p_embed_fn, p_input_dim = get_hashgrid_embedder('sdf', AABB)
            # self.p_embed_fn, p_input_dim = get_frequency_embedder('material', AABB)
        

        self.actv_fn = nn.LeakyReLU(0.01, inplace=False)
        self.ac_out = nn.Tanh()

        vis_layer = []
        dim = p_input_dim
        for i in range(len(dims)):
            vis_layer.append(nn.Linear(dim, dims[i]))
            vis_layer.append(self.actv_fn)
            dim = dims[i]
        vis_layer.append(nn.Linear(dim, p_out_dim))
        self.vis_layer = nn.Sequential(*vis_layer)

        self.vis_layer.apply(self._init_weights)

    def forward(self, points):
        if self.p_embed_fn is not None:
            points = self.p_embed_fn(points)

        vis = self.vis_layer(points)
        # res = (self.ac_out(vis) + 1.) * 0.5 * 0.9 + 0.09

        return vis
    
    @staticmethod
    def _init_weights(m):
        if type(m) == torch.nn.Linear:
            torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if hasattr(m.bias, 'data'):
                m.bias.data.fill_(0.0)


# class MatNetwork(nn.Module):
#     def __init__(self, points_multires=10, p_input_dim = 3, p_out_dim = 4, dims=[128, 128, 128, 128], AABB=None):
#         super().__init__()
        
        
#         self.skips = [4]
#         self.p_embed_fn = None
#         if points_multires > 0:
#             self.p_embed_fn, p_input_dim = get_embedder(points_multires)
#             # self.embeder_param, self.p_embed_fn, p_input_dim = get_hashgrid_embedder('sdf', AABB)
#             # self.p_embed_fn, p_input_dim = get_frequency_embedder('material', AABB)
        

#         self.actv_fn = nn.LeakyReLU(0.01, inplace=False)
#         self.ac_out = nn.Tanh()

#         # build the encoder
#         self.pts_linears = nn.ModuleList(
#             [nn.Linear(p_input_dim, dims[0])] + [nn.Linear(dims[i], dims[i]) if i not in self.skips else nn.Linear(dims[i] + p_input_dim, dims[i]) for i in
#                                         range(0, len(dims)-1)])
#         self.output_linear = nn.Linear(dims[-1], p_out_dim)

#         # vis_layer = []
#         # dim = p_input_dim
#         # for i in range(len(dims)):
#         #     vis_layer.append(nn.Linear(dim, dims[i]))
#         #     vis_layer.append(self.actv_fn)
#         #     dim = dims[i]
#         # vis_layer.append(nn.Linear(dim, p_out_dim))
#         # self.vis_layer = nn.Sequential(*vis_layer)

#         # self.vis_layer.apply(self._init_weights)

        

#     def forward(self, points):
#         if self.p_embed_fn is not None:
#             points = self.p_embed_fn(points)
#         h = points
#         for i, l in enumerate(self.pts_linears):
#             h = self.pts_linears[i](h)
#             h = F.relu(h)
#             if i in self.skips:
#                 h = torch.cat([points, h], -1)
#         vis = self.output_linear(h)
#         # res = (self.ac_out(vis) + 1.) * 0.5 * 0.9 + 0.09

#         return vis
    
#     @staticmethod
#     def _init_weights(m):
#         if type(m) == torch.nn.Linear:
#             torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
#             if hasattr(m.bias, 'data'):
#                 m.bias.data.fill_(0.0)


class PILNetwork(nn.Module):
    def __init__(self, points_multires=10, dirs_multires=4, p_input_dim = 3, p_out_dim = 3, dims=[128, 128, 128, 128], AABB=None):
        super().__init__()
        
        
        self.p_embed_fn = None
        if points_multires > 0:
            self.p_embed_fn, p_input_dim = get_embedder(points_multires)
            # self.embeder_param, self.p_embed_fn, p_input_dim = get_hashgrid_embedder('sdf', AABB)
            # self.p_embed_fn, p_input_dim = get_frequency_embedder('material', AABB)
        
        dir_input_dim = 3
        self.dir_embed_fn = None
        if dirs_multires > 0:
            self.dir_embed_fn, dir_input_dim = get_embedder(dirs_multires)
            # embed_fn, input_ch = get_frequency_embedder('sdf', AABB)

        self.actv_fn = nn.LeakyReLU(0.01, inplace=False)

        vis_layer = []
        dim = p_input_dim + dir_input_dim + 1
        for i in range(len(dims)):
            vis_layer.append(nn.Linear(dim, dims[i]))
            vis_layer.append(self.actv_fn)
            dim = dims[i]
        vis_layer.append(nn.Linear(dim, p_out_dim))
        self.vis_layer = nn.Sequential(*vis_layer)

    def forward(self, points, view_dirs, roughness):
        if self.p_embed_fn is not None:
            points = self.p_embed_fn(points)
        if self.dir_embed_fn is not None:
            view_dirs = self.dir_embed_fn(view_dirs)

        vis = self.vis_layer(torch.cat([points, view_dirs, roughness], -1))
        # res = torch.clamp( vis , min=0.0)

        return vis

# class MatNetwork(nn.Module):
#     def __init__(self, points_multires=10, p_input_dim = 3, p_out_dim = 4, dims=[128, 128, 128, 128], AABB=None):
#         super().__init__()
        
        
        
#         self.net = SirenNet(
#             dim_in = p_input_dim,                        # input dimension, ex. 2d coor
#             dim_hidden = dims[0],              # hidden dimension
#             dim_out = p_out_dim,                       # output dimension, ex. rgb value
#             num_layers = len(dims),                    # number of layers
#             final_activation = nn.Identity(),   # activation of final layer (nn.Identity() for direct output)
#             w0_initial = 30.                   # different signals may require different omega_0 in the first layer - this is a hyperparameter
#         )
#         if AABB is None:
#             raise Exception(" None AABB in SirenNet.")
#         self.AABB = torch.from_numpy(AABB).cuda()

#     def forward(self, points):
#         _inputs = (2*points - self.AABB[0:1,:] - self.AABB[1:2,:]) / (self.AABB[1:2,:] - self.AABB[0:1,:])
#         _inputs = torch.clamp(_inputs, min=-1., max=1.)
#         vis = self.net(points)

#         return vis


# the BRDF network in invrender. (https://github.com/zju3dv/InvRender/blob/d9e13d8e5337e4df363238fddf90f2038e792e7c/code/model/sg_envmap_material.py#L40)
class EnvmapMaterialNetwork(nn.Module):
    def __init__(self, multires=10, 
                 brdf_encoder_dims=[512, 512, 512, 512],
                 brdf_decoder_dims=[128, 128],
                 specular_albedo=0.02,
                 latent_dim=32):
        super().__init__()

        input_dim = 3
        self.embed_fn = None
        if multires > 0:
            self.brdf_embed_fn, brdf_input_dim = get_embedder(multires)

        self.latent_dim = latent_dim
        self.actv_fn = nn.LeakyReLU(0.2)
        ############## spatially-varying BRDF ############
        
        print('BRDF encoder network size: ', brdf_encoder_dims)
        print('BRDF decoder network size: ', brdf_decoder_dims)

        brdf_encoder_layer = []
        dim = brdf_input_dim
        for i in range(len(brdf_encoder_dims)):
            brdf_encoder_layer.append(nn.Linear(dim, brdf_encoder_dims[i]))
            brdf_encoder_layer.append(self.actv_fn)
            dim = brdf_encoder_dims[i]
        brdf_encoder_layer.append(nn.Linear(dim, self.latent_dim))
        self.brdf_encoder_layer = nn.Sequential(*brdf_encoder_layer)
        
        brdf_decoder_layer = []
        dim = self.latent_dim
        for i in range(len(brdf_decoder_dims)):
            brdf_decoder_layer.append(nn.Linear(dim, brdf_decoder_dims[i]))
            brdf_decoder_layer.append(self.actv_fn)
            dim = brdf_decoder_dims[i]
        brdf_decoder_layer.append(nn.Linear(dim, 4))
        self.brdf_decoder_layer = nn.Sequential(*brdf_decoder_layer)


    def forward(self, points):
        if self.brdf_embed_fn is not None:
            points = self.brdf_embed_fn(points)

        brdf_lc = torch.sigmoid(self.brdf_encoder_layer(points))
        brdf = torch.sigmoid(self.brdf_decoder_layer(brdf_lc))
        roughness = brdf[..., 3:] * 0.9 + 0.09
        diffuse_albedo = brdf[..., :3]

        rand_lc = brdf_lc + torch.randn(brdf_lc.shape).cuda() * 0.01
        random_xi_brdf = torch.sigmoid(self.brdf_decoder_layer(rand_lc))
        random_xi_roughness = random_xi_brdf[..., 3:] * 0.9 + 0.09
        random_xi_diffuse = random_xi_brdf[..., :3]

        ret = dict([
            ('roughness', roughness),
            ('diffuse_albedo', diffuse_albedo),
            ('random_xi_roughness', random_xi_roughness),
            ('random_xi_diffuse_albedo', random_xi_diffuse),
        ])
        return ret
    

# the BRDF network in nvdiffrec. (https://github.com/NVlabs/nvdiffrec/blob/3faedd23813ff6a34fd69d4d5b466eb0485c70e1/render/mlptexture.py#L18)
class _MLP(torch.nn.Module):
    def __init__(self, cfg, loss_scale=1.0):
        super(_MLP, self).__init__()
        self.loss_scale = loss_scale
        net = (torch.nn.Linear(cfg['n_input_dims'], cfg['n_neurons'], bias=False), torch.nn.ReLU())
        for i in range(cfg['n_hidden_layers']-1):
            net = net + (torch.nn.Linear(cfg['n_neurons'], cfg['n_neurons'], bias=False), torch.nn.ReLU())
        net = net + (torch.nn.Linear(cfg['n_neurons'], cfg['n_output_dims'], bias=False),)
        self.net = torch.nn.Sequential(*net).cuda()
        
        self.net.apply(self._init_weights)
        
        if self.loss_scale != 1.0:
            self.net.register_full_backward_hook(lambda module, grad_i, grad_o: (grad_i[0] * self.loss_scale, ))

    def forward(self, x):
        return self.net(x.float())

    @staticmethod
    def _init_weights(m):
        if type(m) == torch.nn.Linear:
            torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if hasattr(m.bias, 'data'):
                m.bias.data.fill_(0.0)

#######################################################################################################################################################
# Outward visible MLP class
#######################################################################################################################################################

class MLPTexture3D(torch.nn.Module):
    def __init__(self, AABB, channels = 4, internal_dims = 32, hidden = 2, min_max = None):
        super(MLPTexture3D, self).__init__()

        self.channels = channels
        self.internal_dims = internal_dims
        self.AABB = nn.Parameter(AABB, requires_grad=False)
        self.min_max = min_max
        min = torch.tensor([0.,0.,0., 0.01])
        max = torch.tensor([1.,1.,1., 1])
        self.min_max = nn.Parameter(torch.stack([min, max], dim=0), requires_grad=False)

        # Setup positional encoding, see https://github.com/NVlabs/tiny-cuda-nn for details
        desired_resolution = 4096
        base_grid_resolution = 16
        num_levels = 16
        per_level_scale = np.exp(np.log(desired_resolution / base_grid_resolution) / (num_levels-1))

        enc_cfg =  {
            "otype": "HashGrid",
            "n_levels": num_levels,
            "n_features_per_level": 2,
            "log2_hashmap_size": 19,
            "base_resolution": base_grid_resolution,
            "per_level_scale" : per_level_scale
	    }

        gradient_scaling = 128.0
        self.encoder = tcnn.Encoding(3, enc_cfg)
        self.encoder.register_backward_hook(lambda module, grad_i, grad_o: (grad_i[0] / gradient_scaling, ))

        # Setup MLP
        mlp_cfg = {
            "n_input_dims" : self.encoder.n_output_dims,
            "n_output_dims" : self.channels,
            "n_hidden_layers" : hidden,
            "n_neurons" : self.internal_dims
        }
        self.net = _MLP(mlp_cfg, gradient_scaling)
        print("Encoder output: %d dims" % (self.encoder.n_output_dims))

    # Sample texture at a given location
    def sample(self, points):
        _texc = (points.view(-1, 3) - self.AABB[0][None, ...]) / (self.AABB[1][None, ...] - self.AABB[0][None, ...])
        _texc = torch.clamp(_texc, min=0, max=1)
        
        p_enc = self.encoder(_texc.contiguous())
        out = self.net.forward(p_enc)

        # Sigmoid limit and scale to the allowed range
        out = torch.sigmoid(out) * (self.min_max[1][None, :] - self.min_max[0][None, :]) + self.min_max[0][None, :]

        return out.view(*points.shape[:-1], self.channels) # Remap to [n, h, w, c]

    # In-place clamp with no derivative to make sure values are in valid range after training
    def clamp_(self):
        pass



# the BRDF network in NeILF. (https://github.com/apple/ml-neilf/blob/b1fd650f283b2207981596e0caba7419238599c9/code/model/nn_arch.py#L10)
class SineLayer(nn.Module):
    ''' Siren layer '''
    
    def __init__(self, 
                 in_features, 
                 out_features, 
                 bias=True, 
                 is_first=False, 
                 omega_0=30, 
                 weight_norm=False):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

        if weight_norm:
            self.linear = nn.utils.weight_norm(self.linear)

    def init_weights(self):
        if self.is_first:
            nn.init.uniform_(self.linear.weight, 
                             -1 / self.in_features * self.omega_0, 
                             1 / self.in_features * self.omega_0)
        else:
            nn.init.uniform_(self.linear.weight, 
                             -np.sqrt(3 / self.in_features), 
                             np.sqrt(3 / self.in_features))
        nn.init.zeros_(self.linear.bias)

    def forward(self, input):
        return torch.sin(self.linear(input))

class BRDFMLP(nn.Module):

    def __init__(
            self,
            in_dims=3,
            out_dims=4,
            dims=[
                512,
                512,
                512,
                512,
                512,
                512,
                512,
                512
            ],
            skip_connection=[
                4
            ],
            weight_norm=False,
            multires_view=6
    ):
        super().__init__()
        self.init_range = np.sqrt(3 / dims[0])

        dims = [in_dims] + dims + [out_dims]
        first_omega = 30
        hidden_omega = 30

        self.embedview_fn = lambda x : x
        if multires_view > 0:
            embedview_fn, input_ch = get_embedder(multires_view)
            self.embedview_fn = embedview_fn
            dims[0] += (input_ch - in_dims)

        self.num_layers = len(dims)
        self.skip_connection = skip_connection

        for l in range(0, self.num_layers - 1):

            if l + 1 in self.skip_connection:
                out_dim = dims[l + 1] - dims[0] 
            else:
                out_dim = dims[l + 1]

            is_first = (l == 0) and (multires_view == 0)
            is_last = (l == (self.num_layers - 2))
            
            if not is_last:
                omega_0 = first_omega if is_first else hidden_omega
                lin = SineLayer(dims[l], out_dim, True, is_first, omega_0, weight_norm)
            else:
                lin = nn.Linear(dims[l], out_dim)
                nn.init.zeros_(lin.weight)
                nn.init.zeros_(lin.bias)
                if weight_norm:
                    lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

            self.last_active_fun = nn.Tanh()

    def forward(self, points):
        init_x = self.embedview_fn(points)
        x = init_x

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_connection:
                x = torch.cat([x, init_x], -1)
                
            x = lin(x)

        x = self.last_active_fun(x)
        # convert from [-1,1] to [0,1]
        x = x / 2 + 0.5                                                                     # [N, 5]
        return x


class NeILFMLP(nn.Module):

    def __init__(
            self,
            in_dims=3,
            out_dims=3,
            dims=[
                128,
                128,
                128,
                128,
                128,
                128,
                128,
                128
            ],
            skip_connection=[4],
            position_insertion=[4],
            weight_norm=False,
            multires_view=6
    ):
        super().__init__()
        self.init_range = np.sqrt(3 / dims[0])


        d_pos = 3
        d_dir = 3
        dims = dims + [out_dims]

        first_omega = 30
        hidden_omega = 30

        self.embedview_fn = lambda x : x
        if multires_view > 0:
            # embed view direction
            embedview_fn, input_ch = get_embedder(multires_view)
            self.embedview_fn = embedview_fn
            dims = [input_ch] + dims
        else:
            dims = [d_dir] + dims

        self.num_layers = len(dims)
        self.skip_connection = skip_connection
        self.position_insertion = position_insertion

        for l in range(0, self.num_layers - 1):

            out_dim = dims[l + 1]
            if l + 1 in self.skip_connection:
                out_dim = out_dim - dims[0]
            if l + 1 in self.position_insertion:
                out_dim = out_dim - d_pos

            is_first = (l == 0) and (multires_view == 0)
            is_last = (l == (self.num_layers - 2))
            
            if not is_last:
                omega_0 = first_omega if is_first else hidden_omega
                lin = SineLayer(dims[l], out_dim, True, is_first, omega_0, weight_norm)
            else:
                lin = nn.Linear(dims[l], out_dim)
                nn.init.zeros_(lin.weight)
                nn.init.constant_(lin.bias, np.log(1.5))
                if weight_norm:
                    lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)
            
            # because torch.exp can not backward, we use hdr_scale to replace it
            # self.last_active_fun = torch.exp
            self.last_active_fun = nn.ReLU(False)

    def forward(self, points):

        pose_embed = points[..., 0:3]
        view_embed = self.embedview_fn(points[..., 3:6])
        x = view_embed

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_connection:
                x = torch.cat([x, view_embed], -1)

            if l in self.position_insertion:
                x = torch.cat([x, pose_embed], -1)

            x = lin(x)
        # res = torch.relu(x)

        return x


