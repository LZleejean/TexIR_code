import torch
# import tinycudann as tcnn
import numpy as np

""" Positional encoding embedding. Code was taken from https://github.com/bmild/nerf. """
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn,
                                 freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim
        print("Encoder output: %d dims" % (out_dim))

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires):
    embed_kwargs = {
        'include_input': True,
        'input_dims': 3,
        'max_freq_log2': multires-1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    def embed(x, eo=embedder_obj): return eo.embed(x)
    return embed, embedder_obj.out_dim



class HashGridEmbedder(torch.nn.Module):
    def __init__(self, **kwargs):
        super(HashGridEmbedder, self).__init__()
        self.kwargs = kwargs
    
        if self.kwargs['mode']=='material':

            desired_resolution = 4096
            base_grid_resolution = 16
            num_levels = 16
            per_level_scale = np.exp(np.log(desired_resolution / base_grid_resolution) / (num_levels-1))
        elif self.kwargs['mode']=='sdf':
            desired_resolution = 2048*256   # assume our scene size is 256, the corresponding per level scale is 2.0
            base_grid_resolution = 16
            num_levels = 16
            per_level_scale = np.exp(np.log(desired_resolution / base_grid_resolution) / (num_levels-1))
        else:
            raise Exception("unrecognized embedder mode, please use material or sdf!")

        enc_cfg =  {
                "otype": "HashGrid",
                "n_levels": 16,
                "n_features_per_level": 2,
                "log2_hashmap_size": 19,
                "base_resolution": base_grid_resolution,
                "per_level_scale" : per_level_scale
            }
        self.encoder = tcnn.Encoding(self.kwargs['input_dims'], enc_cfg)
        print("Encoder output: %d dims" % (self.encoder.n_output_dims))
        if self.kwargs['AABB'] is None:
            raise Exception(" None AABB.")
        self.AABB = torch.from_numpy(self.kwargs['AABB']).cuda()
        self.max_len = torch.max(torch.abs(self.AABB))*2
        # gradient_scaling = 128.0
        # self.encoder.register_full_backward_hook(lambda module, grad_i, grad_o: (grad_i[0] / gradient_scaling, ))
    def forward(self, inputs):
        _inputs = (inputs- self.AABB[0:1,:]) / (self.AABB[1:2,:] - self.AABB[0:1,:])
        # _inputs = (inputs + self.max_len/2.) / self.max_len
        _inputs = torch.clamp(_inputs, min=0, max=1)
        return self.encoder(_inputs.contiguous()).to(torch.float32)
    def get_outdim(self):
        return self.encoder.n_output_dims


def get_hashgrid_embedder(mode, AABB):
    embed_kwargs = {
        'include_input': True,
        'input_dims': 3,
        'mode':mode,
        'AABB': AABB
    }

    embeder = HashGridEmbedder(**embed_kwargs)
    def embed(x, eo=embeder): return eo.forward(x)
    return embeder.encoder.parameters(), embed, embeder.get_outdim()


class FrequencyEmbedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
    
        if self.kwargs['mode']=='material':

            desired_resolution = 4096
            base_grid_resolution = 16
            num_levels = 16
            per_level_scale = np.exp(np.log(desired_resolution / base_grid_resolution) / (num_levels-1))
        elif self.kwargs['mode']=='sdf':
            desired_resolution = 2048*256   # assume our scene size is 256, the corresponding per level scale is 2.0
            base_grid_resolution = 16
            num_levels = 16
            per_level_scale = np.exp(np.log(desired_resolution / base_grid_resolution) / (num_levels-1))

        enc_cfg =  {
            "otype": "Frequency", 
            "n_frequencies": 6   
        }
        self.encoder = tcnn.Encoding(self.kwargs['input_dims'], enc_cfg)
        print("Encoder output: %d dims" % (self.encoder.n_output_dims))
        self.AABB = torch.from_numpy(self.kwargs['AABB']).cuda()
        self.max_len = torch.max(torch.abs(self.AABB))*2
        # gradient_scaling = 128.0
        # self.encoder.register_full_backward_hook(lambda module, grad_i, grad_o: (grad_i[0] / gradient_scaling, ))
    def forward(self, inputs):
        # _inputs = (inputs- self.AABB[0:1,:]) / (self.AABB[1:2,:] - self.AABB[0:1,:])
        # _inputs = (inputs + self.max_len/2.) / self.max_len
        # _inputs = torch.clamp(_inputs, min=0, max=1)
        return self.encoder(inputs.contiguous()).to(torch.float32)
    def get_outdim(self):
        return self.encoder.n_output_dims


def get_frequency_embedder(mode, AABB):
    embed_kwargs = {
        'include_input': True,
        'input_dims': 3,
        'mode':mode,
        'AABB': AABB
    }

    embeder = FrequencyEmbedder(**embed_kwargs)
    def embed(x, eo=embeder): return eo.forward(x)
    return embed, embeder.get_outdim()