# from sklearn.decomposition import IncrementalPCA
from gpu_pca import IncrementalPCAonGPU as IncrementalPCA # 2:32
import threading
import psutil
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils import (
    LOGGER,
)


class RealTimeMemoryMonitor:
    """Real-time memory monitor"""
    def __init__(self, update_interval=0.5):
        self.update_interval = update_interval
        self.monitoring = False
        self.monitor_thread = None
        self.gpu_mem = 0
        self.mem = 0
        self.pbar = None  # store progress bar reference
        
    def get_gpu_mem_mb(self):
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() // (1024 * 1024)
        return 0

    def get_mem_mb(self):
        return psutil.Process().memory_info().rss // (1024 * 1024)
    
    def set_progress_bar(self, pbar):
        self.pbar = pbar
    
    def _monitor_loop(self):
        while self.monitoring:
            self.gpu_mem = self.get_gpu_mem_mb()
            self.mem = self.get_mem_mb()
            
            # Real-time update progress bar description
            if self.pbar is not None:
                self.pbar.set_description(f"PCA computing - GPU Mem: {self.gpu_mem:.2f} MB, Mem: {self.mem:.2f} MB")
            
            time.sleep(self.update_interval)
    
    def start_monitoring(self):
        if not self.monitoring:
            self.monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
    
    def stop_monitoring(self):
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
    
    def get_status(self):
        return f"GPU Mem: {self.gpu_mem:.2f} MB, Mem: {self.mem:.2f} MB"


class VSPRegLoss:
    """ Variance-Scaled Projection Regularization loss(VSP-Reg)
    Regularization loss based on variance-scaled projection length of weight updates in the principal component subspace
    """
    def __init__(self, model_update, model_base, module_names, components, variances, means, alpha=0.9, beta=0.1):
        self.model_update = model_update
        self.model_base = model_base
        self.module_names = module_names
        self.components = components
        self.variances = variances
        self.means = means
        self.alpha = alpha
        self.beta = beta
        
        # Freeze PCA results
        for _component, _scale, _bias in zip(components.values(), variances.values(), means.values()):
            _component.requires_grad_(False)
            _scale.requires_grad_(False)
            _bias.requires_grad_(False)

        self.update_modules, self.base_modules = {}, {}
        for n, m in model_update.named_modules():
            if n in module_names:
                self.update_modules[n] = m
        for n, m in model_base.named_modules():
            if n in module_names:
                self.base_modules[n] = m
        
        self.update_weights, self.base_weights = {}, {}
        self._handles = []

    def register_hook(self):
        self.remove_handle_()
        for n in self.module_names:
            u_mod = self.update_modules[n]
            b_mod = self.base_modules[n]
            self._handles.append(u_mod.register_forward_hook(self._hook(self.update_weights, n)))
            self._handles.append(b_mod.register_forward_hook(self._hook(self.base_weights, n)))
 
    def _hook(self, dict_w, n):
        def fn(module, _, __):
            if isinstance(module, nn.Conv2d):
                if module.bias is not None:
                    # If bias exists, augment the weight matrix to represent convolution
                    # as a homogeneous linear transformation: [W; b] * [x^T; 1]^T = W*x + b
                    weight = module.weight.reshape(module.weight.shape[0], -1)
                    bias = module.bias.reshape(module.bias.shape[0], -1)
                    dict_w[n] = torch.cat([weight, bias], dim=1)
                else:
                    dict_w[n] = module.weight.reshape(module.weight.shape[0], -1)
            else:
                raise RuntimeError(f"Module {n}'s type {type(module)} is not supported")
        return fn

    def remove_handle_(self):
        """ When training is complete/no longer needed, remove all hooks, release memory, and prevent memory leaks. """
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def get_loss(self):
        loss = 0
        for n in self.module_names:
            proj = self.components[n] # [c_in*k*k, c_in*k*k]
            scale = torch.sqrt(self.variances[n]) # [c_in*k*k]
            bias = self.means[n] # [c_in*k*k]
            
            update_w = self.update_weights[n]
            base_w = self.base_weights[n]
            delta_w = F.normalize(update_w - base_w, p=2, dim=1)
            
            proj = proj.to(delta_w.device, delta_w.dtype)
            bias = bias.to(delta_w.device, delta_w.dtype)
            scale = scale.to(delta_w.device, delta_w.dtype)

            if proj.shape[1] < delta_w.shape[1]:
                # when the sampled features are less than feature dimension, we need to pad the components matrix and scales array
                proj = torch.cat([proj, torch.zeros((proj.shape[0], delta_w.shape[1] - proj.shape[1]), device=proj.device, dtype=proj.dtype)], dim=1)
                scale = torch.cat([scale, torch.zeros((delta_w.shape[1] - scale.shape[0]), device=scale.device, dtype=scale.dtype)], dim=0)
            
            loss += self.alpha * (delta_w @ proj.T @ torch.diag(scale)).norm(dim=1).mean() \
                + self.beta * (delta_w @ bias.unsqueeze(-1)).square().mean()
        return loss
    
    def set_parameters(self, components, variances, means, alpha=0.9, beta=0.1):
        self.components = components
        self.variances = variances
        self.means = means
        # Freeze PCA results
        for _component, _scale, _bias in zip(components.values(), variances.values(), means.values()):
            _component.requires_grad_(False)
            _scale.requires_grad_(False)
            _bias.requires_grad_(False)
        
        self.alpha = alpha
        self.beta = beta


class PCAHooker:
    def __init__(self, model, layers, device="cuda", check=False):
        self.model = model
        self.layers = layers
        self.modules = {}
        self.pca_operators = {}
        if not torch.cuda.is_available():
            device = "cpu"
            LOGGER.warning("CUDA is not available, using CPU")
        self.device = device
        self.check = check

        def _match(n, m, lid):
            # dfl layer is not a trainable layer
            return f"model.{lid}." in n and "dfl" not in n and isinstance(m, nn.Conv2d)

        self.feature_caches, self._handles = {}, []

        for lid in layers:
            for n, m in model.named_modules():
                if _match(n, m, lid):
                    k, c_in = m.kernel_size, m.in_channels
                    self.modules[n] = m
                    if m.bias is not None:
                        # If bias exists, augment the feature matrix to represent convolution
                        # as a homogeneous linear transformation: [W; b] * [x^T; 1]^T = W*x + b
                        self.pca_operators[n] = IncrementalPCA(n_components=c_in*k[0]*k[1]+1)
                    else:
                        self.pca_operators[n] = IncrementalPCA(n_components=c_in*k[0]*k[1])
                    self.feature_caches[n] = []

    def _get_sample_feature_indices(self, bs, h_out, w_out):
        # randomly sample feature indices
        sample_feature_indices = torch.randperm(bs*h_out*w_out, device=self.device)[:100]
        return sample_feature_indices

    @property
    def names(self):
        return list(self.modules.keys())
        
    def register_hook(self):
        self.remove_handle_()
        for n, mod in self.modules.items():
            self._handles.append(mod.register_forward_hook(self._hook(n)))

    def _hook(self, module_name):
        def fn(module, feat_in, feat_out):
            if isinstance(module, nn.Conv2d):
                if module.groups != 1:
                    raise RuntimeError(f"Group convolution is not supported")
                k, s, p = module.kernel_size, module.stride, module.padding

                feat_in = feat_in[0] # module may accept multiple input feats, and we only extract the first
                bs, nc_in, h, w = feat_in.shape
                
                if p[0] > 0 or p[1] > 0:
                    feat_in_padded = torch.nn.functional.pad(feat_in, (p[1], p[1], p[0], p[0]), mode='constant', value=0)
                else:
                    feat_in_padded = feat_in
                
                # Use the sliding window with the same settings as convolutin kernels
                # to unfold input features into a sequence of vectors
                # [bs, nc_in, h, w] --> [bs, nc_in, h_out, w, k[0]]
                feat_unfold_h = feat_in_padded.unfold(2, k[0], s[0])
                # [bs, nc_in, h_out, w, k[0]] --> [bs, nc_in, h_out, w_out, k[0], k[1]]
                feat_unfold = feat_unfold_h.unfold(3, k[1], s[1])
                # Permute the dims: [bs, nc_in, h_out, w_out, k[0], k[1]] -> [nc_in, k[0], k[1], bs, h_out, w_out]
                feat_unfold = feat_unfold.permute(1, 4, 5, 0, 2, 3).contiguous()
                # Squeeze: [nc_in, k[0], k[1], bs, h_out, w_out] --> [nc_in*k[0]*k[1], bs*h_out*w_out]
                h_out, w_out = feat_unfold.shape[4], feat_unfold.shape[5]
                feat_reshaped = feat_unfold.view(nc_in*k[0]*k[1], bs*h_out*w_out)
                if module.bias is not None:
                    # If bias exists, augment the feature matrix to represent convolution
                    # as a homogeneous linear transformation: [W; b] * [x^T; 1]^T = W*x + b
                    feat_reshaped = torch.concat([feat_reshaped, torch.ones((1, bs*h_out*w_out), device=feat_reshaped.device)], dim=0)

                # The following code is used to check whether the unfolding can represent the convolution operation
                # as a homogeneous linear transformation
                if self.check:
                    weight = module.weight.reshape(module.weight.shape[0], -1)
                    if module.bias is not None:
                        bias = module.bias.reshape(module.bias.shape[0], -1) # [c_out, 1]
                        weight = torch.concat([weight, bias], dim=1) # [c_out, c_in*k*k+1]
                    feat_out_reshaped = weight @ feat_reshaped # [c_out, bs*h_out*w_out]

                    # [c_out, bs*h_out*w_out] --> [c_out, bs, h_out, w_out] --> [bs, c_out, h_out, w_out]
                    feat_out_reshaped_reversed = feat_out_reshaped.view(-1, bs, h_out, w_out).permute(1, 0, 2, 3).contiguous()
                    LOGGER.info(f"Module {module_name}'s unfolding error: {torch.mean((feat_out - feat_out_reshaped_reversed).abs())}")

                sample_feature_indices = self._get_sample_feature_indices(bs, h_out, w_out)
                # Extract selected features for PCA computation
                feat_sampled = feat_reshaped[:, sample_feature_indices] # [c_in*k*k, len(sample_feature_indices)]
                
                feature_cache = self.feature_caches[module_name]
                feature_cache.append(feat_sampled)

                pca_operator = self.pca_operators[module_name]
                if sum([x.shape[1] for x in feature_cache]) >= pca_operator.n_components: # Incremental PCA requires the first batch's size is larger than n_components
                    feat_sampled = torch.cat(feature_cache, dim=1)
                    feature_cache.clear()
                    # pca_operator.partial_fit(feat_sampled.cpu().T.numpy())
                    pca_operator.partial_fit(feat_sampled.T)
            else:
                raise RuntimeError(f"Module type {type(module)} is not supported")

        return fn

    def remove_handle_(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def clear_feature_cache(self):
        for n, cache in self.feature_caches.items():
            if len(cache) > 0:
                # before clearing the cache, we need to fit the PCA operator with the final batch of features
                if not hasattr(self.pca_operators[n], 'components_'):
                    # If components_ is not calculated, it means the PCA operator has not been called yet, and that
                    # means the number of features in this cache has never reached n_components, so we need to
                    # reduce n_components to the number of features in this cache.
                    LOGGER.warning(f"Too few samples to fit PCA in module {n}. Could result in instability.")
                    self.pca_operators[n].n_components = torch.cat(cache, dim=1).shape[1]
                # self.pca_operators[n].partial_fit(torch.cat(cache, dim=1).cpu().T.numpy())
                self.pca_operators[n].partial_fit(torch.cat(cache, dim=1).T)
            cache.clear()
    
    def get_pca_results(self, name):
        return (self.pca_operators[name].components_, 
                self.pca_operators[name].explained_variance_, 
                self.pca_operators[name].mean_)

    def get_pca_operator(self, name):
        return self.pca_operators[name]
    
    def set_pca_operator(self, name, pca_operator):
        self.pca_operators[name] = pca_operator
