from dataclasses import dataclass
from typing import Optional
from abc import ABC, abstractmethod
import torch

@dataclass
class SamplerConfig:
    """
    Configuration for your sampler/diffusion parameters.
    This can hold parameters for ancestral sampling, tau-leaping, etc.
    """
    # Common parameters
    num_steps: int = 1000
    num_samples: int = 8
    # temperature: float = 1.0

    # For ancestral sampling
    mask_index: int = 0
    use_cache: bool = True

    # For tau-leaping sampling
    tau: float = 0.1

    # Potentially more fields...
    # e.g., top_k, top_p, random_seed, etc.

class BaseDiffusionSampler(ABC):
    @abstractmethod
    def sample(
        self, 
        model, 
        input_ids: torch.Tensor, 
        sampler_config: SamplerConfig, 
        **kwargs
    ) -> torch.Tensor:
        """
        Run a diffusion or sampling procedure, returning final token IDs.
        
        Args:
          model (nn.Module): Your AutoModelForMaskedDiffusionLM.
          input_ids (torch.Tensor): The initial sequence(s).
          sampler_config (SamplerConfig): Holds parameters relevant to this sampler.
          **kwargs: Optional extra arguments.

        Returns:
          torch.Tensor: final generated token IDs.
        """
        pass

class AncestralSampler(BaseDiffusionSampler):
    def sample(self, model, input_ids, sampler_config: SamplerConfig, **kwargs):
        """
        Example 'ancestral' approach that uses:
          - sampler_config.num_samples
          - sampler_config.num_steps
        """
        device = input_ids.device
        output_ids = input_ids.clone()
        for step in range(sampler_config.num_steps):
            # Pseudo-code for re-masking, forward pass, sampling, etc.
            pass
        return output_ids
    
    def _sample_prior(self, *batch_dims):
        return self.mask_index * torch.ones(
            *batch_dims, dtype=torch.int64
        )
    
    def _sample(self, num_samples, num_steps, use_cache=True, eps=1e-5):
        """Generate samples from the model."""
        x = self._sample_prior(
            num_samples,
            self.config.model.length
        ).to(self.device)
        timesteps = torch.linspace(
            1, eps, num_steps + 1, device=self.device
        )
        dt = (1 - eps) / num_steps
        p_x0_cache = None

        for i in range(num_steps):
            t = timesteps[i] * torch.ones(
                x.shape[0], 1, device=self.device)
            if not use_cache:
                x = self._ddpm_update(x, t, dt)
            else:
                p_x0_cache, x_next = self._ddpm_caching_update(
                    x, t, dt, p_x0=p_x0_cache
                )
                if not torch.allclose(x_next, x):
                    # Disable caching
                    p_x0_cache = None
                    x = x_next

        # if self.config.sampling.noise_removal:
        #     t = timesteps[-1] * torch.ones(x.shape[0], 1,
        #                                     device=self.device)
        x = self.forward(x).argmax(dim=-1)
        return x

    def _ddpm_caching_update(model, x, t, dt, p_x0=None):
        # assert model.config.noise.type == 'loglinear'
        # sigma_t, _ = model.noise(t)
        if t.ndim > 1:
            t = t.squeeze(-1)
        assert t.ndim == 1
        move_chance_t = t[:, None, None]
        move_chance_s = (t - dt)[:, None, None]
        assert move_chance_t.ndim == 3, move_chance_t.shape
        if p_x0 is None:
            p_x0 = model.forward(x).exp()
        
        assert move_chance_t.ndim == p_x0.ndim
        q_xs = p_x0 * (move_chance_t - move_chance_s)
        q_xs[:, :, model.mask_index] = move_chance_s[:, :, 0]
        _x = _sample_categorical(q_xs)
        
        copy_flag = (x != model.mask_index).to(x.dtype)
        return p_x0, copy_flag * x + (1 - copy_flag) * _x

    def _ddpm_update(model, x, t, dt):
        sigma_t, _ = model.noise(t)
        sigma_s, _ = model.noise(t - dt)
        if sigma_t.ndim > 1:
            sigma_t = sigma_t.squeeze(-1)
        if sigma_s.ndim > 1:
            sigma_s = sigma_s.squeeze(-1)
        assert sigma_t.ndim == 1, sigma_t.shape
        assert sigma_s.ndim == 1, sigma_s.shape
        move_chance_t = 1 - torch.exp(-sigma_t)
        move_chance_s = 1 - torch.exp(-sigma_s)
        move_chance_t = move_chance_t[:, None, None]
        move_chance_s = move_chance_s[:, None, None]
        unet_conditioning = sigma_t
        log_p_x0 = model.forward(x, unet_conditioning)
        assert move_chance_t.ndim == log_p_x0.ndim
        # Technically, this isn't q_xs since there's a division
        # term that is missing. This division term doesn't affect
        # the samples.
        q_xs = log_p_x0.exp() * (move_chance_t
                                - move_chance_s)
        q_xs[:, :, model.mask_index] = move_chance_s[:, :, 0]
        _x = _sample_categorical(q_xs)

        copy_flag = (x != model.mask_index).to(x.dtype)
        return copy_flag * x + (1 - copy_flag) * _x

class TauLeapingSampler(BaseDiffusionSampler):
    def sample(self, model, input_ids, sampler_config: SamplerConfig, **kwargs):
        """
        Example 'tau-leaping' approach that uses:
          - sampler_config.num_steps
          - sampler_config.tau
        """
        device = input_ids.device
        output_ids = input_ids.clone()
        for step in range(sampler_config.num_steps):
            # Some diffusion logic using tau
            pass
        return output_ids

def _sample_categorical(categorical_probs):
  gumbel_norm = (
    1e-10
    - (torch.rand_like(categorical_probs) + 1e-10).log())
  return (categorical_probs / gumbel_norm).argmax(dim=-1)