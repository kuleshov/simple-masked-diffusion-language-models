from dataclasses import dataclass
from typing import Optional
from abc import ABC, abstractmethod
import torch
from mdlm.model.schedule import LogLinearSchedule

@dataclass
class SamplerConfig:
    """
    Configuration for sampler/diffusion parameters.
    Holds parameters for ancestral sampling, tau-leaping, etc.
    """
    # Common parameters
    num_steps: int = 1000
    num_samples: int = 8
    num_tokens: int = 128 # total number of tokens to generate
    temperature: float = 1.0 # currently not used
    eps: float = 1e-5 # numerical precision

    # For ancestral sampling
    mask_index: int = -1 # get it from model or throw error
    use_cache: bool = True # cached mdlm sampler
    schedule: str = "loglinear" # noise schedule

    # For tau-leaping sampling
    tau: float = 0.1

    # Potentially more fields...
    # e.g., top_k, top_p, random_seed, etc.

class BaseDiffusionSampler(ABC):
    @abstractmethod
    def sample(
        self, 
        model, 
        config: SamplerConfig, 
        input_ids: Optional[torch.Tensor], 
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
    def sample(
            self, 
            model, 
            config: SamplerConfig, 
            input_ids: Optional[torch.Tensor] = None, 
            **kwargs
        ):
        """
        Ancestral sampler that uses:
            - model: the discrete diffusion denoising model
            - config: sampling configuration
            - input_ids: initial prompt
        """
        # create first sample (config.num_samples x config.num_tokens)
        zt = self._sample_prior(config, input_ids).to(model.device)
        print(zt)

        # set up sampling steps
        timesteps = torch.linspace(
            1, config.eps, config.num_steps + 1, device=model.device
        )
        dt = (1 - config.eps) / config.num_steps
        
        # initialize schedule and caching
        schedule = self._init_schedule(config)
        p_x0_cache = None

        # sample all the time steps
        for i in range(config.num_steps):
            t = timesteps[i] * torch.ones(zt.shape[0], device=model.device)
            zt, p_x0_cache = self._sample_at_t(
                model, schedule, config, zt, t, dt, p_x0_cache
            )

        zt = model.forward(zt).logits.argmax(dim=-1)
        return zt
    
    def _sample_at_t(self, model, schedule, config, zt, t, dt, p_x0=None):
        # mask_chance_t is 1-\alpha_t in MDLM paper
        # i.e. the cumulative probability at time t that a token is masked
        mask_chance_t = schedule.cumulative(t)[:, None, None]
        mask_chance_s = schedule.cumulative(t - dt)[:, None, None]

        if not config.use_cache or p_x0 is None:
            p_x0 = model.forward(input_ids=zt).logits.exp()

        # this samples from q(zs|zt,x_\theta(zt)), i.e. (6) in the MDLM paper
        q_zs = p_x0 * (mask_chance_t - mask_chance_s) # up to a constant
        q_zs[:, :, config.mask_index] = mask_chance_s[:, :, 0]
        _zs = _sample_categorical(q_zs)

        copy_flag = (zt != config.mask_index).to(zt.dtype)
        zs = copy_flag * zt + (1 - copy_flag) * _zs

        if not torch.allclose(zs, zt):
            # do not return cached values
            p_x0 = None

        return zs, p_x0
    
    def _sample_prior(self, config, input_ids=None):
        zT = config.mask_index * torch.ones(
            config.num_samples, 
            config.num_tokens, 
            dtype=torch.int64
        )
        if input_ids is not None:
            zT[:, :input_ids.shape[1]] = input_ids
        return zT
    
    def _init_schedule(self, config):
        if config.schedule == "loglinear":
            return LogLinearSchedule(eps=config.eps)
        else:
            raise ValueError(f"Unknown schedule: {config.schedule}")

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
    gumbel_norm = (1e-10- (torch.rand_like(categorical_probs) + 1e-10).log())
    return (categorical_probs / gumbel_norm).argmax(dim=-1)