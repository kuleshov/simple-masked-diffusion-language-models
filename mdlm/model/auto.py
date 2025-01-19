from typing import Optional
from dataclasses import dataclass
import torch
from transformers import (
    PretrainedConfig,
    PreTrainedModel,
    AutoModelForMaskedLM,
    AutoConfig,
)
from transformers.modeling_outputs import MaskedLMOutput
from mdlm.eval import compute_likelihoods
from mdlm.sampler import (
    SamplerConfig, 
    BaseDiffusionSampler,
    AncestralSampler,
    TauLeapingSampler
)

class MaskedDiffusionLMConfig(PretrainedConfig):
    """
    A unified config that stores:
      - masked_model_name_or_path: where to load the underlying MLM from
      - evaluation settings
      - any other arguments that should go to the base model
    """
    model_type = "masked_diffusion_lm"

    def __init__(
        self,
        masked_model_name_or_path: str = None,
        train_loss: str = "cross_entropy",
        eval_loss: str = "likelihood",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.masked_model_name_or_path = masked_model_name_or_path

        if train_loss not in ["cross_entropy", "likelihood"]:
            raise ValueError(f"Unknown train_loss: {train_loss}")
        if eval_loss not in ["cross_entropy", "likelihood"]:
            raise ValueError(f"Unknown eval_loss: {eval_loss}")

        self.train_loss = train_loss
        self.eval_loss = eval_loss

@dataclass
class MaskedDiffusionLMCOutput(MaskedLMOutput):
    """
    An MDLM output class that additionally contains:
      - likelihoods: the likelihood of each sequence
      - cross_entropies: the cross entropy loss for each sequence
      - masking_probabilities: the per-sequence masking probabilities
    """
    likelihood: torch.Tensor = None
    cross_entropy: torch.Tensor = None
    masking_probabilities: torch.Tensor = None

class AutoModelForMaskedDiffusionLM(PreTrainedModel):
    """
    A wrapper that:
      - Inherits from PreTrainedModel (for HF compatibility).
      - Uses the single MaskedDiffusionLMConfig for both the wrapper & references to the base model.
      - Loads the underlying MLM with AutoModelForMaskedLM.
      - Exposes a 'generate' method
    """

    config_class = MaskedDiffusionLMConfig 

    def __init__(self, config: MaskedDiffusionLMConfig):
        super().__init__(config)
        # Load the underlying masked LM if a base model name is specified
        if config.masked_model_name_or_path:
            self.masked_model = AutoModelForMaskedLM.from_pretrained(
                config.masked_model_name_or_path
            )
        else:
            self.masked_model = None

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """
        Standard Hugging Face API entry point.
        
        1) Loads MaskedDiffusionLMConfig from 'pretrained_model_name_or_path'.
        2) Instantiates this wrapper.
        3) Loads the underlying masked_model weights from config.masked_model_name_or_path.
        """
        # load the unified config (includes masked_model_name_or_path, diffusion params, etc.)
        config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path, 
            **kwargs
        )
        
        # if config is a diffusion model, load it and its base masked model
        if isinstance(config, MaskedDiffusionLMConfig):
            mdlm = cls(config)
        else:
            # otherwise, create an empty mdlm model and set its base masked model
            mdlm_config = MaskedDiffusionLMConfig(
                masked_model_name_or_path=pretrained_model_name_or_path
            )
            mdlm = cls(mdlm_config)
            # we assume base model is a masked langauge model like BERT
            mdlm.masked_model = AutoModelForMaskedLM.from_pretrained(
                pretrained_model_name_or_path,
                *model_args,
                **kwargs
            )
        
        return mdlm

    def save_pretrained(self, save_directory, **kwargs):
        """
        Saves:
          - self.config (MaskedDiffusionLMConfig) to 'config.json'
          - underlying masked_model (with its own config) to disk as well
        """
        # Save the wrapper config (includes masked_model_name_or_path, diffusion_param1, etc.)
        self.config.save_pretrained(save_directory)
        
        # Save the underlying base model
        self.masked_model.save_pretrained(save_directory, **kwargs)

    def forward(
            self, 
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            masking_probabilities: Optional[torch.Tensor] = None,
            *args, 
            **kwargs
        ) -> MaskedDiffusionLMCOutput:
        """
        Forward pass—delegates to the underlying masked_model.
        """
        # get output from base diffusion model
        masked_lm_output = self.masked_model(
            *args, 
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            **kwargs
        )

        # compute likelihoods and cross-entropy
        cross_entropy, likelihood = None, None
        if "loss" in masked_lm_output:
            cross_entropy = masked_lm_output["loss"]
        if "labels" in kwargs and masking_probabilities is not None:
            labels = kwargs["labels"]
            logits = masked_lm_output["logits"]
            likelihoods = compute_likelihoods(logits, labels, masking_probabilities)
            likelihood = likelihoods.mean()
        
        # create MDLM output
        masked_diff_lm_output = MaskedDiffusionLMCOutput(
            masking_probabilities=masking_probabilities,
            likelihood=likelihood,
            cross_entropy=cross_entropy,
            **masked_lm_output
        )

        # choose the lost that we want to report
        if self.training:
            if self.config.train_loss == "cross_entropy":
                loss = cross_entropy
            elif self.config.train_loss == "likelihood":
                loss = likelihood
        else:
            if self.config.eval_loss == "cross_entropy":
                loss = cross_entropy
            elif self.config.eval_loss == "likelihood":
                loss = likelihood 

        masked_diff_lm_output["loss"] = loss
        # print(masked_diff_lm_output.keys())
        return masked_diff_lm_output

    def generate(
        self,
        input_ids: torch.Tensor,
        sampler_config: Optional[SamplerConfig] = None,
        sampler_class: str = "ancestral",
        **kwargs
    ) -> torch.Tensor:
        """
        A unified generation entry point.
        
        Args:
          input_ids (torch.Tensor): starting sequence (batch_size, seq_len)
          sampler_config (SamplerConfig): contains parameters for the sampler
          sampler_class (str): which sampler to instantiate. E.g. "ancestral", "tau_leaping".
          **kwargs: forwarded to the sampler, if needed.
        
        Returns:
          torch.Tensor: final generated token IDs
        """
        if sampler_config is None:
            # fallback to default config
            sampler_config = SamplerConfig(sampler_name=sampler_class)
        else:
            # overwrite the sampler_name with what's passed
            sampler_config.sampler_name = sampler_class

        # 1) Create the sampler instance
        sampler = self._create_sampler(sampler_config)

        # 2) Delegate sampling
        generated_output = sampler.sample(self, input_ids, sampler_config, **kwargs)
        return generated_output

    def _create_sampler(self, sampler_config: SamplerConfig) -> BaseDiffusionSampler:
        """
        A helper method that instantiates the correct sampler 
        based on sampler_config.sampler_name.
        """
        sampler_name = sampler_config.sampler_name.lower()
        if sampler_name == "ancestral":
            return AncestralSampler()
        elif sampler_name == "tau_leaping":
            return TauLeapingSampler()
        else:
            raise ValueError(f"Unknown sampler: {sampler_name}")