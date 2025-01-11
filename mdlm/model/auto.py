import torch
from torch import nn
from transformers import (
    PretrainedConfig,
    PreTrainedModel,
    AutoModelForMaskedLM,
    AutoConfig,
)

class MaskedDiffusionLMConfig(PretrainedConfig):
    """
    A unified config that stores:
      - base_model_name_or_path: where to load the underlying MLM from
      - diffusion_param1, diffusion_param2: example hyperparameters for your diffusion logic
    """
    model_type = "masked_diffusion_lm"

    def __init__(
        self,
        base_model_name_or_path: str = None,
        diffusion_param1: float = 0.1,
        diffusion_param2: float = 0.2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.base_model_name_or_path = base_model_name_or_path
        self.diffusion_param1 = diffusion_param1
        self.diffusion_param2 = diffusion_param2

class AutoModelForMaskedDiffusionLM(PreTrainedModel):
    """
    A single-class wrapper that:
      - Inherits from PreTrainedModel (for HF compatibility).
      - Uses the single MaskedDiffusionLMConfig for both the wrapper & references to the base model.
      - Loads the underlying MLM with AutoModelForMaskedLM.
      - Exposes a 'generate' method for custom diffusion logic.
    """

    config_class = MaskedDiffusionLMConfig  # Ties this wrapper to the above config

    def __init__(self, config: MaskedDiffusionLMConfig):
        super().__init__(config)
        
        # Store the entire config as self.config (done by PreTrainedModel.__init__).
        # If you want direct access to diffusion params, you could also do:
        #   self.diffusion_param1 = config.diffusion_param1
        #   self.diffusion_param2 = config.diffusion_param2

        # Load the underlying masked LM if a base model name is specified
        if config.base_model_name_or_path:
            self.base_model = AutoModelForMaskedLM.from_pretrained(config.base_model_name_or_path)
        else:
            # Or build from a default config if no path is provided
            fallback_config = AutoConfig.from_pretrained("bert-base-uncased")
            self.base_model = AutoModelForMaskedLM.from_config(fallback_config)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """
        Standard Hugging Face API entry point.
        
        1) Loads MaskedDiffusionLMConfig from 'pretrained_model_name_or_path'.
        2) Instantiates this wrapper.
        3) Loads the underlying base_model weights from config.base_model_name_or_path.
        """
        # 1) Load the unified config (includes base_model_name_or_path, diffusion params, etc.)
        config = cls.config_class.from_pretrained(pretrained_model_name_or_path, **kwargs)
        
        # 2) Create the wrapper instance
        model = cls(config)
        
        # 3) Overwrite the base_model with actual weights from config.base_model_name_or_path
        #    This is optional: if we know the model has already been loaded above,
        #    we could skip this step. But typically you'll want to ensure weights
        #    match the checkpoint you loaded the config from.
        if config.base_model_name_or_path:
            model.base_model = AutoModelForMaskedLM.from_pretrained(
                config.base_model_name_or_path,
                *model_args,
                **kwargs
            )

        return model

    def save_pretrained(self, save_directory, **kwargs):
        """
        Saves:
          - self.config (MaskedDiffusionLMConfig) to 'config.json'
          - underlying base_model (with its own config) to disk as well
        """
        # Save the wrapper config (includes base_model_name_or_path, diffusion_param1, etc.)
        self.config.save_pretrained(save_directory)
        
        # Save the underlying base model
        self.base_model.save_pretrained(save_directory, **kwargs)

    def forward(self, *args, **kwargs):
        """
        Forward pass—delegates to the underlying base_model.
        """
        return self.base_model(*args, **kwargs)

    def generate(self, *args, **kwargs):
        """
        Placeholder for your diffusion-based generation method.
        You can access self.config.diffusion_param1, etc.,
        and use self.base_model(...) at each diffusion step.
        """
        raise NotImplementedError("Custom diffusion generation is not yet implemented.")
