import numpy as np
import torch
from transformers import Trainer

class MDLMTrainer(Trainer):
    """
    A custom trainer that passes additional information to the evaluation step.
    Specifically, it passes mask probabilities to compute_metrics for cacluating PPL.
    """

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """
        1) Do the standard forward pass with no grad.
        2) Extract logits + labels.
        3) Build an EvalInputs object that includes 'masking_probabilities'
        4) Return a 4-tuple (loss, preds, labels, inputs) => the 4th item is stored in EvalPrediction.inputs
        """
        # Pop the masking probabilities from the batch
        mask_probs = inputs.pop("masking_probabilities", None)

        with torch.no_grad():
            outputs = model(**inputs)
            loss = outputs.loss if hasattr(outputs, "loss") else None
            logits = outputs.logits if hasattr(outputs, "logits") else None

        if prediction_loss_only:
            return (loss, None, None)

        # get logits and labels
        # logits_np = logits.detach().cpu().numpy() if logits is not None else None
        # labels_np = inputs["labels"].detach().cpu().numpy() if "labels" in inputs else None
        logits = logits.detach() if logits is not None else None
        labels = inputs["labels"].detach() if "labels" in inputs else None
        
        # convert mask_probs to numpy
        # if mask_probs is not None:
        #     # It's typically a Python list => np.array
        #     mask_probs_np = np.array(mask_probs, dtype=np.float32)
        # else:
        #     mask_probs_np = None

        # Create our custom object
        eval_inputs = EvalInputs(
            masking_probabilities=mask_probs,
            raw_input_ids=None # no need for raw input id's right now
        )

        # Return a 4-tuple => Trainer constructs EvalPrediction(...) with `inputs=custom_eval_inputs`
        return loss, logits, labels, eval_inputs

class EvalInputs:
    """
    An object to hold custom inputs for evaluation.
      - masking_probabilities: np.ndarray of shape [batch_size]
      - raw_input_ids: optional, could store the original input_ids (or None)
    """
    def __init__(self, masking_probabilities=None, raw_input_ids=None):
        self.masking_probabilities = masking_probabilities
        self.raw_input_ids = raw_input_ids

    def __repr__(self):
        # A quick string representation
        return (f"EvalInputs(masking_probabilities={self.masking_probabilities}, "
                f"raw_input_ids={self.raw_input_ids})")
