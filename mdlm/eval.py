import numpy as np
import torch.nn.functional as F
from transformers import EvalPrediction

def compute_metrics(eval_pred: EvalPrediction):
    """
    Computes MDLM likelihood and perplexity during training.
    We expect eval_pred.inputs to be an EvalInputs object.
    """
    logits = eval_pred.predictions 
    label_ids = eval_pred.label_ids
    eval_inputs = eval_pred.inputs  # type: EvalInputs
    mask_probs = eval_inputs.masking_probabilities  # shape [batch_size] or None

    if logits is None or label_ids is None or mask_probs is None:
        return {"likelihood": -1.0}  # Just a placeholder if something is missing

    batch_size, seq_len, vocab_size = logits.shape

    # We'll do a quick "sum of cross-entropy" measure
    flat_logits = logits.view(-1, vocab_size)   # [batch_size*seq_len, vocab_size]
    flat_labels = label_ids.view(-1)            # [batch_size*seq_len]

    ce_per_token = F.cross_entropy(
        flat_logits, flat_labels, reduction="none", ignore_index=-100
    )  # shape: [batch_size*seq_len]

    ce_per_token = ce_per_token.view(batch_size, seq_len)
    ce_per_seq = ce_per_token.sum(dim=1)  # shape [batch_size]

    # we define "likelihood_i" = CE_i / (1 - p_i) 
    p = mask_probs
    denom = np.clip(1.0 - p, 1e-9, 9999)
    likelihoods = ce_per_seq / denom
    mean_likelihood = likelihoods.mean().cpu().numpy()

    metrics = {
        "likelihood": float(mean_likelihood),
        "perplexity": float(np.exp(mean_likelihood))
    }

    return metrics
