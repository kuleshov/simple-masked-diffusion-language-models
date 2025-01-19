import torch
from transformers import DataCollatorForLanguageModeling

def chunk_dataset(examples, chunk_size):
    """
    Convert entire sequences into one big list of tokens,
    then chunk into segments of length `chunk_size`.
    """
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples["input_ids"])
    # Truncate so total_length is multiple of chunk_size
    total_length = (total_length // chunk_size) * chunk_size

    result = {
        k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

class RandomMaskingCollator(DataCollatorForLanguageModeling):
    """
    A data collator that applies random MLM masks to each sequence.

    Masking logic:
      - For each sequence in a batch, we sample a random probability p_i ∈ [min_mlm_prob, max_mlm_prob].
      - That row gets masked with probability p_i:
          * 80% of the masked tokens => [MASK]
          * 20% => remain their original token (no random replacements)

    We also store the per-sequence probabilities in two places:
      (1) batch["masking_probabilities"] as a list
      (2) batch["mask_probabilities_tensor"] as a float tensor (so the model can read them)
    """

    def __init__(
        self,
        tokenizer,
        min_mlm_prob=0.05,
        max_mlm_prob=0.25,
        mlm=True,
        pad_to_multiple_of=None
    ):
        # We initialize with mlm_probability=0.0 since we'll override it per sequence.
        super().__init__(
            tokenizer=tokenizer,
            mlm=mlm,
            mlm_probability=0.0,
            pad_to_multiple_of=pad_to_multiple_of
        )
        self.min_mlm_prob = min_mlm_prob
        self.max_mlm_prob = max_mlm_prob

        self._current_batch_probs = []

    def torch_mask_tokens(self, inputs: torch.Tensor, special_tokens_mask: torch.Tensor = None):
        """
        Vectorized override of the MLM masking step:
          1) Sample a random MLM probability per sequence.
          2) Broadcast that into [batch_size, seq_len].
          3) Use it to Bernoulli-sample masked indices.
          4) 80% => [MASK], 20% => keep original (no random token replacements).
        """
        if not self.mlm:
            return inputs, inputs  # If MLM is disabled, do nothing special

        labels = inputs.clone()
        batch_size, seq_len = labels.shape

        # 1) Sample a random probability per sequence, shape = [batch_size]
        prob_vector = torch.rand(batch_size, device=labels.device) * (
            self.max_mlm_prob - self.min_mlm_prob
        ) + self.min_mlm_prob

        # 2) Broadcast the probability to build a [batch_size, seq_len] matrix
        probability_matrix = prob_vector.unsqueeze(1).expand(batch_size, seq_len)

        # Handle special tokens: never mask them
        if special_tokens_mask is None:
            # If none given, compute from tokenizer
            special_tokens_mask_list = [
                self.tokenizer.get_special_tokens_mask(vals, already_has_special_tokens=True)
                for vals in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask_list, 
                                               dtype=torch.bool, 
                                               device=labels.device)
        else:
            special_tokens_mask = special_tokens_mask.bool()
        
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

        # 3) Sample which tokens to mask
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # Only compute loss on masked tokens

        # 4) p% => [MASK], (1-p)% => keep original
        indices_replaced = (
            torch.bernoulli(torch.full(labels.shape, 0.8, device=labels.device))
            .bool() & masked_indices
        )
        inputs[indices_replaced] = self.tokenizer.mask_token_id
        # The remaining masked tokens (masked_indices & ~indices_replaced) keep their original IDs.
        
        # Store per-sequence probabilities for later use
        self._current_batch_probs = prob_vector

        return inputs, labels

    def __call__(self, examples):
        """
        1) Use parent logic to create/pad the batch and invoke torch_mask_tokens.
        2) Insert the per-sequence probabilities into the batch dict, both
           as a Python list and as a float tensor for direct model access.
        """
        batch = super().__call__(examples)
        batch["masking_probabilities"] = self._current_batch_probs

        return batch
