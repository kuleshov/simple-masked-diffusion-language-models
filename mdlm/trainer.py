from math import exp
from transformers import Trainer

class MDLMTrainer(Trainer):
    """
    A custom trainer that passes additional information to the evaluation step.
    Specifically, it passes mask probabilities to compute_metrics for cacluating PPL.
    """

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        # The main scalar for backprop:
        loss = outputs["loss"]
        
        # Log extra losses if they exist
        if False:
            log_dict = {}
            if "likelihood" in outputs:
                llik = outputs["likelihood"].item()
                log_dict.update({"likelihood": llik})
            if "cross_entropy" in outputs:
                log_dict.update({"cross_entropy": outputs["cross_entropy"].item()})
            self.log(log_dict)

        # HF expects us to return a scalar loss (plus outputs optionally)
        if return_outputs:
            return (loss, outputs)
        return loss