import torch
from transformers import AutoTokenizer
from mdlm.model.auto import AutoModelForMaskedDiffusionLM
from mdlm.sampler import SamplerConfig

# masked_model = "google/bert_uncased_L-2_H-128_A-2"
masked_model = "bert-base-uncased" # bert-base (110M params)

tokenizer = AutoTokenizer.from_pretrained(masked_model)
model = AutoModelForMaskedDiffusionLM.from_pretrained(masked_model)
model.eval()

# prompt_text = "The capital of France is [MASK]."
prompt_text = "[MASK] is the capital of France."
# prompt_text = "[MASK]" * 14
inputs = tokenizer(prompt_text, return_tensors="pt")

# pick the device
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print('using device:', device)

model = model.to(device)
inputs = {k: v.to(device) for k, v in inputs.items()}
print(inputs)

my_config = SamplerConfig(
    num_steps = 10000, # number of sampling steps
    num_samples = 1,
    num_tokens = 16, # total number of tokens to generate
    mask_index = tokenizer.mask_token_id,
    use_cache = True, # cached mdlm sampler
    schedule = "loglinear" # noise schedule
)
with torch.no_grad():
    output_ids = model.generate(
        input_ids=inputs["input_ids"],
        sampler_name="ancestral",
        sampler_config=my_config
    )

print(output_ids[0])
decoded_text = tokenizer.decode(output_ids[0])
print("Generated:", decoded_text)