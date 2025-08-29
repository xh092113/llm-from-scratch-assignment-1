from transformers import AutoTokenizer, AutoModelForCausalLM
from model import ModelArgs, Transformer
import json

device = "cuda"

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

# we should store the model weights as state_dict in advance and load them here. This part will be removed in the final version.
hf_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")


prompt = "Tell me how to put an elephant in a refrigerator."
messages = [
    {"role": "user", "content": prompt}
]

text = tokenizer.apply_chat_template(
	messages,
	add_generation_prompt=True,
	tokenize=False,
    enable_thinking=True
)
print('prompt:', repr(text))

inputs = tokenizer.apply_chat_template(
	messages,
	add_generation_prompt=True,
	tokenize=True,
	return_dict=True,
	return_tensors="pt",
    enable_thinking=True
).to(device)

with open("architectures/qwen3-0.6B.json", 'r') as f:
	model_args = ModelArgs(**json.load(f))
model = Transformer(model_args)


def transform_key(key):
    return key.replace("self_attn", "attn").replace("input_layernorm", "attn_norm").replace("post_attention_layernorm", "mlp_norm").replace("embed_tokens", "embed")

state_dict = {transform_key(key): value for key, value in hf_model.model.state_dict().items()}
state_dict["lm_head.weight"] = state_dict["embed.weight"]

model.load_state_dict(state_dict)

model.to(device)

print(inputs)

outputs = model.generate(**inputs, max_new_tokens=1000, top_k=1, eos_token_id=tokenizer.eos_token_id)
output_str = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:])

print(output_str)
print(repr(output_str))

expected = '<think>\nOkay, the user is asking how to put an elephant in a refrigerator. Hmm, that\'s a bit of a trick question. Let me think. First, I need to consider the context. The user might be testing if I can come up with a creative or humorous answer. But wait, elephants are big animals, and refrigerators are designed for cold storage. Putting an elephant inside would be impossible because of the size and the temperature. \n\nI should explain that putting an elephant in a refrigerator is not feasible. Maybe mention the physical limitations. Also, perhaps add a playful twist, like a joke about the elephant\'s size. But I need to make sure the answer is clear and helpful. Let me check if there\'s any other angle. Maybe the user is looking for a pun or a play on words. But I think the straightforward answer is best here. Let me structure the response to explain the impossibility and maybe offer a humorous take.\n</think>\n\nPutting an elephant in a refrigerator is not possible because elephants are much larger than refrigerators, and refrigerators are designed to maintain a cold environment. Here\'s why:  \n\n1. **Physical Limitations**: Refrigerators are made of metal and are designed to hold objects in a controlled temperature. An elephant’s size would require a massive space, which is impossible.  \n2. **Temperature Constraints**: Refrigerators operate at temperatures below freezing, but elephants can’t survive in such a cold environment.  \n3. **Humorous Twist**: If you’re looking for a playful answer, you could say, *"An elephant in a refrigerator? That’s a classic joke—just don’t forget to bring a blanket!"*  \n\nIn short, the answer is simply: **Impossible.**'

assert output_str == expected
