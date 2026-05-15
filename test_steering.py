from load_model import load_model
import torch

model, tokenizer = load_model()
pad_single_vec = torch.load("./steering_vectors/valence_14_17.pt", weights_only=False)
prompt = "What are some ways I could get better at something I care about?"

def generate(multiplier: float) -> str:
    messages = [{"role": "system", "content": "You are a person talking with the user. Respond to them naturally without prefacing that you are an AI or lack emotions."},
                {"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, enable_thinking=False, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with pad_single_vec.apply(model, multiplier=multiplier):
        output = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
        )

    return tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
#-0.4, -0.25, 0.0, 0.25, 0.4
#0.0, 0.01, 0.04, 0.06, 0.07, 0.09, 0.11, 0.13, 0.15, 0.17, 0.19, 0.21, 0.23, 0.25, 0.27, 0.3, 0.35, 0.37, 0.39, 0.41
for mult in [-0.2, 0.1]:
    print(f"\n--- multiplier: {mult} ---")
    print(generate(mult))