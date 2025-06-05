import os
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from peft import PeftModel, PeftConfig

def load_model(model_path):
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    cfg = PeftConfig.from_pretrained(model_path)
    base = GPT2LMHeadModel.from_pretrained(cfg.base_model_name_or_path)
    model = PeftModel.from_pretrained(base, model_path)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device).eval()
    return tokenizer, model, device

def generate_text(tokenizer, model, device, prompt, max_length=100, temperature=0.8, top_p=0.9):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    path = "./outputs/best_model"

    tokenizer, model, device = load_model(path)

    examples = [
        "Once upon a time in a magical forest,",
        "The scientist discovered that",
        "In the year 2050, humans will"
    ]

    for prompt in examples:
        text = generate_text(tokenizer, model, device, prompt)
       
    while True:
        prompt = input(">> ")
        if prompt.strip().lower() == "quit":
            break
        text = generate_text(tokenizer, model, device, prompt)
        print(f"Generated: {text}\n")

if __name__ == "__main__":
    main()
