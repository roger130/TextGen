import torch
from transformers import GPT2LMHeadModel
from peft import LoraConfig, get_peft_model, TaskType

def get_lora_model(config):
    model = GPT2LMHeadModel.from_pretrained(config.model_name, cache_dir=config.cache_dir)
    for param in model.parameters():
        param.requires_grad = False

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable:,} ({100 * trainable / total:.2f}%)")
    print(f"Total parameters: {total:,}")

    return model
