import torch
from tqdm import tqdm
import numpy as np

def evaluate(model, dataloader, config):
    model.eval()
    total_loss = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(config.device)
            attention_mask = batch['attention_mask'].to(config.device)
            labels = batch['labels'].to(config.device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item() * input_ids.size(0)
            total_samples += input_ids.size(0)
    
    avg_loss = total_loss / total_samples
    perplexity = np.exp(avg_loss)
    
    return avg_loss, perplexity