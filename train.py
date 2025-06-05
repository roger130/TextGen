import os
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
from config import Config
from model import get_lora_model
from dataset import get_dataloaders
from evaluate import evaluate
from utils import save_metrics, plot_training_curves

def train_epoch(model, loader, optimizer, scheduler, config, epoch, metrics):
    model.train()
    total_loss = 0
    batch_losses = []
    bar = tqdm(loader, desc=f"Epoch {epoch}")
    for step, batch in enumerate(bar, 1):
        input_ids = batch['input_ids'].to(config.device)
        attention_mask = batch['attention_mask'].to(config.device)
        labels = batch['labels'].to(config.device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss / config.gradient_accumulation_steps
        total_loss += loss.item()
        batch_losses.append(loss.item() * config.gradient_accumulation_steps)
        loss.backward()

        if step % config.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            metrics['learning_rate'].append(scheduler.get_last_lr()[0])

        bar.set_postfix({'loss': loss.item() * config.gradient_accumulation_steps})

    avg_loss = total_loss / len(loader)
    metrics['batch_losses'].extend(batch_losses)
    return avg_loss

def train(config):
    os.makedirs(config.output_dir, exist_ok=True)
    metrics = {
        'train_loss': [], 'val_loss': [], 'val_perplexity': [],
        'batch_losses': [], 'learning_rate': []
    }

    train_loader, val_loader, tokenizer = get_dataloaders(config)
    model = get_lora_model(config).to(config.device)

    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    total_steps = len(train_loader) * config.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=total_steps
    )

    best_val = float('inf')
    patience = 0

    for epoch in range(1, config.num_epochs + 1):
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler,
            config, epoch, metrics
        )
        metrics['train_loss'].append(train_loss)

        val_loss, val_ppl = evaluate(model, val_loader, config)
        metrics['val_loss'].append(val_loss)
        metrics['val_perplexity'].append(val_ppl)

        save_metrics(metrics, os.path.join(config.output_dir, 'metrics.json'))
        plot_training_curves(metrics, config.output_dir)

        if val_loss < best_val - config.min_delta:
            best_val = val_loss
            patience = 0
            model.save_pretrained(os.path.join(config.output_dir, "best_model"))
            tokenizer.save_pretrained(os.path.join(config.output_dir, "best_model"))
        else:
            patience += 1
            if patience >= config.patience:
                break

    model.save_pretrained(os.path.join(config.output_dir, "final_model"))
    tokenizer.save_pretrained(os.path.join(config.output_dir, "final_model"))
    save_metrics(metrics, os.path.join(config.output_dir, 'metrics_final.json'))
    plot_training_curves(metrics, config.output_dir)

if __name__ == "__main__":
    cfg = Config()
    train(cfg)
