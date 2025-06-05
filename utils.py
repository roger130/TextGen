import os
import json
import matplotlib.pyplot as plt

def save_metrics(metrics, filepath):
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=4)

def load_metrics(filepath):
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return None

def plot_training_curves(metrics, output_dir):
    epochs = range(1, len(metrics['train_loss']) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(epochs, metrics['train_loss'], 'b-', label='Train Loss')
    ax1.plot(epochs, metrics['val_loss'], 'r-', label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(epochs, metrics['val_perplexity'], 'g-', label='Val Perplexity')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Perplexity')
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()

    if 'learning_rate' in metrics:
        plt.figure(figsize=(8, 5))
        plt.plot(metrics['learning_rate'], 'b-')
        plt.xlabel('Steps')
        plt.ylabel('Learning Rate')
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'learning_rate.png'), dpi=300, bbox_inches='tight')
        plt.close()

def plot_live_loss(train_losses, val_losses=None, window=50):
    plt.ion()
    plt.clf()

    if len(train_losses) > window:
        smoothed = [sum(train_losses[i-window:i]) / window for i in range(window, len(train_losses))]
        plt.plot(range(window, len(train_losses)), smoothed, 'b-', label='Train Loss (smoothed)')
    else:
        plt.plot(train_losses, 'b-', label='Train Loss')

    if val_losses:
        plt.plot(val_losses, 'r-', label='Val Loss', marker='o')

    plt.xlabel('Steps/Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.pause(0.01)
    plt.ioff()
