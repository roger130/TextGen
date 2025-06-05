import os
import json
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from peft import PeftModel, PeftConfig
from captum.attr import IntegratedGradients

class GPT2Interpretability:
    def __init__(self, model_path="./outputs/best_model"):
        # Load tokenizer and set pad_token to eos_token
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load base model and PEFT fine-tuned weights
        cfg = PeftConfig.from_pretrained(model_path)
        base = GPT2LMHeadModel.from_pretrained(
            cfg.base_model_name_or_path,
            attn_implementation="eager"
        )
        self.model = PeftModel.from_pretrained(base, model_path)

        # Set device (MPS if available, otherwise CPU)
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model.to(self.device).eval()

        # Output directory
        self.out = "./results"
        os.makedirs(self.out, exist_ok=True)

    def extract_attention_weights(self, text):
        """Extract attention weights for every layer and head."""
        ids = self.tokenizer(text, return_tensors="pt").to(self.device)["input_ids"]
        with torch.no_grad():
            attn = self.model(ids, output_attentions=True).attentions
        # Convert to NumPy and remove batch dimension
        data = [layer.squeeze(0).cpu().numpy() for layer in attn]
        tokens = self.tokenizer.convert_ids_to_tokens(ids[0])
        return {"tokens": tokens, "attn": data}

    def visualize_attention(self, attn_data, layer_idx):
        """Plot attention heatmaps in a 3×2 grid for up to 6 heads of a specified layer."""
        layer = attn_data["attn"][layer_idx]
        tokens = attn_data["tokens"]
        heads = layer.shape[0]
        n = min(heads, 6)

        # Arrange up to 6 heads in 3 rows × 2 columns
        rows, cols = 3, 2
        fig, axes = plt.subplots(rows, cols, figsize=(12, 12))
        axes = axes.flatten()

        for i in range(n):
            sns.heatmap(
                layer[i],
                xticklabels=tokens,
                yticklabels=tokens,
                cmap="Blues",
                cbar=False,
                ax=axes[i]
            )
            axes[i].set_title(f"Layer {layer_idx} - Head {i}")
            axes[i].tick_params(axis="x", rotation=45)

        # Hide any unused subplots
        for j in range(n, rows * cols):
            axes[j].axis("off")

        plt.tight_layout()
        path = os.path.join(self.out, f"attention_layer{layer_idx}.png")
        plt.savefig(path, dpi=200, bbox_inches="tight")
        plt.close()

    def analyze_token_importance_gradients(self, text, target_pos=-1):
        """Compute token importance using Integrated Gradients."""
        ids = self.tokenizer(text, return_tensors="pt").to(self.device)["input_ids"]
        tokens = self.tokenizer.convert_ids_to_tokens(ids[0])
        embedding_layer = self.model.get_input_embeddings()
        baseline = torch.full_like(ids, self.tokenizer.pad_token_id)

        def forward_func(x):
            # x: [1, seq_len, embed_dim]
            outputs = self.model(inputs_embeds=x)
            logits = outputs.logits  # [1, seq_len, vocab_size]
            pos = logits.shape[1] - 1 if target_pos == -1 else target_pos
            max_logit = torch.max(logits[0, pos, :])  # scalar
            return max_logit.unsqueeze(0)  # shape [1]

        input_emb = embedding_layer(ids)      # [1, seq_len, embed_dim]
        baseline_emb = embedding_layer(baseline)
        ig = IntegratedGradients(forward_func)
        attributions = ig.attribute(input_emb, baseline_emb, n_steps=50, return_convergence_delta=False)
        scores = attributions.sum(dim=-1).squeeze(0).cpu().numpy()  # [seq_len]
        scores = np.abs(scores)
        if scores.max() > 0:
            scores = scores / scores.max()
        return {"tokens": tokens, "scores": scores}

    def visualize_token_importance(self, data):
        """Plot a bar chart showing each token's Integrated Gradients score."""
        tokens, scores = data["tokens"], data["scores"]
        x = np.arange(len(tokens))
        colors = plt.cm.Reds(scores)
        plt.figure(figsize=(8, 3))
        bars = plt.bar(x, scores, color=colors)
        plt.xticks(x, tokens, rotation=45)
        plt.ylim(0, 1)
        for bar, score in zip(bars, scores):
            plt.text(bar.get_x() + 0.1, score + 0.02, f"{score:.2f}", fontsize=6)
        plt.tight_layout()
        path = os.path.join(self.out, "gradients.png")
        plt.savefig(path, dpi=200, bbox_inches="tight")
        plt.close()

    def perturbation_analysis(self, text, samples=50):
        """Estimate token importance by randomly replacing tokens and computing KL divergence."""
        ids = self.tokenizer(text, return_tensors="pt").to(self.device)["input_ids"]
        tokens = self.tokenizer.convert_ids_to_tokens(ids[0])
        with torch.no_grad():
            original_probs = F.softmax(self.model(ids).logits, dim=-1)
        seq_len = len(tokens)
        importance = np.zeros(seq_len)
        for i in range(seq_len):
            kl_values = []
            for _ in range(samples):
                perturbed_ids = ids.clone()
                rand_token = torch.randint(0, len(self.tokenizer), (1,)).item()
                perturbed_ids[0, i] = rand_token
                with torch.no_grad():
                    perturbed_probs = F.softmax(self.model(perturbed_ids).logits, dim=-1)
                kl = F.kl_div(perturbed_probs.log(), original_probs, reduction="batchmean").item()
                kl_values.append(kl)
            importance[i] = np.mean(kl_values)
        if importance.max() > 0:
            importance = importance / importance.max()
        return {"tokens": tokens, "scores": importance}

    def visualize_perturbation(self, data):
        """Plot a single-row heatmap showing each token's perturbation score."""
        tokens, scores = data["tokens"], data["scores"]
        arr = scores.reshape(1, -1)
        plt.figure(figsize=(8, 1))
        plt.imshow(arr, cmap="Reds", aspect="auto")
        plt.xticks(np.arange(len(tokens)), tokens, rotation=45)
        plt.yticks([])
        for i, score in enumerate(scores):
            plt.text(i, 0, f"{score:.2f}", ha="center", va="center",
                     color="white" if score > 0.5 else "black", fontsize=6)
        plt.tight_layout()
        path = os.path.join(self.out, "perturbation.png")
        plt.savefig(path, dpi=200, bbox_inches="tight")
        plt.close()

    def compare_methods(self, text):
        """Run attention, gradient-based, and perturbation analyses, then save all results."""
        # 1. Attention extraction and visualization
        attn_data = self.extract_attention_weights(text)
        total_layers = len(attn_data["attn"])
        layers_to_plot = [0, total_layers // 2, total_layers - 1]
        for idx in layers_to_plot:
            self.visualize_attention(attn_data, idx)

        # 2. Integrated Gradients analysis
        grad_data = self.analyze_token_importance_gradients(text)
        self.visualize_token_importance(grad_data)

        # 3. Perturbation analysis
        pert_data = self.perturbation_analysis(text)
        self.visualize_perturbation(pert_data)

        # 4. Save JSON results
        results = {
            "text": text,
            "tokens": grad_data["tokens"],
            "gradient_scores": grad_data["scores"].tolist(),
            "perturbation_scores": pert_data["scores"].tolist()
        }
        with open(os.path.join(self.out, "results.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

def run_full_analysis():
    model = GPT2Interpretability()
    sentences = [
        "The scientist discovered that the new compound could",
        "Once upon a time in a magical forest",
        "The future of artificial intelligence will",
        "She walked into the room and immediately noticed"
    ]
    for sent in sentences:
        model.compare_methods(sent)
    print(f"All analyses complete. Check the '{model.out}' directory for outputs.")

if __name__ == "__main__":
    run_full_analysis()

