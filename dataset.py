import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer
from datasets import load_dataset
import random

class MultiDataset(Dataset):
    def __init__(self, tokenizer, split='train', max_length=512, max_samples=20000):
        self.tokenizer = tokenizer
        self.max_length = max_length
        all_samples = []

        # WritingPrompts
        try:
            wp = load_dataset("euclaise/writingprompts", split=split)
            wp_list = []
            for i, item in enumerate(wp):
                if i >= max_samples // 3:
                    break
                if 'prompt' in item and 'story' in item:
                    text = f"{item['prompt'].strip()} {item['story'].strip()}"
                    word_count = len(text.split())
                    if 20 < word_count < 1000:
                        wp_list.append(text)
            all_samples += wp_list
        except:
            pass

        # ROC Stories
        try:
            roc_split = split if split in ['train', 'validation', 'test'] else 'validation'
            roc = load_dataset("Ximing/ROCStories", split=roc_split)
            roc_list = []
            for i, item in enumerate(roc):
                if i >= max_samples // 3:
                    break
                if 'prompt' in item and 'continuation' in item:
                    text = f"{item['prompt'].strip()} {item['continuation'].strip()}"
                    if len(text.split()) > 10:
                        roc_list.append(text)
            all_samples += roc_list
        except:
            pass

        # WikiText
        try:
            wiki = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)
            wiki_list = []
            for i, item in enumerate(wiki):
                if i >= max_samples // 3:
                    break
                text = item['text'].strip()
                word_count = len(text.split())
                if 50 < word_count < 500:
                    wiki_list.append(text)
            all_samples += wiki_list
        except:
            pass

        # OpenWebText if needed
        if len(all_samples) < max_samples * 0.8:
            try:
                owt = load_dataset("openwebtext", split='train', streaming=True)
                owt_list = []
                for i, item in enumerate(owt):
                    if i >= max_samples // 4 or len(owt_list) >= max_samples // 3:
                        break
                    text = item['text'].strip()
                    if 50 < len(text.split()) < 500 and any(w in text.lower() for w in ['said','was','were','then','when','after']):
                        owt_list.append(text)
                all_samples += owt_list
            except:
                pass

        if not all_samples:
            raise ValueError(f"No samples loaded for {split}")

       
        if len(all_samples) < max_samples:
            base_list = all_samples.copy()
            while len(all_samples) < max_samples and base_list:
                needed = max_samples - len(all_samples)
                sample_size = min(len(base_list), needed)
                all_samples += random.sample(base_list, sample_size)

        random.shuffle(all_samples)
        self.samples = all_samples[:max_samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text = self.samples[idx]
        enc = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        input_ids = enc['input_ids'].squeeze()
        attention_mask = enc['attention_mask'].squeeze()
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

def get_dataloaders(config):
    tokenizer = GPT2Tokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    train_ds = MultiDataset(tokenizer, 'train', config.max_length, config.train_size)
    val_ds = MultiDataset(tokenizer, 'validation', config.max_length, config.val_size)

    num_workers = 0 if config.device == "mps" else 2
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=False)

    return train_loader, val_loader, tokenizer
