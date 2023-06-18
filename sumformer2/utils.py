import math
import os
import torch
import wandb

from datasets import load_dataset, concatenate_datasets
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, SequentialSampler, BatchSampler


def create_data_loader(dataset, batch_size, collate_fn):
    sampler = SequentialSampler(dataset)
    batch_sampler = BatchSampler(sampler, batch_size, drop_last=False)
    return DataLoader(dataset, batch_sampler=batch_sampler, collate_fn=collate_fn)


def load_xsum():
    """Load the XSum dataset and split into train, validation, and test sets. Keep only the docs and their summary."""
    train_dataset = load_dataset("xsum", split="train")
    val_dataset = load_dataset("xsum", split="validation")
    test_dataset = load_dataset("xsum", split="test")

    # Adding a the doc length of each instance
    train_dataset = train_dataset.map(lambda x: {'doc_len': len(x['document'])})
    val_dataset = val_dataset.map(lambda x: {'doc_len': len(x['document'])})
    test_dataset = test_dataset.map(lambda x: {'doc_len': len(x['document'])})

    # Sort the datasets by document length
    train_dataset = train_dataset.sort('doc_len')
    val_dataset = val_dataset.sort('doc_len')
    test_dataset = test_dataset.sort('doc_len')

    return train_dataset, val_dataset, test_dataset


def load_reddit(train_split, val_split, min_len=50):
    """Concatenate the short and long reddit TIFU datasets and split into train, validation, and test sets. Keep only the docs and their summary."""
    dataset_short = load_dataset("reddit_tifu", "short")
    dataset_long = load_dataset("reddit_tifu", "long")
    dataset = concatenate_datasets([dataset_short["train"], dataset_long["train"]])
    dataset = dataset.rename_columns({'documents': 'document', 'tldr': 'summary'})
    dataset = dataset.map(lambda x: {'document': x['title'] + " " + x['document'], 'summary': x['summary']})  # Concatenate the contents of the title column to the front of the documents column for every row
    dataset = dataset.remove_columns(['ups', 'num_comments', 'upvote_ratio', 'score', 'title'])

    # Filtering out too short documents and summaries
    dataset = dataset.filter(lambda x:
                             len(x["document"]) > min_len
                             and len(x["summary"]) > min_len
                             and len(x["document"]) > 1.5 * len(x["summary"]))

    # Adding a the doc length of each instance
    dataset = dataset.map(lambda x: {'doc_len': len(x['document'])})

    # Split the dataset into train, validation, and test sets after shuffling
    train_dataset, val_dataset, test_dataset = split_data(dataset, train_split, val_split)

    # Sort the datasets by document length
    train_dataset = train_dataset.sort('doc_len')
    val_dataset = val_dataset.sort('doc_len')
    test_dataset = test_dataset.sort('doc_len')

    return train_dataset, val_dataset, test_dataset


def split_data(dataset, train_split, val_split):
    # Shuffle the dataset
    dataset = dataset.shuffle()
    dataset = dataset.flatten_indices()  # rewrite the shuffled dataset on disk again as contiguous chunks for speed
    
    dataset = dataset.train_test_split(test_size=1-train_split)
    train_dataset = dataset['train']
    test_dataset = dataset['test'].train_test_split(test_size=1-(val_split/(1-train_split)))
    val_dataset = test_dataset['train']
    test_dataset = test_dataset['test']

    return train_dataset, val_dataset, test_dataset


def init_schedule(optimizer, sched, train_loader, lr, epochs, emb_dim):
    warmup_steps = 0.05 * len(train_loader) * epochs
    if sched == "constant" or sched == "none":
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1)
    elif sched == "cosineannealing":
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=0.0, last_epoch=-1)
    elif sched == "invsqrt":
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1/math.sqrt(epoch) if epoch > 0 else 1)
    elif sched == "linear":
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda current_step: current_step/warmup_steps if current_step < warmup_steps else 1.0)
    elif sched == "onecycle":
        scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=lr, total_steps=len(train_loader)*epochs, pct_start=0.05, anneal_strategy="linear")
    elif sched == "noam":  # TODO: fix this!!
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda current_step: (emb_dim ** -0.5) * min((current_step+1) ** -0.5, (current_step+1) * (warmup_steps ** -1.5)))
    elif sched == "plateau":
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=True)
    elif sched == "warmup":
        warmup_steps = 1000
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return (lr - lr/10) / warmup_steps * current_step + lr/10
            else:
                return ((1/5 * lr) - lr) / (len(train_loader)*epochs - warmup_steps) * (current_step - warmup_steps) + lr
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    else:
        raise ValueError("Invalid scheduler option provided.")
    return scheduler


def save_best_model(model, epoch, model_params):  # Local
    models_dir = os.path.join(os.path.dirname(__file__), "models", wandb.run.name)
    
    if not os.path.exists(models_dir):
        os.makedirs(models_dir, exist_ok=True)
    
    model_path = os.path.join(models_dir, f"model_{wandb.run.name}_e{epoch}.pt")
    model_info = {
        'params': model_params,
        'state_dict': model.state_dict()
    }
    
    torch.save(model_info, model_path)
    wandb.save(model_path)