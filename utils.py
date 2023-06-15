import math
import os

from datasets import load_dataset, concatenate_datasets
import torch
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, SequentialSampler, BatchSampler
import wandb


def create_data_loader(dataset, batch_size, collate_fn):
    sampler = SequentialSampler(dataset)
    batch_sampler = BatchSampler(sampler, batch_size, drop_last=False)
    return DataLoader(dataset, batch_sampler=batch_sampler, collate_fn=collate_fn)


def load_reddit(train_split, val_split, min_len=50):
    """Concatenate the short and long reddit TIFU datasets and split into train, validation, and test sets. Keep only the docs and their summary."""
    dataset_short = load_dataset("reddit_tifu", "short")
    dataset_short = dataset_short.remove_columns(['ups', 'num_comments', 'upvote_ratio', 'score', 'tldr'])
    dataset_short = dataset_short.rename_columns({'documents': 'document', 'title': 'summary'})

    dataset_long = load_dataset("reddit_tifu", "long")
    dataset_long = dataset_long.remove_columns(['ups', 'num_comments', 'upvote_ratio', 'score', 'title'])
    dataset_long = dataset_long.rename_columns({'documents': 'document', 'tldr': 'summary'})

    dataset = concatenate_datasets([dataset_short["train"], dataset_long["train"]])

    # Filtering out too short documents and summaries
    dataset = dataset.filter(lambda x:
                             len(x["document"]) > min_len
                             and len(x["summary"]) > min_len
                             and len(x["document"]) > len(x["summary"]))

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
    warmup_steps = 0.3 * len(train_loader) * epochs
    if sched == "constant" or sched == "none":
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1)
    elif sched == "cosineannealing":
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=0.0, last_epoch=-1)
    elif sched == "invsqrt":
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1/math.sqrt(epoch) if epoch > 0 else 1)
    elif sched == "linear":
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda current_step: current_step/warmup_steps if current_step < warmup_steps else 1.0)
    elif sched == "onecycle":
        scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=lr, total_steps=len(train_loader)*epochs, pct_start=0.3, anneal_strategy="linear")
    elif sched == "noam":  # TODO: fix this!!
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda current_step: (emb_dim ** -0.5) * min((current_step+1) ** -0.5, (current_step+1) * (warmup_steps ** -1.5)))
    else:
        raise ValueError("Invalid scheduler option provided.")
    return scheduler


def save_best_model(model, epoch, model_params):
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
