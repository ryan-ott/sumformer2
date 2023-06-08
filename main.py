import itertools
import math
import random
import fire
import numpy as np
import os
import torch
import wandb

from torch.optim import AdamW, lr_scheduler
from torch.utils.data import DataLoader
from transformers import T5Tokenizer

from transformer import Sumformer
from utils import load_reddit


def main(epochs=1, batch_size=16, lr=5e-4, sched="onecycle", emb_dim=512, max_len=512, enc_heads=1, enc_hidden=1, enc_depth=1, enc_dropout=0.1, dec_heads=1, dec_hidden=1, dec_depth=1, dec_dropout=0.1, sample=None):
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    random.seed(69420)
    np.random.seed(69420)
    torch.manual_seed(69420)
    torch.cuda.manual_seed_all(69420)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device "{device}"')

    # Initialize wandb
    wandb.init(project="Sumformer", entity="ryanott", config={
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "emb_dim": emb_dim,
        "max_len": max_len,
        "enc_heads": enc_heads,
        "enc_hidden": enc_hidden,
        "enc_depth": enc_depth,
        "enc_dropout": enc_dropout,
        "dec_heads": dec_heads,
        "dec_hidden": dec_hidden,
        "dec_depth": dec_depth,
        "dec_dropout": dec_dropout,
        "schedule": sched
    })

    # Load dataset and prepare DataLoader
    train_dataset, val_dataset, test_dataset = load_reddit(0.8, 0.1, min_len=50)
    tokenizer = T5Tokenizer.from_pretrained('t5-base', use_fast=True, model_max_length=512)
    PAD_TOKEN_ID = tokenizer.pad_token_id  # 0
    VOCAB_SIZE = len(tokenizer.get_vocab())

    def collate_fn(batch):
        docs = [item['document'] for item in batch]
        summaries = [item['summary'] for item in batch]

        encoder_inputs = tokenizer(docs, truncation=True, padding='longest', return_tensors='pt')
        decoder_inputs = tokenizer(summaries, truncation=True, padding='longest', return_tensors='pt')

        # create attention masks to ignore padding tokens
        encoder_inputs["padding_mask"] = encoder_inputs["input_ids"].ne(PAD_TOKEN_ID)
        decoder_inputs["padding_mask"] = decoder_inputs["input_ids"].ne(PAD_TOKEN_ID)

        return encoder_inputs.to(device), decoder_inputs.to(device)

    if sample is not None:
        train_dataset = train_dataset.select(range(sample))
    train_loader = DataLoader(train_dataset, batch_size, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size, collate_fn=collate_fn)

    # Define your model, optimizer and criterion
    model = Sumformer(device, emb_dim, VOCAB_SIZE, max_len, enc_heads, enc_hidden, enc_depth, enc_dropout, dec_heads, dec_hidden, dec_depth, dec_dropout)
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=PAD_TOKEN_ID, reduction='mean')  # * see without padding mask in decoder
    lr_schedules = {
        "linear": lr_scheduler.LinearLR(optimizer, start_factor=lr/10, end_factor=lr, total_iters=len(train_loader)*epochs),
        "constant": lr_scheduler.ConstantLR(optimizer, factor=1, total_iters=0),  # an actual constant learning rate
        "onecycle": lr_scheduler.OneCycleLR(optimizer, max_lr=lr, total_steps=len(train_loader)*epochs, pct_start=0.3, anneal_strategy="linear"),
        "invsqrt": lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1/math.sqrt(epoch) if epoch > 0 else 1),
        "cosinedecay": lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader) * epochs)}
    scheduler = lr_schedules[sched]

    # Training loop
    model.train()
    for epoch in range(epochs):
        # torch.autograd.set_detect_anomaly(True)  # ! REMOVE WHEN NOT NEEDED

        print(f"Epoch {epoch+1}")
        for b_idx, (encoder_inputs, decoder_inputs) in enumerate(train_loader):
            print(f"Batch: {b_idx+1}")
            # reset the gradients
            optimizer.zero_grad()

            # forward pass
            outputs = model(source=encoder_inputs["input_ids"], target=decoder_inputs["input_ids"], source_mask=encoder_inputs["padding_mask"])

            # shift the decoder inputs to the right by 1 (for teacher forcing technique)
            shifted_outputs = outputs[:, :-1, :].contiguous()
            shifted_labels = decoder_inputs["input_ids"][:, 1:].contiguous()

            # compute the loss
            loss = criterion(shifted_outputs.view(-1, shifted_outputs.size(-1)), shifted_labels.view(-1))
            # loss = ((~decoder_inputs["attention_mask"].to(torch.float)) * loss).mean()

            # backpropagate the loss
            loss.backward()

            # gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) TODO: look into this
            grad_norm = 0.0
            for param in model.parameters():
                grad_norm += param.grad.data.norm(2).item() ** 2
            grad_norm = grad_norm ** 0.5
            wandb.log({"Gradient L2 norm": grad_norm})

            # update the weights
            optimizer.step()

            # log and update the learning rate
            scheduler.step()
            wandb.log({"lr": scheduler.get_last_lr()[0]})

            # log the loss value to wandb and print every 20 batches
            if b_idx % 100 == 0:
                print(f"Batch {b_idx+1} - Loss: {loss.item()}")
            wandb.log({"loss": loss.item()})

    # Save the trained model
    if not os.path.exists("models"):
            os.mkdir("models")
    torch.save(model.state_dict(), f"models/model_{wandb.run.name}_{epoch+1}.pt")  # TODO: only save the model with best loss
    wandb.save(f"models/model_{epoch+1}.pt")


if __name__ == '__main__':
    fire.Fire(main)

