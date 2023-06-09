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

from torch.utils.data import DataLoader, Dataset, Sampler, SequentialSampler, BatchSampler


def create_data_loader(dataset, batch_size, collate_fn):
    sampler = SequentialSampler(dataset)
    batch_sampler = BatchSampler(sampler, batch_size, drop_last=False)
    return DataLoader(dataset, batch_sampler=batch_sampler, collate_fn=collate_fn)


def main(epochs=1, batch_size=16, lr=5e-4, sched="onecycle", emb_dim=512, max_len=512, clip=None, enc_heads=1, enc_hidden=1, enc_depth=1, enc_dropout=0.1, dec_heads=1, dec_hidden=1, dec_depth=1, dec_dropout=0.1, sample=None):
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
        "schedule": sched,
        "clip": clip,
    })

    # Load dataset and prepare DataLoader
    train_dataset, val_dataset, test_dataset = load_reddit(0.8, 0.1, min_len=50)

    tokenizer = T5Tokenizer.from_pretrained('t5-base', use_fast=True, model_max_length=512)
    PAD_TOKEN_ID = tokenizer.pad_token_id  # 0
    VOCAB_SIZE = len(tokenizer.get_vocab())

    def collate_fn(batch):
        docs = [item['document'] for item in batch]
        summaries = [item['summary'] for item in batch]

        # encode, truncate and pad to the length of the longest sequence in the batch
        encoder_inputs = tokenizer(docs, truncation=True, padding='longest', return_tensors='pt')
        decoder_inputs = tokenizer(summaries, truncation=True, padding='longest', return_tensors='pt')

        # TODO: Pad manually like a real man (cause this doesn't seem to work)

        # create attention masks to ignore padding tokens
        encoder_inputs["padding_mask"] = encoder_inputs["input_ids"].ne(PAD_TOKEN_ID)
        decoder_inputs["padding_mask"] = decoder_inputs["input_ids"].ne(PAD_TOKEN_ID)

        return encoder_inputs.to(device), decoder_inputs.to(device)

    if sample is not None:
        train_dataset = train_dataset.select(range(sample))
        val_dataset = val_dataset.select(range(sample))
        test_dataset = test_dataset.select(range(sample))

    train_loader = create_data_loader(train_dataset, batch_size, collate_fn)
    val_loader = create_data_loader(val_dataset, batch_size, collate_fn)
    test_loader = create_data_loader(test_dataset, batch_size, collate_fn)
    
    # Printing some sample data and padding mask
    # data_iter = iter(train_loader)
    # for i in range(3):
    #     encoder_inputs, decoder_inputs = next(data_iter)
    #     print(f"Encoder inputs shape: {encoder_inputs['input_ids'].shape}")
    #     first_doc = encoder_inputs["input_ids"][0]
    #     print(f"First doc: {first_doc}")
    #     first_doc_mask = encoder_inputs["padding_mask"][0]
    #     print(f"First doc mask: {first_doc_mask}")
    #     first_doc_decoded = tokenizer.decode(first_doc)
    #     print(f"First doc decoded: {first_doc_decoded}")


    # Define your model, optimizer and criterion
    model = Sumformer(device, emb_dim, VOCAB_SIZE, max_len, enc_heads, enc_hidden, enc_depth, enc_dropout, dec_heads, dec_hidden, dec_depth, dec_dropout)
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=PAD_TOKEN_ID, reduction='mean')  # * see without padding mask in decoder

    if sched == "constant" or sched == "none":
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1)
    elif sched == "cosinedecay":
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader) * epochs)
    elif sched == "invsqrt":
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1/math.sqrt(epoch) if epoch > 0 else 1)
    elif sched == "linear":
        scheduler = lr_scheduler.LinearLR(optimizer, start_factor=lr/5, end_factor=lr, total_iters=len(train_loader)*epochs)
    elif sched == "onecycle":
        scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=lr, total_steps=len(train_loader)*epochs, pct_start=0.3, anneal_strategy="linear")
    else:
        raise ValueError("Invalid scheduler option provided.")
    
    # Init best loss
    best_val_loss = float("inf")

    for epoch in range(epochs):
        # torch.autograd.set_detect_anomaly(True)  # ! REMOVE WHEN NOT NEEDED
        # -----TRAINING-----
        model.train()

        print(f"Epoch {epoch+1}")
        for b_idx, (encoder_inputs, decoder_inputs) in enumerate(train_loader):
            print(f"Batch: {b_idx+1}")
            # reset the gradients
            optimizer.zero_grad()

            # forward pass  # ! NO SOURCE MASK IS BEING PASSED RIGHT NOW
            outputs = model(source=encoder_inputs["input_ids"], target=decoder_inputs["input_ids"], source_mask=encoder_inputs["padding_mask"])

            # shift the decoder inputs to the right by 1 (for teacher forcing technique)
            shifted_outputs = outputs[:, :-1, :].contiguous()
            shifted_labels = decoder_inputs["input_ids"][:, 1:].contiguous()

            # compute the loss
            loss = criterion(shifted_outputs.view(-1, shifted_outputs.size(-1)), shifted_labels.view(-1))
            # loss = ((~decoder_inputs["attention_mask"].to(torch.float)) * loss).mean()

            # backpropagate the loss
            loss.backward()

            # clip the gradients if set
            if clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            
            # log the gradient norm to wandb
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

            # log the loss value to wandb and print every 100 batches
            if b_idx % 100 == 0:
                print(f"Batch {b_idx+1} - Train loss: {loss.item()}")
            wandb.log({"train_loss": loss.item()})
        
        # -----VALIDATION-----
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for b_idx, (encoder_inputs, decoder_inputs) in enumerate(val_loader):
                # forward pass
                outputs = model(source=encoder_inputs["input_ids"], target=decoder_inputs["input_ids"], source_mask=encoder_inputs["padding_mask"])

                # shift the decoder inputs to the right by 1 (for teacher forcing technique)
                shifted_outputs = outputs[:, :-1, :].contiguous()
                shifted_labels = decoder_inputs["input_ids"][:, 1:].contiguous()

                # compute the loss
                loss = criterion(shifted_outputs.view(-1, shifted_outputs.size(-1)), shifted_labels.view(-1))
                # loss = ((~decoder_inputs["attention_mask"].to(torch.float)) * loss).mean()
                total_val_loss += loss.item()
        
        # log the validation loss to wandb
        avg_val_loss = total_val_loss / len(val_loader)
        wandb.log({"val_loss": avg_val_loss})
        print(f"Epoch {epoch+1} validation loss: {avg_val_loss}")

    # Save the trained model if it has best val loss
    if avg_val_loss < best_val_loss:
        print("Saving model...")
        best_val_loss = avg_val_loss
        if not os.path.exists(f"models/{wandb.run.name}"):
            os.mkdir(f"models/{wandb.run.name}")
        torch.save(model.state_dict(), f"models/{wandb.run.name}/model_{wandb.run.name}_e{epoch}.pt")
        wandb.save(f"models/{wandb.run.name}/model_{wandb.run.name}_e{epoch}.pt")


if __name__ == '__main__':
    fire.Fire(main)

