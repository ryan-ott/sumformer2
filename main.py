import random
import time
import fire
import numpy as np
import os
import torch
import wandb

from torch.optim import AdamW
from transformers import T5Tokenizer

from transformer import Sumformer
from utils import load_reddit, init_schedule, create_data_loader


INTERVAL = 100
MIN_INPUT_LEN = 50
MAX_INPUT_LEN = 256


def main(train=True, test=False, epochs=1, batch_size=16, lr=5e-4, sched="onecycle", emb_dim=512, max_out_len=256, clip=None, sample=None, load=None, GLU=False,
         enc_heads=1, enc_hidden=1, enc_depth=1, enc_dropout=0.1,
         dec_heads=1, dec_hidden=1, dec_depth=1, dec_dropout=0.1):
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    random.seed(69420)
    np.random.seed(69420)
    torch.manual_seed(69420)
    torch.cuda.manual_seed_all(69420)

    best_val_loss = float("inf")  # Init best loss
    running_time = 0.0  # Init throughput measurements
    running_tokens = 0
    interval = 100  # Init logging interval
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device "{device}"')

    # Initialize wandb
    wandb.init(project="Sumformer", entity="ryanott", config={
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "emb_dim": emb_dim,
        "max_len": max_out_len,
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

    tokenizer = T5Tokenizer.from_pretrained('t5-base', use_fast=True, model_max_length=MAX_INPUT_LEN)
    tokenizer.bos_token = "<s>"
    tokenizer.bos_token_id = tokenizer.convert_tokens_to_ids("<s>")
    tokenizer.add_special_tokens({"bos_token": "<s>"})
    PAD_TOKEN_ID = tokenizer.pad_token_id  # 0
    BOS_TOKEN_ID = tokenizer.bos_token_id  # 32100
    EOS_TOKEN_ID = tokenizer.eos_token_id  # 1
    VOCAB_SIZE = len(tokenizer.get_vocab())

    def collate_fn(batch):
        docs = [item['document'] for item in batch]
        summaries = [item['summary'] for item in batch]

        # encode, truncate and pad to the length of the longest sequence in the batch
        encoder_inputs = tokenizer(docs, truncation=True, padding='longest', return_tensors='pt')
        decoder_inputs = tokenizer(summaries, truncation=True, padding='longest', return_tensors='pt')

        # TODO: Pad manually cause this doesn't seem to work

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

    if train:
        model = Sumformer(device, emb_dim, VOCAB_SIZE, max(MAX_INPUT_LEN, max_out_len), GLU, enc_heads, enc_hidden, enc_depth, enc_dropout, dec_heads, dec_hidden, dec_depth, dec_dropout).to(device)
        optimizer = AdamW(model.parameters(), lr)
        criterion = torch.nn.CrossEntropyLoss(ignore_index=PAD_TOKEN_ID, reduction='mean')  # * see without padding mask in decoder
        scheduler = init_schedule(optimizer, sched, train_loader, lr, epochs, emb_dim)

        for epoch in range(epochs):
            # torch.autograd.set_detect_anomaly(True)  # ! REMOVE WHEN NOT NEEDED
            # -----TRAINING-----
            model.train()

            print(f"Epoch {epoch+1}")
            for b_idx, (encoder_inputs, decoder_inputs) in enumerate(train_loader):
                start_time = time.time()

                # reset the gradients
                optimizer.zero_grad()

                # forward pass  # ! NO SOURCE MASK IS BEING PASSED RIGHT NOW
                outputs = model(source=encoder_inputs["input_ids"], target=decoder_inputs["input_ids"])#, source_mask=encoder_inputs["padding_mask"])

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
                
                # update the weights
                optimizer.step()

                # measure throughput
                end_time = time.time()
                elapsed_time = end_time - start_time
                num_tokens = encoder_inputs["input_ids"].size(1) * batch_size  # number of tokens in the batch
                running_time += elapsed_time
                running_tokens += num_tokens

                if b_idx % interval == 0:
                    wandb.log({"Throughput": running_tokens / running_time})
                    running_time = 0.0
                    running_tokens = 0

                # log and update the learning rate
                scheduler.step()
                wandb.log({"Learning Rate": scheduler.get_last_lr()[0]})
                
                # log the gradient norm to wandb
                grad_norm = 0.0
                for param in model.parameters():
                    grad_norm += param.grad.data.norm(2).item() ** 2
                grad_norm = grad_norm ** 0.5
                wandb.log({"Gradient L2 norm": grad_norm})

                # log the loss value to wandb and print
                if b_idx % interval == 0:
                    print(f"Batch {b_idx+1} - Train loss: {loss.item()}")
                wandb.log({"Training Loss": loss.item()})
            
            # -----VALIDATION-----
            model.eval()
            total_val_loss = 0.0
            with torch.no_grad():
                for b_idx, (encoder_inputs, decoder_inputs) in enumerate(val_loader):
                    # forward pass
                    outputs = model(source=encoder_inputs["input_ids"], target=decoder_inputs["input_ids"])#, source_mask=encoder_inputs["padding_mask"])

                    # shift the decoder inputs to the right by 1 (for teacher forcing technique)
                    shifted_outputs = outputs[:, :-1, :].contiguous()
                    shifted_labels = decoder_inputs["input_ids"][:, 1:].contiguous()

                    # compute the loss
                    loss = criterion(shifted_outputs.view(-1, shifted_outputs.size(-1)), shifted_labels.view(-1))
                    # loss = ((~decoder_inputs["attention_mask"].to(torch.float)) * loss).mean()
                    total_val_loss += loss.item()
            
            # log the validation loss to wandb
            avg_val_loss = total_val_loss / len(val_loader)
            wandb.log({"Validation Loss": avg_val_loss})
            print(f"Epoch {epoch+1} validation loss: {avg_val_loss}")

        # Save the trained model if it has best val loss
        if avg_val_loss < best_val_loss:
            print("Saving model...")
            best_val_loss = avg_val_loss
            models_dir = os.path.join(os.path.dirname(__file__), "models", wandb.run.name)
            if not os.path.exists(models_dir):
                os.makedirs(models_dir, exist_ok=True)
            model_path = os.path.join(models_dir, f"model_{wandb.run.name}_e{epoch}.pt")
            torch.save(model.state_dict(), model_path)
            wandb.save(model_path)
    
    # -----TESTING-----
    if test:
        if load is not None:
            test_model = Sumformer(device, emb_dim, VOCAB_SIZE, max(MAX_INPUT_LEN, max_out_len), GLU, enc_heads, enc_hidden, enc_depth, enc_dropout, dec_heads, dec_hidden, dec_depth, dec_dropout).to(device)
            test_model.load_state_dict(torch.load(f"{load}"))
        else:
            try:
                test_model = model
            except NameError:
                print("No model to test. Either train a model or load a model with --load <path>")
                return

        test_model.eval()
        total_test_loss = 0.0
        with torch.no_grad():
            for b_idx, (encoder_inputs, decoder_inputs) in enumerate(test_loader):
                # forward pass
                generated_ids = test_model.generate(
                    encoder_inputs["input_ids"], start_token=BOS_TOKEN_ID, end_token=EOS_TOKEN_ID, max_len=test_model.max_len, source_mask=encoder_inputs["padding_mask"])
                generated_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
                
                print(f"\nBatch {b_idx+1}")
                print(f"Generated ids shape: {generated_ids.shape}")
                print(f"Generated ids: {generated_ids}")
                print(f"Generated ids decoded: {generated_text}")

        # log the test loss to wandb
        avg_test_loss = total_test_loss / len(test_loader)
        wandb.log({"Test Loss": avg_test_loss})



if __name__ == '__main__':
    fire.Fire(main)
    # main(train=False, test=True, load="models/fearless-oath-237/model_fearless-oath-237_e0.pt")
