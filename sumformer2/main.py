import random
import time
import fire
import numpy as np
import torch
import wandb

from torch.cuda.amp import autocast, GradScaler
from torch.nn.functional import pad
from torch.optim import AdamW
from transformers import T5Tokenizer

from .transformer import Sumformer
from .utils import *


INTERVAL = 100  # Init logging interval
MIN_INPUT_LEN = 50
MAX_INPUT_LEN = 256


def main(train=True, test=False, epochs=1, batch_size=16, lr=3.4e-4, sched="onecycle", emb_dim=512, max_out_len=30, clip=0.0, sample=None, load=None, pos_enc=False, GLU=False, gen="greedy", ignore_pad=False,
         enc_heads=8, enc_hidden=6, enc_depth=8, enc_dropout=0.3,
         dec_heads=8, dec_hidden=6, dec_depth=8, dec_dropout=0.3):
    # Ensure deterministic behavior
    set_seed(69420)

    best_val_loss = float("inf")  # Init best loss
    running_time = 0.0  # Init throughput measurements
    running_tokens = 0
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device "{device}"')

    # Initialize wandb
    wandb.init(project="Sumformer", entity="ryanott", config={
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "emb_dim": emb_dim,
        "max_out_len": max_out_len,
        "max_model_len": max(MAX_INPUT_LEN, max_out_len),
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
        "pos_enc": pos_enc,
        "GLU": GLU,
        "inference_type": gen,
        "ignore_padding": ignore_pad
    })

    # Load dataset and prepare DataLoader
    # train_dataset, val_dataset, test_dataset = load_reddit(0.8, 0.1, min_len=MIN_INPUT_LEN)
    train_dataset, val_dataset, test_dataset = load_xsum()
    if sample is not None:
        print(f"Sampling {sample} rows from dataset")
        train_dataset = train_dataset.select(range(sample))
        val_dataset = val_dataset.select(range(sample))
        test_dataset = test_dataset.select(range(sample))
    print("Number of rows: ", len(train_dataset))

    # Init the T5 tokenizer
    tokenizer = setup_tokenizer()

    def collate_fn(batch):
        docs = [item['document'] for item in batch]
        summaries = [item['summary'] for item in batch]

        # encode, truncate and pad to the length of the longest sequence in the batch
        encoder_inputs = tokenizer(docs, truncation=True, padding='longest', return_tensors='pt')
        decoder_inputs = tokenizer(summaries, truncation=True, padding='longest', return_tensors='pt')

        # TODO: Pad manually cause this doesn't seem to work

        # create attention masks to ignore padding tokens
        encoder_inputs["padding_mask"] = encoder_inputs["input_ids"].ne(tokenizer.pad_token_id)
        decoder_inputs["padding_mask"] = decoder_inputs["input_ids"].ne(tokenizer.pad_token_id)

        return encoder_inputs.to(device), decoder_inputs.to(device)

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

    scaler = GradScaler()  # Init gradient scaler for mixed precision training

    if train:
        model = Sumformer(device, emb_dim, len(tokenizer.get_vocab()), max(MAX_INPUT_LEN, max_out_len), pos_enc, GLU, enc_heads, enc_hidden, enc_depth, enc_dropout, dec_heads, dec_hidden, dec_depth, dec_dropout).to(device)
        optimizer = AdamW(model.parameters(), lr)
        criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id if ignore_pad else -100, reduction='mean')  # * see without padding mask in decoder
        scheduler = init_schedule(optimizer, sched, train_loader, lr, epochs, emb_dim)

        for epoch in range(epochs):
            # torch.autograd.set_detect_anomaly(True)  # ! REMOVE WHEN NOT NEEDED
            # -----TRAINING-----
            model.train()
            print("Training...")
            print(f"Epoch {epoch+1}")
            for b_idx, (encoder_inputs, decoder_inputs) in enumerate(train_loader):
                start_time = time.time()
                
                optimizer.zero_grad()

                # gradually decrease teacher forcing
                teacher_forcing_prob = 1.0 - (epoch / epochs)
                teacher_forcing = random.random() < teacher_forcing_prob

                with autocast():
                    if teacher_forcing:
                        train_outputs = model(source=encoder_inputs["input_ids"], target=decoder_inputs["input_ids"])
                        train_logits = train_outputs[:, :-1, :].contiguous()  # shift the decoder inputs one to the right
                    else:
                        train_outputs, train_logits = model.greedy(encoder_inputs["input_ids"], start_token=tokenizer.bos_token_id, end_token=tokenizer.eos_token_id, max_len=max_out_len, logits=True)
                    
                    train_targets = decoder_inputs["input_ids"][:, 1:].contiguous()  # shift the targets one to the left

                    # Make the logits and targets same size in the sequence dimension
                    train_logits, train_targets = pad_sequences(train_logits, train_targets, pad_token=tokenizer.pad_token_id)

                    loss = criterion(train_logits.view(-1, train_logits.size(-1)), train_targets.view(-1))

                scaler.scale(loss).backward()

                # clip the gradients if set
                if clip > 0.0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                
                scaler.step(optimizer)
                scaler.update()

                # measure throughput
                end_time = time.time()
                elapsed_time = end_time - start_time
                num_tokens = encoder_inputs["input_ids"].size(1) * batch_size  # number of tokens in the batch
                running_time += elapsed_time
                running_tokens += num_tokens

                if b_idx % INTERVAL == 0:
                    wandb.log({"Throughput": running_tokens / running_time})
                    running_time = 0.0
                    running_tokens = 0

                # log and update the learning rate
                scheduler.step()
                wandb.log({"Learning Rate": scheduler.get_last_lr()[0]})
                
                # log the gradient norm to wandb
                grad_norm = 0.0
                for name, param in model.named_parameters():
                    if "pos_embedding" not in name and param.grad is not None:  # ignore positional encoding as it is not learned
                        grad_norm += param.grad.data.norm(2).item() ** 2
                grad_norm = grad_norm ** 0.5
                wandb.log({"Gradient L2 norm": grad_norm})

                # log the loss value to wandb and print
                if b_idx % INTERVAL == 0:
                    print(f"Batch {b_idx+1} - Train loss: {loss.item()}")
                wandb.log({"Training Loss": loss.item()})

                # Clear CUDA cache
                torch.cuda.empty_cache()
            
            # -----VALIDATION-----
            print("Validating...")
            model.eval()
            total_val_loss = 0.0
            with torch.no_grad():
                for b_idx, (encoder_inputs, decoder_inputs) in enumerate(val_loader):
                    with autocast():
                        val_outputs, val_logits = model.greedy(encoder_inputs["input_ids"], start_token=tokenizer.bos_token_id, end_token=tokenizer.eos_token_id, max_len=max_out_len, logits=True)

                        # Decode an example output
                        if b_idx == len(val_loader) - 1:
                            example_input = tokenizer.decode(encoder_inputs["input_ids"][0], skip_special_tokens=True)
                            example_target = tokenizer.decode(decoder_inputs["input_ids"][0], skip_special_tokens=True)
                            example_output = tokenizer.decode(val_outputs[0], skip_special_tokens=True)
                            print(f"Example input: {example_input}")
                            print(f"Example target: {example_target}")
                            print(f"Example output: {example_output}")

                        val_targets = decoder_inputs["input_ids"][:, 1:].contiguous()  # shift the targets one to the left

                        # Make the logits and targets same size in the sequence dimension
                        val_logits, val_targets = pad_sequences(val_logits, val_targets, pad_token=tokenizer.pad_token_id)

                        loss = criterion(val_logits.view(-1, val_logits.size(-1)), val_targets.view(-1))

                        total_val_loss += loss.item()

            # log the validation loss to wandb
            avg_val_loss = total_val_loss / len(val_loader)
            wandb.log({"Validation Loss": avg_val_loss})
            print(f"Epoch {epoch+1} validation loss: {avg_val_loss}")

            # Save the trained model if it has best val loss
            if avg_val_loss < best_val_loss:
                print("Saving model with better validation loss")
                best_val_loss = avg_val_loss
                model_params = {
                    'device': device,
                    'emb_dim': emb_dim,
                    'vocab_size': tokenizer.vocab_size,
                    'max_len': max(MAX_INPUT_LEN, max_out_len),
                    'pos_encoding': pos_enc,
                    'GLU': GLU,
                    'enc_heads': enc_heads,
                    'enc_hidden': enc_hidden,
                    'enc_depth': enc_depth,
                    'enc_dropout': enc_dropout,
                    'dec_heads': dec_heads,
                    'dec_hidden': dec_hidden,
                    'dec_depth': dec_depth,
                    'dec_dropout': dec_dropout
                }
                save_best_model(model, epoch, model_params)
            
            # Clear CUDA cache
            torch.cuda.empty_cache()
    
    # -----TESTING-----
    if test:
        print("Testing...")
        if load is not None:  # TODO automatically load the correct params
            test_model = Sumformer(device, emb_dim, len(tokenizer.get_vocab()), max(MAX_INPUT_LEN, max_out_len), pos_enc, GLU, enc_heads, enc_hidden, enc_depth, enc_dropout, dec_heads, dec_hidden, dec_depth, dec_dropout).to(device)
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
                if gen == "greedy":
                    test_outputs = test_model.greedy(
                        encoder_inputs["input_ids"], start_token=tokenizer.bos_token_id, end_token=tokenizer.eos_token_id, max_len=max_out_len)
                elif gen == "beam":  # ! DOESN'T WORK YET
                    test_outputs = test_model.beam(encoder_inputs["input_ids"], start_token=tokenizer.bos_token_id, end_token=tokenizer.eos_token_id, max_len=max_out_len, source_mask=None)
                else:
                     raise ValueError(f"{gen} is not a valid generation method")
                generated_text = tokenizer.decode(test_outputs, skip_special_tokens=False)
                
                print(f"\nBatch {b_idx+1}")
                print(f"Generated ids shape: {test_outputs.shape}")
                print(f"Generated ids: {test_outputs}")
                print(f"Generated ids decoded: {generated_text}")

        # log the test loss to wandb
        avg_test_loss = total_test_loss / len(test_loader)
        wandb.log({"Test Loss": avg_test_loss})
    
    wandb.finish()


def pad_sequences(seq1, seq2, pad_token):
    seq1_len, seq2_len = seq1.size(1), seq2.size(1)

    # Determine the maximum sequence length and pad the shorter sequence
    max_seq_len = max(seq1_len, seq2_len)
    if seq1_len < max_seq_len:
        padding_size = max_seq_len - seq1_len
        seq1 = pad(seq1, pad=(0, 0, 0, padding_size), value=pad_token)
    elif seq2_len < max_seq_len:
        padding_size = max_seq_len - seq2_len
        seq2 = pad(seq2, pad=(0, padding_size), value=pad_token)

    return seq1, seq2


def setup_tokenizer():
    tokenizer = T5Tokenizer.from_pretrained('t5-base', use_fast=True, model_max_length=MAX_INPUT_LEN)
    tokenizer.bos_token = "<s>"
    tokenizer.bos_token_id = tokenizer.convert_tokens_to_ids("<s>")
    tokenizer.add_special_tokens({"bos_token": "<s>"})

    return tokenizer


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_n_sum(model_path, input_str, tokenizer, max_out_len):
    # Load the model
    model_info = torch.load(model_path)
    model_params = model_info['params']
    model = Sumformer(**model_params).to(model_params['device'])
    model.load_state_dict(model_info['state_dict'])
    model.eval()

    # Encode the input string
    input_ids = tokenizer.encode(input_str, return_tensors="pt").to(model_params['device'])

    # Generate the summary
    with torch.no_grad():
        summary_ids = model.greedy(input_ids, start_token=tokenizer.bos_token_id, end_token=tokenizer.eos_token_id, max_len=max_out_len)
    
    # Return the decoded summary
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


# if __name__ == '__main__':
#     fire.Fire(main)
    # main(train=False, test=True, load="models/fearless-oath-237/model_fearless-oath-237_e0.pt")
    # main(batch_size=8, sample=48, max_out_len=32, enc_heads=2, dec_heads=4, test=True)


def start():
    fire.Fire(main)