import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoTokenizer

dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)
val_dataset = load_dataset("allenai/c4", "en", split="validation", streaming=True)
tokenizer_name = "gpt2"           
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token   



class BlockDataset(IterableDataset):
    """
    Streams raw samples, tokenises them, keeps an internal buffer,
    and yields fixed-length training blocks:
        item = {"input_ids": LongTensor[block_size],
                "labels":    LongTensor[block_size]}
    """
    def __init__(self, hf_iterable, tokenizer, block_size):
        self.data      = hf_iterable
        self.tok       = tokenizer
        self.block_sz  = block_size

    def __iter__(self):
        buffer = []                          # holds unfinished tokens
        for ex in self.data:                 # stream forever
            buffer.extend(
                self.tok(
                    ex["text"],
                    add_special_tokens=False,
                    truncation=False,
                ).input_ids
            )

            # while we have enough tokens for one more block (+1 for shift)
            while len(buffer) >= self.block_sz + 1:
                # slice out one contiguous chunk
                chunk  = buffer[: self.block_sz + 1]
                buffer = buffer[self.block_sz :]     # drop consumed part

                input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                labels     = torch.tensor(chunk[1:], dtype=torch.long)

                yield {"input_ids": input_ids, "labels": labels}



def collate_fn(batch):
    """
    batch is a list of `BATCH_BLOCKS` dicts returned by BlockDataset.
    Because every item already has the correct length, we just stack.
    """
    input_ids = torch.stack([item["input_ids"] for item in batch])
    labels    = torch.stack([item["labels"]    for item in batch])
    return input_ids, labels


def get_loader(batch_size, max_len, num_workers, prefetch_factor):
    
    block_stream = BlockDataset(dataset, tokenizer, max_len)
    val_block_stream = BlockDataset(val_dataset, tokenizer, max_len)
    
    dataloader = DataLoader(
        block_stream,                 # IterableDataset
        batch_size=batch_size,    
        collate_fn=collate_fn,
        num_workers=num_workers,            
        pin_memory=True,
        prefetch_factor=prefetch_factor 
    )    
    val_dataloader = DataLoader(
        val_block_stream,                 # IterableDataset
        batch_size=batch_size,    
        collate_fn=collate_fn,
        num_workers=num_workers,            
        pin_memory=True,
        prefetch_factor=prefetch_factor 
    )

    return dataloader, val_dataloader
    