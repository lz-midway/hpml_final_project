import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
# from torchtext.datasets import WikiText2
# from torchtext.data.utils import get_tokenizer
# from torchtext.vocab import build_vocab_from_iterator
import os
from torch.utils.data import DataLoader, DistributedSampler

# Dataloader for Food 101 and WikiText
def get_food101_dataloaders(distributed=False,
                            rank=0,
                            world_size=1,
                            data_root = "./data",
                            batch_size = 64,
                            num_workers = 4):
    os.makedirs(data_root, exist_ok=True)

    tfms_train = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    tfms_test = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    train_set = datasets.Food101(root=data_root, split="train",
                                 transform=tfms_train, download=True)
    test_set  = datasets.Food101(root=data_root, split="test",
                                 transform=tfms_test, download=True)
    
    if distributed:
        train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True)
        test_sampler = DistributedSampler(test_set, num_replicas=world_size, rank=rank, shuffle=False)
    else:
        train_sampler = None
        test_sampler = None

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=(train_sampler is None),
                              sampler=train_sampler, num_workers=num_workers)
    test_loader  = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                              sampler=test_sampler, num_workers=num_workers)

    return train_loader, test_loader

# def get_wikitext_dataloaders(batch_size = 32,
#                              seq_len = 32):
#     tokenizer = get_tokenizer("basic_english")

#     train_iter = WikiText2(split="train")
#     valid_iter = WikiText2(split="valid")

#     def yield_tokens(data_iter):
#         for text in data_iter:
#             yield tokenizer(text)

#     vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
#     vocab.set_default_index(vocab["<unk>"])

#     train_iter = WikiText2(split="train")
#     valid_iter = WikiText2(split="valid")

#     def data_process(raw_text_iter):
#         data = [torch.tensor(vocab(tokenizer(item)), dtype=torch.long)
#                 for item in raw_text_iter]
#         return torch.cat(data)

#     train_data = data_process(train_iter)
#     valid_data = data_process(valid_iter)

#     def batchify(data):
#         n_batch = data.size(0) // (batch_size * seq_len)
#         data = data[:n_batch * batch_size * seq_len]
#         data = data.view(batch_size, -1)
#         return data

#     train_data = batchify(train_data)
#     valid_data = batchify(valid_data)

#     def get_batches(data):
#         for i in range(0, data.size(1) - seq_len, seq_len):
#             x = data[:, i:i+seq_len]
#             y = data[:, i+1:i+seq_len+1]
#             yield x, y

#     train_loader = list(get_batches(train_data))
#     valid_loader = list(get_batches(valid_data))

#     return train_loader, valid_loader, vocab