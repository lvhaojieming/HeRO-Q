import sys
import os
import argparse
import random
import time
import math
import logging
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
sys.path.append(os.path.abspath("Megatron-LM"))
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig
from megatron.core.datasets.utils import Split
from megatron.core.tokenizers import MegatronTokenizerBase


DEFAULT_MODEL_PATH = "/home/zjh/model/Qwen-2.5-3B-Instruct"
DEFAULT_DATA_PATH = "/home/zjh/C4_Qwen/C4_Qwen_text_document"


def setup_logger(rank):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO if rank == 0 else logging.WARNING)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO if rank == 0 else logging.WARNING)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_ddp():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        print("Not using DDP, falling back to single process (CPU/GPU).")
        return 0, 0, 1 
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=rank)
    return local_rank, rank, world_size

def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()

class LocalQwenTokenizer(MegatronTokenizerBase):
    def __init__(self, path):
        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, local_files_only=True)
        self.eod_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>") or self.tokenizer.eos_token_id
    def vocab_size(self): return len(self.tokenizer)
    def vocab(self): return self.tokenizer.get_vocab()
    @property
    def eod(self): return self.eod_id
    def tokenize(self, t): return self.tokenizer.encode(t)
    def detokenize(self, i): return self.tokenizer.decode(i)
    def apply_chat_template(self, m, **k): return self.tokenizer.apply_chat_template(m, **k)
    @property
    def unique_identifiers(self): return {"class": "LocalQwenTokenizer", "path": self.tokenizer.name_or_path}


def get_dataloader(data_path, model_path, batch_size, seq_len, seed, rank, world_size):
    config = GPTDatasetConfig(
        random_seed=seed, 
        sequence_length=seq_len, 
        tokenizer=LocalQwenTokenizer(model_path),
        reset_position_ids=True, 
        reset_attention_mask=True, 
        eod_mask_loss=True,
        blend=[[data_path], None], 
        split="98,2,0",
        path_to_cache=os.path.join(os.path.dirname(data_path), "cache"),
        mmap_bin_files=True
    )
    
    def is_rank0():
        return rank == 0

    builder = BlendedMegatronDatasetBuilder(GPTDataset, [None,None,None], is_rank0, config)
    train_ds = builder.build()[Split.train.value]
    sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=False, drop_last=True) 
    loader = DataLoader(
        train_ds, 
        batch_size=batch_size, 
        sampler=sampler,
        num_workers=4,          
        pin_memory=True,        
        prefetch_factor=2,      
        persistent_workers=True 
    )
    return loader, sampler
