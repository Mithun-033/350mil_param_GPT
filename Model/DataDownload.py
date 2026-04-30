"""
Dataset download and tokenisation utility.

Streams the nvidia/Nemotron-ClimbMix dataset and splits it into 100-million-
token shards using all available CPU cores. Each shard is saved as a uint16
NumPy array with a BOS token prepended to every document. Running this script
to completion produces ~50 shards (~5 billion tokens total).
"""

import multiprocessing
import math
from datasets import load_dataset
import os
from transformers import AutoTokenizer
import numpy as np

tok=AutoTokenizer.from_pretrained("gpt2")
bos_id=tok.bos_token_id

ds = load_dataset("nvidia/Nemotron-ClimbMix",streaming=True,split="train")

num_shards=os.cpu_count()

def shard_100mil(num_shards,idx):
    shards=ds.shard(num_shards=num_shards, index=idx)
    shard=[]
    count=0

    for row in shards:
        length=row["token_count"]
        if length>1024:
            continue
        shard.extend([bos_id]+row["tokens"])
        count+=length
        if count>=100_000_000:
            break
    np.save(f"shard_{idx}.npy",np.array(shard,dtype=np.uint16))
    
if __name__=="__main__":
    for _ in range(math.ceil(50/num_shards)):
        processes=[]

        for idx in range(num_shards):
            process=multiprocessing.Process(target=shard_100mil,args=[num_shards,idx])
            process.start()
            processes.append(process)

        for process in processes:
            process.join()

    
