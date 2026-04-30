import multiprocessing
import math
from datasets import load_dataset
import os
from transformers import AutoTokenizer
import numpy as np

''' The code in this file is used to create 100 million token shards from the Nemotron-ClimbMix dataset.
The dataset is streamed and sharded using the number of CPU cores available. 
Each shard is saved as a numpy array of uint16 tokens, with a bos token at the beginning of each shard. 

The process is repeated until 50 shards are created, which will contain a total of ~5 billion tokens.'''

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

    
