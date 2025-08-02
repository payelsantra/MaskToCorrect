import argparse
import json
import math
import random
import pickle
from tqdm import tqdm
from transformers import AutoTokenizer
class Masker:
    def mask(self, data, args):
        sample_prob=0.15
        tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        for idx in tqdm(list(data.keys())):
            input_claim = data[idx]["input_claim"]
    
    		# Tokenize using BERT tokenizer
            tokens = tokenizer.tokenize(input_claim)
            detok_text = tokenizer.convert_tokens_to_string(tokens)
            split_claim = detok_text.split()
    
    		# Determine number of tokens to mask
            num_tokens = len(split_claim)
            num_to_mask = max(1, min(num_tokens, math.ceil(num_tokens * sample_prob)))
            mask_indices = random.sample(range(num_tokens), k=num_to_mask)
    
    		# Create masked claim
            masked_tokens = split_claim[:]
            for i in mask_indices:
                masked_tokens[i] = "[MASK]"
            masked_claim = " ".join(masked_tokens)
    
            data[idx]["masked_claim"] = masked_claim  # Use index as ID
        print(data[idx])

        return data

