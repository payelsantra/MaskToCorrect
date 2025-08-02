import argparse
import json
import pickle
from tqdm import tqdm
from transformers import AutoTokenizer


def all_lower(tokens):
    return [t.lower() for t in tokens]


def all_in(tokens, token_set):
    return [t in token_set for t in tokens]


def mask_uncommon(tokens, mask_flags, mask_token="[MASK]"):
    return [t if keep else mask_token for t, keep in zip(tokens, mask_flags)]


def common_tokens(tokens1, tokens2):
    common = set(all_lower(tokens1)).intersection(all_lower(tokens2))
    mask_flags = all_in(all_lower(tokens1), common)
    masked = mask_uncommon(tokens1, mask_flags)
    input_mask = [i for i, keep in enumerate(mask_flags) if not keep]
    return masked, input_mask
class Masker:
    def mask(self, data,args):
        tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        for idx in tqdm((data.keys())):
            input_claim = data[idx]["input_claim"]
    
            # Tokenize input claim
            tokens = tokenizer.tokenize(input_claim)
            detok = tokenizer.convert_tokens_to_string(tokens)
            split_claim = detok.split()
    
            # Get top-k evidence strings and tokenize
            evidence_strings = data[idx]["evidence"]
            joined_evidence = " ".join(evidence_strings)
    
            evidence_tokens = tokenizer.tokenize(joined_evidence)
            evidence_text = tokenizer.convert_tokens_to_string(evidence_tokens).split()
    
            # Mask input claim based on evidence overlap
            masked_tokens, input_mask = common_tokens(split_claim, evidence_text)
            masked_claim = " ".join(masked_tokens)
    
            # Store output
            data[idx]["masked_claim"]=[masked_claim]
        #print(data[idx])
        return data

