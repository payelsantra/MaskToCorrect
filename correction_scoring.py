import torch
import torch.nn as nn
from transformers import AutoTokenizer
from transformers.models.roberta.modeling_roberta import RobertaModel
from typing import Dict
import numpy as np
import rouge
from tqdm import tqdm
from typing import List, Dict
import nltk,pickle
nltk.download('punkt')
nltk.download('punkt_tab')


bert_hidden_dim = 1024
# pretrain_model_dir = 'roberta-large'

label_list = ["entailment", "not_entailment"]#, "contradiction"]
num_labels = len(label_list)

class RobertaClassificationHead(nn.Module):
    """wenpeng overwrite it so to accept matrix as input"""

    def __init__(self, bert_hidden_dim, num_labels):
        super(RobertaClassificationHead, self).__init__()
        self.dense = nn.Linear(bert_hidden_dim, bert_hidden_dim) #linearize the midek
        self.dropout = nn.Dropout(0.1) #randomly zeroes some input elements
        self.out_proj = nn.Linear(bert_hidden_dim, num_labels) #linearize the model output

    def forward(self, features):
        x = features#[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        #softmaxing
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class RobertaForSequenceClassification(nn.Module):
    def __init__(self, tagset_size, pretrain_model_dir):
        super(RobertaForSequenceClassification, self).__init__()
        self.tagset_size = tagset_size #2 from the num of labels

        self.roberta_single= RobertaModel.from_pretrained(pretrain_model_dir) 
        self.single_hidden2tag = RobertaClassificationHead(bert_hidden_dim, tagset_size) #defining our classification head

    def forward(self, input_ids, input_mask):
        outputs_single = self.roberta_single(input_ids, input_mask, None) #getting the output from roberta model
        hidden_states_single = outputs_single[1]#torch.tanh(self.hidden_layer_2(torch.tanh(self.hidden_layer_1(outputs_single[1])))) #(batch, hidden)

        score_single = self.single_hidden2tag(hidden_states_single) #(batch, tag_set) 
        return score_single

class EntailmentModel:
    def __init__(self, args):

        entailment_model_path = args.entailment_model_path
        entailment_tokenizer_path = args.entailment_tokenizer_path

        self.model = RobertaForSequenceClassification(num_labels, entailment_tokenizer_path).cuda()#.from_pretrained(pretrain_model_dir).cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(entailment_tokenizer_path)
        checkpoint = torch.load(entailment_model_path, map_location=f'cuda:0')
        #checkpoint = torch.load(entailment_model_path, map_location=torch.device('cpu'))
        self.model.load_state_dict(checkpoint, strict=False)
	#ffor calculating rogue score
        self.evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l'],
                           #max_n=2,
                           limit_length=False,
                           apply_avg=False,
                           apply_best=True,
                           alpha=0.5, # Default F1_score
                           weight_factor=1.2,
                           stemming=True)
	
    def run_entailment_prediction(self, sample: Dict):
        sample['correction_scores'] = []
        sample['rouge_scores'] = []
        sample['corrected sentences'] = sample['corrected sentences'] + [sample['input_claim']]#+ sample['g_correction']
        for correction in sample['corrected sentences']:
            current_correction_scores = []
            rouge_score = self.evaluator.get_scores([correction], [sample['input_claim']])['rouge-l']['f']
            sample['rouge_scores'].append(rouge_score)
            evidence = sample['evidence'] #if 'full_evidence' in sample else sample['evidence_text']


		
            for _, ctx in enumerate(evidence):
		
                # if is_wikipedia:
                #     ctx = ' '.join([' '.join(e.split('\t')[1:]) for e in ctx.split('\n')])
                # else:
		 
                ctx = ' '.join(ctx.split('\t'))
                ctx = ' '.join(ctx.split('\n'))
                # ctx = ' '.join([e for e in ctx.split('\n')])
		
                # dynamically determine how much input to use
                encoded_ctx = self.tokenizer.encode(ctx)[:-1] # remove [SEP]
                encoded_correction = self.tokenizer.encode(correction)[1:] # remove [CLS]
			#truncating the evidence[i] to match the input size
                encoded_ctx_truncated = encoded_ctx[:512 - 1 - len(encoded_correction) ] # - [SEP] - encoded_correction
			#generating input ids for the encoded evidence
                input_ids = torch.LongTensor(encoded_ctx_truncated + [self.tokenizer.sep_token_id] + encoded_correction).cuda().unsqueeze(0)
		
                attention_mask = torch.LongTensor([1] * len(input_ids)).cuda().unsqueeze(0)


                inputs = {'input_ids':input_ids,
                        'input_mask': attention_mask}

				
                with torch.no_grad():
                    self.model.eval()
                    logits = self.model(**inputs)
			#applying softmax to the logits output to get probabilities 
                    probs = torch.nn.Softmax(dim=1)(logits)
			#probablity for when "entailment"
                    correct_prob = probs[0][0].item()
                    current_correction_scores.append(correct_prob)
                    #using the highest entailment score
            if len(current_correction_scores):
                sample['correction_scores'].append(max(current_correction_scores))

	#using that sample with the highest DocNLI+ROGUE score
        if sample['correction_scores']:

            argmax = np.argmax(np.array(sample['correction_scores']) + np.array(sample['rouge_scores']))

            sample['final_answer'] = sample['corrected sentences'][argmax]
        else:
            sample['final_answer'] = sample['input_claim']

        return sample

class Correction:

    def __init__(self, args) -> None:
        # init all the model
        self.entailment_model = EntailmentModel(args)


        print("Finish loading models.")


    def correct(self, sample: Dict):
        sample = self.entailment_model.run_entailment_prediction(sample)


        return sample

    def batch_correct(self, samples: List[Dict]):

        return [self.correct(sample) for sample in tqdm(samples, total=len(samples))]
import argparse
parser = argparse.ArgumentParser(description="Candidate correction scoring script")

parser.add_argument("output_file", type=str, help="Path to the output file")

args = parser.parse_args()

with open(args.output_file,"rb") as sp:
    sentences=pickle.load(sp)
samples=[{}]*len(sentences)
j=0
list_dict_map=[0]*len(sentences)
for i in range(len(sentences)):
    #print(i)
    samples[i]=sentences[int(i)]
    samples[i]["corrected sentences"]=[x.strip() for x in samples[i]["corrected sentences"]]
    #samples[j]["evidence"]=[samples[j]["evidence"][0]+samples[j]["evidence"][1]]
    list_dict_map[i]=i
    samples[j]["evidence"]=[' '.join(samples[j]["evidence"])]
    #for j in range(len(samples[i]["answers"])):
    #    samples[i]["answers"][j]= samples[i]["answers"][j].strip().strip("\"").strip("}").strip()

    
from types import SimpleNamespace
import json

model_args = {
    'entailment_model_path': 'DocNLI.pretrained.RoBERTA.model.pt',
    'entailment_tokenizer_path':'roberta-large',
}
model_args = SimpleNamespace(**model_args)
correction = Correction(model_args)
outputs = correction.batch_correct(samples)
for i in range(len(sentences)):
    sentences[str(list_dict_map[i])]=outputs[i]
cs_output=args.output_file.split(".pkl")[0]+"_cs.pkl"
with open(cs_output,'wb') as eo:
    pickle.dump(sentences, eo)

