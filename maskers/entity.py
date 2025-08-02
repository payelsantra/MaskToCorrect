from typing import  Dict
import spacy
import stanza


def get_phrases(tree, label):
    if tree.is_leaf():
        return []
    results = [] 
    for child in tree.children:
        results += get_phrases(child, label)
    
    
    if tree.label == label:
        return [' '.join(tree.leaf_labels())] + results
    else:
        return results
class Masker:
    def mask(self, data, args):
        if args.dataset=="scifact":
            nlp = spacy.load('en_core_sci_md')
            
        else:
            nlp = spacy.load('en_core_web_lg')
        stanza_nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency')
        candidate_answers={}
        for i in tqdm(range(len(data))):
	    	doc = nlp(data[i]['input_claim'])
		stanza_doc =stanza_nlp(data[i]['input_claim'])
		
		ents = [ent.text for sent in doc.sents for ent in sent.noun_chunks] 
		ents += [ent.text  for sent in doc.sents for ent in sent.ents]
		ents += [phrase for sent in stanza_doc.sentences for phrase in get_phrases(sent.constituency, 'NP')]
		ents += [phrase for sent in stanza_doc.sentences for phrase in get_phrases(sent.constituency, 'VP')]
		ents += [word.text for sent in stanza_doc.sentences for word in sent.words if word.upos in ['VERB','ADV','ADJ','NOUN']]
		
		negations = [word for word in ['not','never'] if word in data[i]['input_claim']]

		middle = []
		start_match = ''
		end_match = ''
		for ent in ents:
		    if data[i]['input_claim'].startswith(ent) and len(ent) > len(start_match):
		        start_match = ent
		    if data[i]['input_claim'].endswith(ent+'.') and len(ent) > len(end_match):
		        end_match = ent
		
		
		if len(start_match) > 0 and len(end_match) > 0:
		    
		    middle.append(data[i]['input_claim'][len(start_match):-len(end_match)-1].strip())
		    
		candidate_answers[i] = list(set(ents + negations + middle))
		if "" in candidate_answers[i]:
        		candidate_answers[i].remove("")
		data[i]["masked_claim"]=[data[i]["input_claim"].replace(x,"[MASK]") for x in candidate_answers[i]]
   

        return data

