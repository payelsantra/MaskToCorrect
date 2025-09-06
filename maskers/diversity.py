from keybert import KeyBERT
import json, pickle
from tqdm import tqdm
kw_model = KeyBERT()
def map_phrases_by_edge_tokens(sentence, phrases):
    sentence_tokens = sentence.split()
    sentence_lower = [t.lower().strip(",.") for t in sentence_tokens]
    results = []

    for phrase in phrases:
        phrase_tokens = phrase.lower().split()
        if not phrase_tokens:
            results.append(None)
            continue

        first_token = phrase_tokens[0]
        last_token = phrase_tokens[-1]
        matched = None

        # Look for spans where first and last token match
        for i in range(len(sentence_tokens)):
            if sentence_lower[i] != first_token:
                continue
            for j in range(i, len(sentence_tokens)):
                if sentence_lower[j] == last_token:
                    span = " ".join(sentence_tokens[i:j+1])
                    matched = span
                    break
            if matched:
                break

        results.append(matched)
    return results
class Masker:
    def mask(self, data, args):
        for i in tqdm(range(len(data))):
            n_tokens=int(0.5*(len(data[i]["input_claim"].split(" "))))
            keywords = kw_model.extract_keywords(
            data[i]["input_claim"],
            keyphrase_ngram_range=(1, n_tokens),
            stop_words=None,
            use_mmr=True,
            diversity=0.3,
            top_n=10
            )    
            #print(keywords, n_tokens)
            keyphrases=[k[0] for k in keywords]
            #print(keyphrases)

            map_k=map_phrases_by_edge_tokens(data[i]["input_claim"], keyphrases)
            #i["masked_claim"] = i["input_claim"]
            #print(map_k)
            data[i]["masked_claim"]=[]
            for j in map_k:
                if j is not None:
                    data[i]["masked_claim"].append(data[i]["input_claim"].replace(j, "[MASK]"))
        return data


