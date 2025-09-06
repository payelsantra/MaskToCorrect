import argparse
import sys
import importlib
import json, pickle
import subprocess
from collections import Counter
from vllm import LLM, SamplingParams

def pick_most_common_from_sublist(sublist):
    if not sublist:
        return None
    sublist = [x for x in sublist if x != '']

    # Lowercase mapping to original
    lowercase_to_original = {}
    lowercased_sublist = []
    for sentence in sublist:
        lower_sentence = sentence.lower()
        lowercased_sublist.append(lower_sentence)
        if lower_sentence not in lowercase_to_original:
            lowercase_to_original[lower_sentence] = sentence

    counts = Counter(lowercased_sublist)
    most_common = counts.most_common()

    if len(most_common) == 1 or (most_common[0][1] > most_common[1][1]):
        # Clear winner
        return lowercase_to_original[most_common[0][0]]
    else:
        # Tie
        if len(sublist) > 3:
            return sublist[3]
        else:
            return sublist[0]

def run_in_other_env(output_file):
    script="correction_scoring.py"
    env_name="scorer"
    cmd = [
        "conda", "run", "-n", env_name,
        "python", script,
        output_file
    ]
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print(f"Error running {script} in env {env_name}")
        sys.exit(result.returncode)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="M2C")

    parser.add_argument("--input_file", type=str, required=True,
                        help="Path to the input file")
    parser.add_argument("--retriever", type=str, required=True,
                        choices=["gold", "bm25", "monot5", "rm3", "contriver", "colbert", "ensemble"],
                        help="Retriever type to use")
    parser.add_argument("--shots", type=int, default=0,
                        help="Number of few-shot evidence sentences to include (for hm)")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Where to write the output")
    parser.add_argument("--model", type=str, required=True,
                        help="Name or path of the model to use")
    parser.add_argument("--masker", type=str, required=True,
                        choices=["heuristic", "random", "diversity", "entity"],
                        help="Masking strategy to apply")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["fever", "scifact"],
                        help="Dataset used")
    args = parser.parse_args()
    print(f"Using masker: {args.masker}")
   
    with open(args.input_file, 'r') as f:
         input_claims=[json.loads(line) for line in f]
    try:
        masker_module = importlib.import_module(f"maskers.{args.masker}")
        masker = masker_module.Masker()
    except ModuleNotFoundError:
        print(f"Error: Masker '{args.masker}' not found.")
        sys.exit(1)

    data = {}
    if args.model=="llama":
            model_chosen="TheBloke/Llama-2-70b-Chat-AWQ"
    elif args.model=="qwen":
            model_chosen="Qwen/Qwen2.5-32B-Instruct-AWQ"
        
    llm = LLM(model=model_chosen)
    if args.retriever=="ensemble":
        evidence_file=["bm25.pkl", "monot5.pkl", "rm3.pkl", "colbert.pkl", "contriver.pkl"]
        for ef in evidence_file:
            with open(ef,"rb") as e_file:
                input_evidence=pickle.load(e_file)
            for i in range(len(input_claims)):  
                data[i]=input_claims[i]
                data[i]["evidence"]=input_evidence[i][0:args.shots]
            masked_output = masker.mask(data, args)
            corrector_module=importlib.import_module("corrector.corrected_candidates")
            corrector=corrector_module.Corrector()
            corrected_candidates=corrector.correct(masked_output, llm, args)
            output_ret="ensemble_"+ef
            with open(output_ret,"wb") as ret_file:
                pickle.dump(corrected_candidates,ret_file)
        for ef in evidence_file:
            output_ret="ensemble_"+ef
            run_in_other_env(output_ret)
        import pickle
        with open('ensemble_contriver_cs.pkl', 'rb') as f:
            contriver = pickle.load(f)
        with open('ensemble_bm25_cs.pkl', 'rb') as f:
            bm25 = pickle.load(f)
        with open('ensemble_colbert_cs.pkl', 'rb') as f:
            colbert = pickle.load(f)
        with open('ensemble_monot5_cs.pkl', 'rb') as f:
            monot5 = pickle.load(f)
        with open('ensemble_rm3_cs.pkl', 'rb') as f:
            rm3 = pickle.load(f)
        final_data={}
        for i in range(0,len(bm25)):
          final_data[i]={}
          final_data[i]["candidate"]=[contriver[i]["final_answer"].strip(),bm25[i]["final_answer"].strip(),colbert[i]["final_answer"].strip(),monot5[i]["final_answer"].strip(),rm3[i]["final_answer"].strip()]
          final_data[i]["gt_claim"]=contriver[i]["gt_claim"]
          final_data[i]["input_claim"]=contriver[i]["input_claim"]   
          final_data[i]["final_answer"]=pick_most_common_from_sublist(final_data[i]["candidate"])
        with open(args.output_file,"wb") as ret_file:
            pickle.dump(final_data,ret_file)
    else:
    	ef=args.retriever+".pkl"
    	with open(ef,"rb") as e_file:
    		input_evidence=pickle.load(e_file)
    	for i in range(len(input_claims)):
    		data[i]=input_claims[i]
    		data[i]["evidence"]=input_evidence[i][0:args.shots]
    	masked_output = masker.mask(data,args)
    	print("Correction")
    	corrector_module=importlib.import_module("corrector.corrected_candidates")
    	corrector=corrector_module.Corrector()
    	corrected_candidates=corrector.correct(masked_output, llm, args)
    	with open(args.output_file,"wb") as ret_file:
    	    	pickle.dump(corrected_candidates,ret_file)
    	run_in_other_env(args.output_file)
    print("Masking completed.")
 


