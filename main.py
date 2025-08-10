import argparse
import sys
import importlib
import json, pickle
import subprocess

def run_in_other_env(output_file, retriever, shots):
    script="correction_scoring.py"
    env_name="scorer"
    cmd = [
        "conda", "run", "-n", env_name,
        "python", script,
        output_file,
        retriever,
        str(shots)
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
    
    if args.model=="ensemble":
	    evidence_file=["bm25.pkl","monot.pkl","rm3.pkl","colbert.pkl","contriver.pkl"]
	    for ef in evidence_file:
		    with open(ef,"rb") as e_file:
		    	input_evidence=pickle.load(e_file)
		    for i in range(len(input_claims)):  
		    	data[i]=input_claims[i]
		    	data[i]["evidence"]=input_evidence[i][0:args.shots]
		    masked_output = masker.mask(data, args)
		    corrector_module=importlib.import_module("corrector.corrected_candidates")
		    corrector=corrector_module.Corrector()
		    corrected_candidates=corrector.correct(masked_output, args)
		    output_ret="ensemble_"+ef
		    with open(output_ret,"wb") as ret_file:
		    	pickle.dump(corrected_candidates,ret_file)
		    run_in_other_env(output_ret)

		    
		    
    else:
    	ef=args.retriever+".pkl"
    	with open(ef,"rb") as e_file:
    		input_evidence=pickle.load(e_file)
    	for i in range(len(input_claims)):
    		data[i]=input_claims[i]
    		data[i]["evidence"]=input_evidence[i][0:args.shots]
    	masked_output = masker.mask(data,args)
    	corrector_module=importlib.import_module("corrector.corrected_candidates")
    	corrector=corrector_module.Corrector()
    	corrected_candidates=corrector.correct(masked_output, args)
    	with open(args.output_file,"wb") as ret_file:
    	    	pickle.dump(corrected_candidates,ret_file)
    print("Masking completed.")
 
