from vllm import LLM, SamplingParams
def prompt_generator(data, args):
    if args.model=="llama":
        model_chosen="TheBloke/Llama-2-70b-Chat-AWQ"
        instruction="""Your task is to correct a claim by filling in the [MASK] using the provided input evidence, ensuring that the corrected claim is supported by the evidence and only differs from the input claim in the masked positions. If the input claim is correct, do not edit it and give the input claim as output. Your output claim should be faithful to the provided evidence and should not deviate much from the input claim.

        Please use the most relevant evidence to correct the claim. The corrected claim shouldn't contain any [MASK].

        Input: 
        Claim: [sentence]"""
        if args.dataset=="scifact":
            
            instruction="""Your task is to correct a claim by filling in the [MASK] using the provided input evidence, ensuring that the corrected claim is supported by the evidence and only differs from the input claim in the masked positions. If the input claim is correct, do not edit it and give the input claim as output. Your output claim should be faithful to the provided evidence and should not deviate much from the input claim.

Please use the most relevant evidence to correct the claim. The corrected claim shouldn't contain any [MASK].

Input: 
Claim: [sentence]"""
    elif args.model=="qwen":
        instruction="""Your task is to correct a claim by filling in the [MASK] using the provided input evidence, ensuring that the corrected claim is supported by the evidence and only differs from the input claim in the masked positions. If the input claim is correct, do not edit it and give the input claim as output. Your output claim should be faithful to the provided evidence and should not deviate much from the input claim.

    Please use the most relevant evidence from the given evidence, to correct the claim. The corrected claim shouldn't contain any [MASK].

    Input: 
    Claim: [sentence]"""
        if args.dataset=="scifact":
            instruction="""Your task is to correct a claim by filling in the [MASK] using the provided input evidence, ensuring that the corrected claim is supported by the evidence and only differs from the input claim in the masked positions. If the input claim is correct, do not edit it and give the input claim as output. Your output claim should be faithful to the provided evidence and should not deviate much from the input claim.

Please use the most relevant evidence to correct the claim. The corrected claim shouldn't contain any [MASK].

Input: 
Claim: [sentence]"""
            
    instruction_zero="""Your task is to correct a claim by filling in the [MASK], ensuring that the corrected claim is supported by the evidence and only differs from the input claim in the masked positions. If the input claim is correct, do not edit it and give the input claim as output. Your output claim should be faithful to the provided evidence and should not deviate much from the input claim.

    Please use the most relevant evidence to correct the claim. The corrected claim shouldn't contain any [MASK].

    Input: 
    Claim: [sentence]"""
    if args.dataset=="scifact":
        instuction_zero="""Your task is to correct a claim by filling in the [MASK] using the provided input evidence, ensuring that the corrected claim is supported by the evidence and only differs from the input claim in the masked positions. If the input claim is correct, do not edit it and give the input claim as output. Your output claim should be faithful to related evidence and should not deviate much from the input claim.

Please use the most relevant evidence to correct the claim. The corrected claim shouldn't contain any [MASK].

Input: 
Claim: [sentence]"""
    
    for i in range(1,args.shots+1):
        instruction=instruction+"""\nEvidence"""+str(i)+""": [document]"""
    instruction=instruction+"""Masked Claim: [masked sentence]

    Output: 
    Correction: [sentence]
    """
    j=0
    use_zeroshot=0
    prompt_dict={}
    o_id={}
    for key in data.keys():
        try:
            e=data[key]['evidence'][0:args.shots]
        except:
            e=data[key]['evidence']
            use_zeroshot=1
        claim=data[key]['input_claim']
        
        for masked_claim in data[key]["masked_claim"]:
            #print(i) 
            if use_zeroshot==0:
                prompt_answer=instruction+"""\nInput:\nClaim: """+claim
                #print(len(e))           
                for e_id in range(1,len(e)+1):
                    prompt_answer=prompt_answer+"""\nEvidence"""+str(e_id)+e[e_id-1]
                prompt_answer=prompt_answer+"""\nMasked Claim: """+masked_claim
            else:
                
                prompt_answer=instruction_zero+"""\nInput:\nClaim: """+claim+"""\nMasked Claim: """+masked_claim
            prompt_dict[j]=prompt_answer+"\n\nOutput:"
            o_id[j]=key
            j=j+1
            #print(key)
    #print(len(prompt_dict))
    return prompt_dict,o_id
class Corrector:
    def correct(self, data, llm, args):
        
        prompt_dict,o_id=prompt_generator(data, args)
        prompt_list=list(prompt_dict.values())
        if args.model=="llama": 
            sampling_params = SamplingParams(temperature=0.05, top_p=0.5, min_tokens=10, max_tokens=100, stop=["}"], include_stop_str_in_output=True)
        elif args.model=="qwen":
            sampling_params = SamplingParams(temperature=0.12, top_p=0.5, seed=9, min_tokens=10, max_tokens=100, stop=["}"], include_stop_str_in_output=True)
        outputs = llm.generate(prompt_list, sampling_params)
#s=dict(zip(list(prompt.keys()),outputs))
        import ast
        s=dict(zip(list(prompt_dict.keys()),outputs))
        for i in s:
          generated=s[i].outputs[0].text
          #print(generated)
          try:
              f=generated.split(":")[1]
              try:
                f=f.split("\n")[0]
              except:
                f=f.strip()
          except:
              f=generated
          #print(f)
          s[i]=f
        #for i in range(0,3882):
        #    ms[i]["final_answer"]=s[i].strip(" ").strip("\"")

        #print(ms)
        for i in data.keys():
            data[i]["corrected sentences"]=[]
        for i in s:
                data[o_id[i]]["corrected sentences"].append(s[i])
        return data
        
            



