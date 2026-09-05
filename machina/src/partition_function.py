"""
python partition_function.py -m pythia-70M

"""

import argparse
import json
import sys

import torch
from transformers import  AutoModel, AutoModelForCausalLM
from tqdm import tqdm

DEFAULT_CACHE_DIR = "./cache"

STEPS = 20
LAST_STEP = 143
MODEL_STEPS = ["step0"] + [f"step{int(i*LAST_STEP/STEPS)}000" for i in range(1, STEPS+1)]


def gptneox_get_embeddings(
        model_params
        ):
    """_summary_ gets the input and output embeddings for the given pythia model

    Args:
        model_params (_type_): _description_ model paramters 

    Returns:
        _type_: _description_ emb_in, emb_out
    """

    model = AutoModelForCausalLM.from_pretrained(**model_params)

    emb_in = model.gpt_neox.embed_in.weight
    emb_out = model.embed_out.weight

    return emb_in, emb_out


def falcon_get_embeddings(
        model_params
        ):
    """_summary_ gets the input and output embeddings for the falcon 7B model

    Args:
        model_params (_type_): _description_ model paramters 

    Returns:
        _type_: _description_ emb_in, emb_out
    """

    model = AutoModelForCausalLM.from_pretrained(**model_params)

    emb_in = None #in and out are same matrix
    emb_out = model.transformer.word_embeddings.weight

    return emb_in, emb_out


def opt_get_embeddings(
        model_params
        ):
    """_summary_ gets the input and output embeddings for the opt 7B model

    Args:
        model_params (_type_): _description_ model paramters 

    Returns:
        _type_: _description_ emb_in, emb_out
    """

    model = AutoModelForCausalLM.from_pretrained(**model_params)

    emb_in = None #in and out are same matrix
    emb_out = model.model.decoder.embed_tokens.weight

    return emb_in, emb_out 


def crystal_get_embeddings(
        model_params
        ):
    """_summary_ gets the input and output embeddings for the opt crystal coded model

    Args:
        model_params (_type_): _description_ model paramters 

    Returns:
        _type_: _description_ emb_in, emb_out
    """

    model = AutoModelForCausalLM.from_pretrained(**model_params)

    emb_in = model.transformer.wte.weight
    emb_out = model.lm_head.weight

    return emb_in, emb_out 

def amber_get_embeddings(
        model_params
        ):
    """_summary_ gets the input and output embeddings for the amber model from llm360

    Args:
        model_params (_type_): _description_ model paramters 

    Returns:
        _type_: _description_ emb_in, emb_out
    """

    model = AutoModelForCausalLM.from_pretrained(**model_params)

    emb_in = model.model.embed_tokens.weight
    emb_out = model.lm_head.weight

    return emb_in, emb_out 


MODEL_CONFIGS = {
    "pythia-70m":{
        "model_addr":"EleutherAI/pythia-70M",
        "get_emb_fn":gptneox_get_embeddings
    },
    "pythia-160m":{
        "model_addr":"EleutherAI/pythia-160M",
        "get_emb_fn":gptneox_get_embeddings
    },
    "pythia-410m":{
        "model_addr":"EleutherAI/pythia-410M",
        "get_emb_fn":gptneox_get_embeddings
    },
    "pythia-1b":{
        "model_addr":"EleutherAI/pythia-1B",
        "get_emb_fn":gptneox_get_embeddings
    },
    "pythia-1.4b":{
        "model_addr":"EleutherAI/pythia-1.4B",
        "get_emb_fn":gptneox_get_embeddings
    },
    "pythia-2.8b":{
        "model_addr":"EleutherAI/pythia-2.8B",
        "get_emb_fn":gptneox_get_embeddings
    },
    "pythia-6.9b":{
        "model_addr":"EleutherAI/pythia-6.9B",
        "get_emb_fn":gptneox_get_embeddings
    },
    "pythia-12b":{
        "model_addr":"EleutherAI/pythia-12B",
        "get_emb_fn":gptneox_get_embeddings
    },
    "gpt-neox-20b":{
        "model_addr":"EleutherAI/gpt-neox-20B",
        "get_emb_fn":gptneox_get_embeddings
    },
    "falcon-7b":{
        "model_addr":"tiiuae/falcon-7B",
        "get_emb_fn":falcon_get_embeddings
    },
    "opt-7b":{
        "model_addr":"facebook/opt-6.7B",
        "get_emb_fn":opt_get_embeddings
    },
    "crystalcoder":{
        "model_addr":"LLM360/CrystalCoder",
        "get_emb_fn":crystal_get_embeddings
    },
    "amber":{
        "model_addr":"LLM360/Amber",
        "get_emb_fn":amber_get_embeddings
    }
}


def parse_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", 
        "--model", 
        type=str, 
        required=True, 
        help=f"the model to evaluate. valid choices are {list(MODEL_CONFIGS.keys())}"
    )
    parser.add_argument(
        "-c", 
        "--cache_dir", 
        type=str, 
        required=False, 
        default=DEFAULT_CACHE_DIR,
        help="directory to cache huggingface models defaults to ./cache"
    )
    parser.add_argument(
        "-t", 
        "--training", 
        default=False, 
        action=argparse.BooleanOptionalAction, 
        help="Use the 21 training checkpoints from the paper (only works with Pythia models)"
        )

    args = parser.parse_args()

    if args.model.lower() not in MODEL_CONFIGS:
        print()
        print(f"model {args.model} not found")
        print(f"valid options are {list(MODEL_CONFIGS.keys())}")
        sys.exit()
    else:
        args.model = args.model.lower()

    return args


def compute_IW(
        W:torch.tensor
    )->float:
    """_summary_ calculate the estimated I(W) value for this set of embeddings

    Computing I(W) as per Auora et al. 2016 and Bis et al 2021

    Args:
        W (torch.tensor): _description_ embedding matrix of size |V|x<embedding size>

    Returns:
        float: _description_ the estimated I(W) value for the set of embeddings
    """	

    #get the eignevectors we will use to estimate I(W)
    WtW = torch.mm(torch.t(W), W) #a symetric matrix
    eigen_values, eigen_vectors = torch.linalg.eigh(WtW)
    X = eigen_vectors #change notation to match paper equations

    """
    columns are eigenvectors according to documentation
    https://pytorch.org/docs/stable/generated/torch.linalg.eigh.html

    since W has rows as embeddings this calculates the inner terms of the sum in 
    eq (7) of bis et al 2021 for each eignevector

    giving us |V|x<number eignevector> inner terms
    """
    c_x = torch.mm(W, eigen_vectors)  
    exp_c_x = torch.exp(c_x)

    """
    Sums the inner terms for each eignevector

    this gives us <number eignevector> sums
    """
    sum_exp_c_x = torch.sum(exp_c_x, dim=0)
    
    #find the min and max sums to estimate I(W)
    min_sum_exp_c_x = torch.min(sum_exp_c_x)
    max_sum_exp_c_x = torch.max(sum_exp_c_x)

    #compute estimated I(W) as the ratio of the min and max sums
    IW = min_sum_exp_c_x/max_sum_exp_c_x		

    return IW.item()


def prep_model_params(
        model_addr, 
        revision, 
        args
        ):
    """_summary_ prepares the model params

    Args:
        model_addr (_type_): _description_ model address
        revision (_type_, optional): _description_. Defaults to None. model revision (i.e. checkpoint)
        args (_type_, optional): _description_. command line arguments

    Returns:
        _type_: _description_ model paramters to be fed to from_pretrained loaded from HF
    """
    
    model_params = {
        "pretrained_model_name_or_path":model_addr
    }

    if args.cache_dir is not None:
        cache_dir = f"{args.cache_dir}/{model_addr}"
    else:
        cache_dir = f"{DEFAULT_CACHE_DIR}/{model_addr}"

    if revision is not None:
        model_params["revision"] = revision
        cache_dir = f"{cache_dir}/{revision}"

    model_params["cache_dir"] = cache_dir

    return model_params


def calc_isotropy(
            model_config,
            revision,
            args
        ):
    
    model_addr = model_config["model_addr"]
    get_emb_fn = model_config["get_emb_fn"]

    model_params = prep_model_params(
        model_addr, 
        revision, 
        args
        )

    emb_in, emb_out = get_emb_fn(model_params)

    results = {}
    results["IW_in"] = compute_IW(emb_in) if emb_in is not None else None
    results["IW_out"] = compute_IW(emb_out) if emb_out is not None else None

    #also calculate the I(W) when W is centered

    emb_in = emb_in - torch.mean(emb_in, dim=0).unsqueeze(0) if emb_in is not None else None
    emb_out = emb_out - torch.mean(emb_out, dim=0).unsqueeze(0) if emb_out is not None else None

    results["IW_in_centered"] = compute_IW(emb_in) if emb_in is not None else None
    results["IW_out_centered"] = compute_IW(emb_out) if emb_out is not None else None

    return results


def main():

    args = parse_arguments()

    #should we use the 21 training checkpoints or not (edit the constants if you want to view checkpoints not used in the paper)
    if args.training:
        revisions = MODEL_STEPS
    else:
        revisions = [None]

    model_config = MODEL_CONFIGS[args.model]

    results = {}
    with torch.no_grad():
        for revision in tqdm(revisions, total= len(revisions), desc="Calculating Isotropy: "):

            isotropy_results = calc_isotropy(
                model_config,
                revision,
                args
            )

            if revision is None:
                revision = ""

            results[revision] = isotropy_results

    print(json.dumps(results, indent=2)) #redirect terminal output to saveany

        
if __name__ == "__main__":

    with torch.no_grad():
        main()
