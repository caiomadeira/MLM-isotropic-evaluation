
import argparse
import random
import sys
import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

NUM_EMBS = 12800
EMB_SIZE = 512

BATCH_SIZE = 128
PARTIAL_SUM_MAX_SIZE = 1280


def get_dataloader(embs):

    dataloader = DataLoader(
        embs,
        batch_size=BATCH_SIZE
    )

    return dataloader


def fast_cosine_batched(embs):

    start_time = time.time()

    dataloader = get_dataloader(embs)

    partial_emb_sums = []
    partial_emb_sum = torch.zeros((EMB_SIZE))
    partial_sum_size = 0

    for batch_embs in tqdm(dataloader, total=len(dataloader)):
        
        emb_sums = torch.sum(batch_embs, dim=0)  #sum normalized vectors
        
        partial_emb_sum += emb_sums
        partial_sum_size += len(batch_embs)

        #storing partial sums to avoid floating point errors (doesn't really matter for this demo but needs to be done if you are using a large corpus)
        if partial_sum_size >= PARTIAL_SUM_MAX_SIZE:
            partial_sum_size = 0
            partial_emb_sums.append(partial_emb_sum)
            partial_emb_sum = partial_emb_sum*0
   
    partial_emb_sums.append(partial_emb_sum)

    emb_sums = torch.stack(partial_emb_sums) #combine partial sums
    emb_sums = torch.sum(emb_sums, dim=0)
    sum_cosine = torch.sum(emb_sums*emb_sums) #compute cosine of all vectors with all vectors by computing dot product of sum vector    

    avg_cosine = (sum_cosine - NUM_EMBS)/NUM_EMBS/(NUM_EMBS - 1)

    total_time = time.time() - start_time

    print()
    print(f"Fast Cosine batched took {total_time:.2f} seconds")

    return avg_cosine


def slow_cosine_batched(embs):

    start_time = time.time()

    dataloader = get_dataloader(embs)

    partial_cos_sums = []

    partial_cos_sum = 0.0
    partial_sum_size = 0

    for b_i, batch_embs_a in tqdm(enumerate(dataloader), total=len(dataloader)):

        for b_j, batch_embs_b in enumerate(dataloader):

            #only compute the scores once per pair
            if b_j > b_i:
                break

            cos_scores = batch_embs_a @ batch_embs_b.t()

            #if A and B are the same embddings we only want the lower triangular values
            if b_j == b_i:
                cos_scores = torch.tril(cos_scores, diagonal=-1)

            cos_sum = torch.sum(cos_scores)
            partial_cos_sum += cos_sum
            
            partial_sum_size += len(batch_embs_b)
            #storing partial sums to avoid floating point errors (doesn't really matter for this demo but needs to be done if you are using a large corpus)
            if partial_sum_size >= PARTIAL_SUM_MAX_SIZE:
                partial_sum_size = 0
                partial_cos_sums.append(partial_cos_sum)
                partial_cos_sum = 0.0

    partial_cos_sums.append(partial_cos_sum)

    cos_sums = sum(partial_cos_sums)

    avg_cos = 2*cos_sums/NUM_EMBS/(NUM_EMBS-1) #this formula is different as it's the sum of unqiue pairs

    total_time = time.time() - start_time

    print()
    print(f"Slow Cosine batched took {total_time:.2f} seconds")

    return avg_cos


def fast_cosine(embs):

    start_time = time.time()

    emb_sums = torch.sum(embs, dim=0)  #sum normalized vectors
    sum_cosine = torch.sum(emb_sums*emb_sums) #compute cosine of all vectors with all vectors by computing dot product of sum vector

    avg_cosine = (sum_cosine - NUM_EMBS)/NUM_EMBS/(NUM_EMBS - 1)

    total_time = time.time() - start_time

    print()
    print(f"Fast Cosine took {total_time:.2f} seconds")

    return avg_cosine


def slow_cosine(embs):

    start_time = time.time()

    cosines = embs@embs.t()

    sum_cosine = torch.sum(cosines)

    avg_cosine = (sum_cosine - NUM_EMBS)/NUM_EMBS/(NUM_EMBS - 1)

    total_time = time.time() - start_time

    print()
    print(f"Slow cosine took {total_time:.2f} seconds")

    return avg_cosine


def compare_cosine_batched(embs):

    print()
    print("Computing Fast Cosine..")

    fast_sum = fast_cosine_batched(embs)
    print(f"Fast Cosine value: {fast_sum:.6f}")

    print()
    print("Computing Slow Cosine..")
    print("NOTE: Batches will get slower as they progress as we are looping j<i for all i")
   
    slow_sum = slow_cosine_batched(embs)
    print(f"Slow Cosine value: {slow_sum:.6f}")


def compare_cosines(embs):

    print()
    print("Computing Fast Cosine..")

    fast_sum = fast_cosine(embs)
    print(f"Fast Cosine value: {fast_sum:.6f}")

    print()
    print("Computing Slow Cosine..")
   
    slow_sum = slow_cosine(embs)
    print(f"Slow Cosine value: {slow_sum:.6f}")


def parse_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-ne", 
        "--number_emb", 
        type=int, 
        default=128000,
        help=f"The number of random embeddings to generate"
    )
    parser.add_argument(
        "-es", 
        "--emb_size", 
        type=int, 
        default=512,
        help=f"The size of the embeddings"
    )
    parser.add_argument(
        "-i", 
        "--isotropic", 
        default=False, 
        action=argparse.BooleanOptionalAction, 
        help="Isotropic: emb values in N(0,1), Not Isotropic: emb values in uniform [0,1)"
        )
    
    parser.add_argument(
        "-ss", 
        "--sample_size", 
        type=int, 
        default=10000,
        help=f"Number of samples to estimate true cosine similarity (will still compute full cosine)"
    )
    
    parser.add_argument(
        "-b", 
        "--batched", 
        default=True, 
        action=argparse.BooleanOptionalAction, 
        help="Use batched methods (may OOM if not using batched methods)"
        )
    parser.add_argument(
        "-bs", 
        "--batch_size", 
        type=int, 
        default=None,
        help=f"Batch size if using batched methods, default:128"
    )
    parser.add_argument(
        "-psm", 
        "--partial_sum_max", 
        type=int, 
        default=None,
        help=f"Number of values in each partial sum (to prevent floating point errors), if using batched methods. default: 1280"
    )


    parser.add_argument(
        "-s", 
        "--seed", 
        type=int, 
        default=4321,
        help=f"Random seed for consistency"
    )


    args = parser.parse_args()

    if args.number_emb < 2:
        print()
        print(f"There must be at least 2 embeddings. chosen size {args.number_emb}")
        sys.exit()

    global NUM_EMBS
    NUM_EMBS = args.number_emb

    if args.emb_size < 1:
        print()
        print(f"Embedding size must be at least 2. chosen size {args.emb_size}")
        sys.exit()

    global EMB_SIZE
    EMB_SIZE = args.emb_size

    if args.sample_size < 1:
        print()
        print(f"Sample size must be at least 1. chosen size {args.sample_size}")
        sys.exit()

    if args.sample_size >= args.number_emb:
        print()
        print(f"Warning: sample size {args.sample_size} greater than number of embeddings {args.number_emb}")

    if not args.batched:

        if (args.batch_size is not None):
            print()
            print("Warning: batch size set but not set to use batching (-b)")

        if (args.partial_sum_max is not None):
            print()
            print("Warning: partial sum max size set but not set to use batching (-b)")

    else:

        if (args.batch_size is not None):

            if args.batch_size < 0:
                print()
                print(f"Batch size must be greater than 0. given {args.batch_size}")
                sys.exit()

            global BATCH_SIZE
            BATCH_SIZE = args.batch_size 

        if (args.partial_sum_max is not None):

            if args.partial_sum_max < 0:
                print()
                print(f"partial sum max size must be greater than 0. given {args.partial_sum_max}")
                sys.exit()

            global PARTIAL_SUM_MAX_SIZE
            PARTIAL_SUM_MAX_SIZE = args.partial_sum_max    

    return args


def main():

    args = parse_arguments()

    #set seed for reproducibility
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print()
    print("Creating Embeddings...")

    #get a set of random embeddings
    if args.isotropic:
        embs = torch.randn((args.number_emb, args.emb_size)) #normaldistribution N(0,1)
    else:
        embs = torch.rand((args.number_emb, args.emb_size)) #uniform distribution on interval [0,1)

    print()
    print("Normalizing Embeddings..")

    #normalize the embedding (both methods want to do this so it's only done once, as then dot product = cosine similarity)
    embs = torch.nn.functional.normalize(embs, dim=1)


    print()
    print("--------------------------------------------------------------------------------")
    print()
    print("Quick test with <= 10 emeddings as full test will take a while")

    #because the other one might take a looooong time....
    compare_cosines(embs[:10])

    if args.number_emb > args.sample_size:
        print()
        print("--------------------------------------------------------------------------------")
        print()
        print(f"Computing cosine estimate with Random Sample of {args.sample_size} embeddings (to see quality of estimate)")
        random_emb_indexes = random.sample(range(0, len(embs)), k=args.sample_size)
        random_embs = embs[random_emb_indexes]
        compare_cosines(random_embs)
    

    print()
    print("--------------------------------------------------------------------------------")
    print()
    print()
    print(f"Running full test on {NUM_EMBS} embedings")

    if args.batched:
        compare_cosine_batched(embs)
    else:
        compare_cosines(embs)


if __name__ == "__main__":

    with torch.no_grad():
        main()
