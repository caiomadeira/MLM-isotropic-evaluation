from safetensors import safe_open
from transformers import AutoModelForMaskedLM
import numpy as np
from plots import *
import torch
import sys, os, datetime

class _Tee:
    def __init__(self, *outs):
        self.outs = outs
    def write(self, text):
        for o in self.outs:
            o.write(text)
            o.flush()
    def flush(self):
        for o in self.outs:
            o.flush()


try:
    import machina.src.partition_function as pf
    import machina.src.cosine_improvement as ci
    MACHINA_CODE = True
    print("partition_function.compute_IW: ", pf.compute_IW.__module__)
    print("cosine_improvement.fast_cosine: ", ci.fast_cosine.__module__)
except ImportError as e:
    MACHINA_CODE = False
    print("error:", e)


MODERN_MODELS = {
    "modernbert_base": {
        "path": "/home/caiomadeira/modernbert/model.safetensors",
        "model_id": "answerdotai/ModernBERT-base"
    },

    "modernbert_large": {
        "path": "/home/caiomadeira/modernbert_large/model.safetensors",
        "model_id": "answerdotai/ModernBERT-large"
    },

    "mmbert_base": {
        "path": "/home/caiomadeira/mmbert/model.safetensors",
        "model_id": "jhu-clsp/mmBERT-base"
    },

    "neobert_base": {
        "path": "/home/caiomadeira/neobert/model.safetensors",
        "model_id": "chandar-lab/NeoBERT"
    },

    "eurobert_610m": {
        "path": "/home/caiomadeira/eurobert/model.safetensors",
        "model_id": "EuroBERT/EuroBERT-610m"
    }
}

OLD_MODELS = {
    "bert_uncased": {
        "path": "/home/caiomadeira/bert_uncased/model.safetensors",
        "model_id": "google-bert/bert-base-uncased"
    },

    "bert_cased": {
        "path": "/home/caiomadeira/bert_cased/model.safetensors",
        "model_id": "google-bert/bert-base-cased"
    },

    "roberta": {
        "path": "/home/caiomadeira/roberta/model.safetensors",
        "model_id": "FacebookAI/roberta-base"
    }
}

EMB_TENSOR_NAMES = (
    "model.embeddings.tok_embeddings.weight", # ModernBERT / mmBERT
    "embeddings.tok_embeddings.weight", # ModernBERT / mmBERT
    "encoder.weight", # NeoBERT
    "embeddings.word_embeddings.weight", # BERT / RoBERTa
)

def mean_cosine_np(W, chunk=20000):
    n = W.shape[0]
    s = np.zeros(W.shape[1])
    for i in range(0, n, chunk):
        B = W[i:i+chunk]
        s += (B / np.linalg.norm(B, axis=1, keepdims=True)).sum(axis=0)
    return float((-n + s @ s) / (n*n - n))

def mean_cosine(W):
    U = W / np.linalg.norm(W, axis=1, keepdims=True)
    n = U.shape[0]
    s = U.sum(axis=0)
    return float((-n + s @ s) / (n * n - n))

def compute_both_IW(W):
    W = np.asarray(W, dtype=np.float64)
    print(W)
    _, X = np.linalg.eigh(W.T @ W)
    S = W @ X

    mp = S.max(axis=0)
    logZp = mp + np.log(np.exp(S - mp).sum(axis=0))

    mn = -S.min(axis=0)                              
    logZn = mn + np.log(np.exp(-S - mn).sum(axis=0))

    iw1 = float(np.exp(logZp.min() - logZp.max()))
    all = np.concatenate([logZp, logZn])
    iw2 = float(np.exp(all.min() - all.max()))
    return iw1, iw2

def isotropic_reference(W, seed=0):
    rng = np.random.default_rng(seed)
    R = rng.normal(size=W.shape)
    R /= np.linalg.norm(R, axis=1, keepdims=True)
    R *= np.linalg.norm(W, axis=1, keepdims=True)
    return R

def load_emb_matrix(path, tensor_name):
    with safe_open(path, framework="np") as f:
        keys = list(f.keys())
        if tensor_name not in keys:
            print(f"error o tensor {tensor_name} nao foi encontrado")
        W = f.get_tensor(tensor_name)
        return np.asarray(W, dtype=np.float64)

def print_preliminar_results(name, path):
    print("================")
    print(name)
    print("================")
    W = load_emb_matrix(path)
    V, d = W.shape
    norms = np.linalg.norm(W, axis=1)
    print(f"shape (Vxd) (V lines for d columns): {V}x{d}")
    print(f"V (vocabulary): {V}")
    print(f"d (dim): {d}")
    print(f"V/d (proportion between size of vocab and dim. QUantify how the vocab is larger than vetorial space): {V}x{d}")
    print(f"mean norm: {V}x{d}")

def compute_centroid_norm(W):
    centroid = W.mean(axis=0)
    print(f"centroid norm: {np.linalg.norm(centroid):.4f}")
    return centroid

def get_tensor_name(path):
    with safe_open(path, framework="pt") as f:
        keys = list(f.keys())
        for k in keys:
            print(k, f.get_slice(k).get_shape())
        for emb in EMB_TENSOR_NAMES:
            if emb in keys:
                return emb
    return None

def preliminar_report(raw_W, Wc, Wref):
    print("="*40)
    print(f"I(raw W)={raw_W:.4f}")
    print(f"I(Wc)={Wc:.4f}")
    print(f"I(Wref) reference={Wref:.4f}")
    print(f"ratio I(raw_W)/I(Wref)={raw_W/Wref:.4f} (baseline)")
    print(f"ratio I(Wc)/I(Wref)={Wc/Wref:.4f}")
    print("="*40)

def my_exp_report(name, tensor, W, iw_raw, iw_centroid, iw_ref):
    V, d = W.shape
    norms = np.linalg.norm(W, axis=1)
    nm  = norms.mean()
    mu  = W.mean(axis=0)
    centroid_norm = float(np.linalg.norm(mu))

    relative_displacement = centroid_norm / nm
    ang  = np.degrees(np.arccos(np.clip(relative_displacement, -1, 1)))
    cos  = mean_cosine_np(W)
    print("="*74)
    print(f" {name} | {tensor} | {V} x {d}")
    print("="*74)
    print("\n=== EMBEDDING \"CLOUD\" SHAPE ===\n")
    print(f"rows mean norm={nm:8.4f}  (dp {norms.std():.4f})")
    print(f"mean norm of vector ||w_bar|| = {centroid_norm:8.4f}")
    print(f"relative displacement ||w_bar||/mean norm={relative_displacement:8.4f}")
    print(f"mean angle between one token and mean vector: {ang:8.1f} degrees")
    print(f"mean cosine between tokens pairs = {cos:8.4f}")
    print("\n=== PARTITION FUNCTION (I(W)) ===\n")
    print(f"Z(c) = sum exp(<c,wi>) about the {V} tokens. c travesers the {d} eigenvectors of W^T W.\n")
    print(f"    {'':44s}{'c = +v':>10s}{'c = +v e -v':>14s}")
    print(f"    {'I(W) on the original matrix':44s}{iw_raw[0]:10.4f}{iw_raw[1]:14.4f}")
    print(f"    {'I(W) on matrix - mean vector':44s}{iw_centroid[0]:10.4f}{iw_centroid[1]:14.4f}")
    print(f"    {'I(W) with noise and same norms':44s}{iw_ref[0]:10.4f}{iw_ref[1]:14.4f}")
    print(f"    {'predict value by exp(-k*||w_bar||)':44s}"f"{np.exp(-centroid_norm):10.4f}{np.exp(-2*centroid_norm):14.4f}")

def machina_exp_report(W_np, name, raw_W1, Wc1, sample=10000, seed=4321):
    V, d = W_np.shape
    print(V, d)
    W64 = torch.from_numpy(np.ascontiguousarray(W_np, dtype=np.float64))
    W32 = W64.float()
    print("=" * 74)
    print(f"{name} | {V}x{d}")
    print("=" * 74)
    print("partition_function.compute_IW")
    iw32 = pf.compute_IW(W32)
    iw64 = pf.compute_IW(W64)

    print(f"I(W), matrix in float32: {iw32:.6f}")
    print(f"I(W), matrix in float64: {iw64:.6f}")
    print(f"difference between them: {abs(iw32-iw64):.2e}")
    print(f"compute_both_IW, 1 signal: {raw_W1:.6f}")
    print(f"dif for compute_both_IW: {abs(iw64-raw_W1):.2e}")

    Wc64 = W64 - W64.mean(dim=0, keepdim=True)
    print(f"\nI(W centered), float64: {pf.compute_IW(Wc64):.6f}")
    print(f"my Wc1: {Wc1:.6f}")

    print("\n Machina cosine_improvement.fast_cosine\n")
    ci.NUM_EMBS = V
    ci.EMB_SIZE = d
    U = torch.nn.functional.normalize(W32, dim=1)
    cos_full = float(ci.fast_cosine(U))
    print(f"mean cossine, {V} tokens : {cos_full:.6f}")
    g = torch.Generator().manual_seed(seed)
    
    idx = torch.randperm(V, generator=g)[:min(sample, V)]
    Us = U[idx].contiguous()
    ci.NUM_EMBS = Us.shape[0]
    cos_f = float(ci.fast_cosine(Us))
    cos_s = float(ci.slow_cosine(Us))
    ci.NUM_EMBS = V
    print(f"\nconference in a sample of {Us.shape[0]} tokens:")
    print(f"fast_cosine (O(n)): {cos_f:.6f}")
    print(f"slow_cosine (O(n^2)): {cos_s:.6f}")
    print(f"difference: {abs(cos_f-cos_s):.2e}")
    mu = W_np.mean(axis=0)
    relative_displacement = np.linalg.norm(mu) / np.linalg.norm(W_np, axis=1).mean()
    print(f"mean cosine: {cos_full:.4f}")
    print(f"relative displacement^2 : {relative_displacement**2:.4f}") 
    ang = np.degrees(np.arccos(np.clip(cos_full, -1, 1)))
    print(f"mean angle between two random tokens: {ang:.1f} degrees")
    print("=" * 74)
    return dict(modelo=name, IW_machina_fp32=iw32, IW_machina_fp64=iw64, cos_medio=cos_full, cos_amostra_fast=cos_f, cos_amostra_slow=cos_s)

def compute_output_matrix(model_id):
   m = AutoModelForMaskedLM.from_pretrained(model_id, trust_remote_code=True)
   state_dict = m.state_dict()
   print("k:", [k for k in state_dict if "decoder" in k or "encoder.weight" in k])
   W_out = state_dict["decoder.weight"].detach().cpu().numpy().astype(np.float64)
   b_out = state_dict["decoder.bias"].detach().cpu().numpy().astype(np.float64)
   assert W_out.shape[0] == W.shape[0], f"transpose: {W_out.shape}"
   assert not np.allclose(W, W_out), "are tied besides config"
   print("input", W.shape, " output", W_out.shape, "equals?", np.allclose(W, W_out))
   result = my_exp_report(model_id, "decoder.weight", W_out, compute_both_IW(W_out),
                        compute_both_IW(W_out - W_out.mean(axis=0)), compute_both_IW(isotropic_reference(W_out)))
   print("corr(b_j, ||w_j||) =", np.corrcoef(b_out, np.linalg.norm(W_out, axis=1))[0, 1])
   return result

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    #os.makedirs("logs", exist_ok=True)

    for k, v in MODERN_MODELS.items():
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = os.path.join("results", f"{k}_run_{timestamp}.log")
        _log = open(log_filename, "w", encoding="utf-8")
        sys.stdout = _Tee(sys.__stdout__, _log)
        sys.stderr = _Tee(sys.__stderr__, _log)

        path = v["path"]
        tensor_name = get_tensor_name(path)
        if tensor_name is None:
            print(f"tensor not found skiping")
            continue
        print(tensor_name)
        print(f"================\n{k}================\n")
        W = load_emb_matrix(path, tensor_name)
        V, d = W.shape
        norms = np.linalg.norm(W, axis=1)
        print(f"shape (Vxd) (V lines for d columns): {V}x{d}")
        print(f"V (vocabulary): {V}")
        print(f"d (dim): {d}")
        print(f"V/d (proportion between size of vocab and dim. Quantify how the vocab is larger than vetorial space): {V/d:.1f}")
        print(f"mean norm: {norms.mean():.4f}  (sd {norms.std():.4f})")
        centroid = compute_centroid_norm(W)

        print("measure I(W) in raw matrix")
        raw_W1, raw_W2 = compute_both_IW(W)
        print(f"I(W) = {raw_W1:.4f}")
        print(f"I(W) 2 = {raw_W2:.4f}")

        print("measuring I(W) for mean centered matrix (Bis et al. 2021)")
        Wc = W - centroid
        Wc1, Wc2 = compute_both_IW(Wc)

        print(f"I(W centered) = {Wc1:.4f}")
        print(f"I(W centered) 2 = {Wc2:.4f}")
        print("=========================================")
        print("isotropic reference -> same shape and norm")

        ref1, ref2 = compute_both_IW(isotropic_reference(W))
        print(f"I(W reference) = {ref1:.4f}")
        print(f"I(W reference) 2 = {ref2:.4f}")
        print("=========================================")
        print("plots")
        plot_2d_projection(W, f"{k} input")
        plot_2d_projection(Wc, f"{k} input centered 1")
        plot_hexbin_projection2(W, f"{k} input")
        plot_2d_projection(Wc, f"{k} input centered 1")
        print("=========================================")
        print("====== MY REPORT CODE ============")
        my_report = my_exp_report(k, tensor_name, W,(raw_W1, raw_W2), (Wc1, Wc2), (ref1, ref2))
        print(my_report)
        print("====== MACHINA ORIGINAL CODE ============")
        machina_report = machina_exp_report(W, k, raw_W1, Wc1) if MACHINA_CODE else None
        print(machina_report)

        if "neobert" in k.lower():
            compute_output_matrix(v["model_id"])