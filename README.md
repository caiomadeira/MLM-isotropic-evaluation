Machina full code: https://github.com/anemily-machina/isotropy_transformers


Evidence of RoBERTa x mmBERT scalar:
```
# I(alpha * W) mesma matriz porem multiplicada a um escalar
_, eigenvectors = np.linalg.eigh(W.T @ W)
projections = W @ eigenvectors 

for alpha in [1, 2.49, 5]:
    scaled_projs = alpha * projections 
    maxi = scaled_projs.max(axis=0)
    log_Z = maxi + np.log(np.exp(scaled_projs - maxi).sum(axis=0))  # log-sum-exp
    I = np.exp(log_Z.min() - log_Z.max())
    print(f"alpha = {alpha:5.2f}   norma media = {alpha * norms.mean():.3f}   I(alpha*W) = {I:.4f}")
```
