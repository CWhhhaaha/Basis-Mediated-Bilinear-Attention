# Methods and implementation notes

## Shared notation

Let $X\in\mathbb{R}^{T\times d}$, with $H$ heads and ambient head width $d_k$. BMB introduces a layer-local basis $B\in\mathbb{R}^{d\times r}$ and shared latent $Z=XB$. The basis and head-specific interactions are learned jointly during training; they are not obtained by decomposing a trained dense checkpoint.

### Native BMB

$$
L_h=ZM_hZ^\top/\sqrt r,\qquad
A_h=\mathrm{softmax}(L_h),\qquad O_h=A_hV_h.
$$

The head matrix is parameterized by a shared core and centered head residuals. In `gpt2_attention_variants.py` this is computed as

```python
centered = self.head_residual - self.head_residual.mean(dim=0, keepdim=True)
head_matrix = self.core.unsqueeze(0) / float(self.n_head) + centered
```

The direct native implementation explicitly forms `[batch, heads, T, T]` logits, then applies scaling, masking, softmax, dropout, and the PV multiplication. This is faithful but not a fused-kernel implementation: materializing and revisiting the quadratic logits is the main systems cost.

### BMB-UV

$$
M_h=U_h^Q{U_h^K}^\top,\qquad Q_h=ZU_h^Q,\qquad K_h=ZU_h^K.
$$

Explicit Q/K tensors make BMB-UV compatible with standard attention interfaces. Calling PyTorch SDPA does not itself guarantee a particular backend: backend selection depends on PyTorch, device, dtype, head widths, and other constraints. FlashAttention is a specific fused implementation, not a synonym for SDPA.

When $r=s=64$, UV factors can represent an arbitrary $64\times64$ head matrix. Native BMB and UV64 should therefore not be distinguished by a claim that native BMB has strictly greater expressive capacity; their parameterization, optimization, and execution differ.

## GQA + BMB-UV and explicit cache

For $H_q$ query heads and $H_{kv}$ KV heads, let $\gamma(h)$ map a query head to its KV group:

$$
Z=XB,\quad Q_h=ZU_h^Q,\quad K_j=ZU_j^K,\quad V_j=XW_j^V,
$$
$$
O_h=\mathrm{softmax}\!\left(\frac{Q_hK_{\gamma(h)}^\top}{\sqrt s}\right)V_{\gamma(h)}.
$$

The implementation applies RoPE to Q and K before caching K. Cache shapes per layer are

$$
K_{\text{cache}}:[N,H_{kv},T,s],\qquad
V_{\text{cache}}:[N,H_{kv},T,d_v].
$$

Thus GQA reduces $H_{kv}$, while BMB-UV can additionally make $s<d_k$. In XL, $d_k=d_v=64$ and $s=32$, so K+V changes from 128 to 96 elements per KV head per token: a 25% reduction. If $s=d_k$, there is no additional key-width saving.

### Why the reported experiments do not cache $Z$

Caching the shared latent would store $r+H_{kv}d_v$ elements per token and layer, compared with $H_{kv}(s+d_v)$ for explicit K/V. It saves memory only when $r<H_{kv}s$. XL uses $r=128$, $H_{kv}=2$, and $s=32$, so the condition is false: the shared latent would be wider than all explicit K tensors combined.

The explicit cache also preserves the standard position-encoding and attention workflow. A latent-cache variant would need validated position-transform handling, per-KV-head K reconstruction, and compatible cache/kernel layouts. It becomes a reasonable direction when $r<H_{kv}s$ and those systems requirements are met.

## Eight compared QK pathways

| Method | Structural definition | Important accounting detail |
|---|---|---|
| MHA | Independent Q/K per head | Standard reference |
| FullShr | Fully shared Q and K projections | Shared weights; Q and K projections still counted separately |
| PartShr | Shared block plus private Q/K components | `partshr728` is layer-level: total Q/K width 768, shared 728, private 40 |
| LowRank | Low-rank independent projections | `lowrank32` uses rank 32 |
| BBᵀ | Shared $BB^\top$ score structure | One common score matrix per layer |
| BMB-UV32 | Shared basis plus UV factors | $r=s=32$ |
| BMB-UV64 | Shared basis plus UV factors | $r=s=64$ |
| BMB64 | Shared basis plus native head matrices | $r=64$ |

PartShr's 728 does **not** mean sharing 728 dimensions inside each 64-dimensional head.

## Initialization

- BMB-UV basis matrices and each head factor use independent Xavier-uniform initialization.
- Native BMB basis, core, and head residuals use Xavier-uniform initialization; three-dimensional head parameters are initialized head by head.
- The Llama-style FineWeb model preserves these BMB-specific initializations after framework initialization.
- Residual output projections use standard deviation $0.02/\sqrt{2L}$, where $L$ is the number of layers.
- The FineWeb non-QK stack uses tied GPT-2 embeddings, RoPE, parameter-free RMSNorm, SwiGLU, and no biases.

## Parameter and FLOP accounting

`attention_accounting.py` is the accounting entry point for the eight-method GPT-2/vision comparisons. It counts multiplication and addition as two FLOPs and excludes bias operations, softmax, masks, dropout, nonlinearities, and reshape operations. These definitions follow the paper; profiler counts may differ under other conventions.

For the FineWeb GQA+BMB-UV models, exact parameter counts are checked against

$$
P=Vd+L\left(dr+(H_q+H_{kv})rs+dH_{kv}d_v+(H_qd_v)d+3dd_{\mathrm{ff}}\right).
$$

This yields 110,592,000 parameters for Small and 1,328,793,600 for XL. In particular, `o_proj` has input width `Hq*dv`; XL uses `48*64=3072`, not the model width 1600.
