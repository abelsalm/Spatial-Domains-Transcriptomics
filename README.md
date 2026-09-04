# GASTON-Consensus

This repository contains **GASTON-Mix** (`src/gastonmix/`) and an experimental successor, **GASTON-Consensus**, implemented in `gaston_improvement_support.py` and exercised in `gaston_mieux.ipynb`.

If you already know GASTON-Mix, the Mix MoE formula is unchanged. What changes is the **expert**: the scalar isodepth bottleneck is replaced by a **gene-specific, low-rank expansion in a shared spatial dictionary**, optionally evaluated in a **single learned orthonormal frame**. Domains are defined as regions that share **boundaries**, not a common 1-D gradient.

This README is the specification of that extension. It is not a drop-in replacement of the Mix CLI.

---

## 0. Notation

| Symbol | Meaning |
|---|---|
| \(i=1,\ldots,N\) | spots / cells |
| \(\mathbf s_i\in\mathbb R^2\) | spatial coordinates, after affine map to \([-1,1]^2\) (each axis independently) |
| \(g=1,\ldots,G\) | genes, or PCs treated as genes |
| \(p=1,\ldots,P\) | domains / experts |
| \(\mathbf y_i\in\mathbb R^G\) | observed target (counts, Gaussian expression, or whitened PCs) |
| \(G_p(\mathbf s)\) | gate probability of domain \(p\) at \(\mathbf s\) |
| \(E_p(\mathbf s)\in\mathbb R^G\) | expert \(p\)'s predicted expression field |
| \(\Phi(\mathbf s)\in\mathbb R^B\) | spatial dictionary (basis) |
| \(R\) | rank of gene–profile factorization inside a domain |
| \(\theta\) | shared axis angle (Consensus, `learned_axes=True`) |

Coordinates used by the model are

\[
s^{(a)}_i = 2\cdot\frac{s^{\mathrm{raw},a}_i-\min_j s^{\mathrm{raw},a}_j}{\max_j s^{\mathrm{raw},a}_j-\min_j s^{\mathrm{raw},a}_j}-1,
\qquad a\in\{x,y\}.
\]

This **anisotropically** stretches a non-square bounding box. The learned angle lives in this normalized plane, not necessarily in physical microns.

---

## 1. What GASTON-Mix already does (recap)

### 1.1 GASTON (single domain)

One neural **isodepth** \(d=S(\mathbf s)\in\mathbb R\) and one neural **expression map** \(A:\mathbb R\to\mathbb R^G\),

\[
\hat{\mathbf y}(\mathbf s)=A\bigl(S(\mathbf s)\bigr).
\]

Every gene is a 1-D function of the same latent coordinate. Layered tissue with one dominant gradient is the intended geometry.

### 1.2 GASTON-Mix (several domains, still isodepth experts)

A spatial **gating network** \(G(\mathbf s)\in\Delta^{P-1}\) and \(P\) independent GASTON experts:

\[
\hat{\mathbf y}_i=\sum_{p=1}^{P} G_p(\mathbf s_i)\,E_p(\mathbf s_i),\qquad
E_p(\mathbf s)=A_p\bigl(S_p(\mathbf s)\bigr).
\]

- \(S_p:\mathbb R^2\to\mathbb R\) is an MLP (optionally with random Fourier / positional features).
- \(A_p:\mathbb R\to\mathbb R^G\) is an MLP. The expert is a **composition through a scalar**.
- Default routing is **hard top-1**: after softmax, keep the largest coordinate and renormalize (with \(k=1\) the selected weight is \(1\)). Training therefore gives the gate little reconstruction gradient except through the discrete choice.
- Load-balancing uses a squared coefficient of variation on expert usage.
- Reconstruction is MSE on a transformed expression matrix (or a Poisson-style NLL in an alternative branch).

Post-hoc, Mix fits **piecewise-linear Poisson** models of each gene against the learned isodepth inside each domain (`segmented_fit.py`). That analysis assumes the 1-D coordinate is already the right geometry.

**What Mix does not do:** let genes inside a domain follow *different* spatial mechanisms (a linear gene and a sigmoidal gene and a constant gene) unless those mechanisms are all functions of one scalar \(S_p\).

---

## 2. Scientific claim of Consensus

Keep the Mix mixture

\[
\hat{\mathbf y}_i=\sum_{p=1}^{P} G_p(\mathbf s_i)\,E_p(\mathbf s_i).
\]

Drop the isodepth bottleneck. A **domain** is a spatial region in which genes have their own spatial behaviours (constant, linear, sigmoidal, exponential), sharing **boundaries** more than they share a gradient.

A valid boundary is a **consensus** of gene-behaviour changes: enough genes must jump, but most genes are allowed to stay the same. That is the opposite of “every gene is a function of isodepth, and a domain is a piece of that 1-D axis.”

Consequences:

1. Experts must be more flexible than \(A_p\circ S_p\), otherwise Consensus collapses to Mix.
2. Experts must **not** be unconstrained neural fields: a neural field can put a discontinuity *inside* a domain, so the gate is no longer identifiable as “the place where behaviour changes.”
3. Coefficient differences \(\theta_{p}-\theta_{q}\) are only meaningful if \(\Phi\) is the **same dictionary** in every domain. Hence a **fixed** (or jointly rotated) basis, not a learned per-domain MLP dictionary.

---

## 3. Architecture (Consensus)

Implemented as `ConsensusSpatialMoE`.

### 3.1 Mixture (unchanged algebra, different \(E_p\))

\[
\hat{\mathbf y}_i=\sum_{p=1}^{P} G_p(\mathbf s_i)\,E_p(\mathbf s_i).
\]

During training the gate is **soft**. After fitting, hard labels are

\[
z_i=\arg\max_p\,\ell_p(\mathbf s_i),
\]

where \(\ell(\mathbf s)\) are the gate logits (not the tempered softmax).

### 3.2 Gate (the only neural network)

MLP with \(\tanh\) hidden units, last layer linear of width \(P\):

\[
\ell(\mathbf s)=W_L\,\tanh(W_{L-1}\cdots\tanh(W_1\,\psi(\mathbf s))),
\qquad
G(\mathbf s)=\mathrm{softmax}\bigl(\ell(\mathbf s)/\tau\bigr).
\]

\(\tau>0\) is an annealed temperature (see §6). As \(\tau\to 0\), \(G\) approaches a one-hot map. Softmax during training is required: Mix-style top-1 makes \(G_{i,z_i}=1\), so reconstruction does not tell neighbouring logits how to move.

**Gate features \(\psi(\mathbf s)\).** By default \(\psi(\mathbf s)=\mathbf s\in\mathbb R^2\). Optionally, **Fourier features for the gate only**:

\[
\psi(\mathbf s)=\Bigl[
\mathbf s,\;
\sin(\pi f\,\mathbf d_k^\top\mathbf s),\;
\cos(\pi f\,\mathbf d_k^\top\mathbf s)
\Bigr]_{f\in\mathcal F_{\mathrm{gate}},\;k=1,\ldots,4},
\]

with fixed directions

\[
\mathbf d\in\bigl\{(1,0),\;(0,1),\;(2^{-1/2},2^{-1/2}),\;(2^{-1/2},-2^{-1/2})\bigr\}.
\]

If \(\mathcal F_{\mathrm{gate}}=(1,2,3)\), the gate input has dimension \(2+2\cdot 3\cdot 4=26\). These frequencies **are not** expert basis functions. They only help the partition represent curved or disconnected regions. Set `gate_fourier_frequencies=()` to disable them.

The gate **never** sees the learned tissue angle \(\theta\). Orientation is a property of the expression dictionary.

### 3.3 Spatial dictionary \(\Phi\)

Evaluate on \(\mathbf s\in[-1,1]^2\). Let \(\mathbf D=(\mathbf u_1,\ldots,\mathbf u_M)\in\mathbb R^{2\times M}\) be unit directions and \(z_m=\mathbf u_m^\top\mathbf s\).

**Raw columns** (before standardization):

| Family | Formula | Notes |
|---|---|---|
| constant | \(1\) | never standardized |
| linear | \(z_1,\;z_2\) | only the first two axes (the orthonormal pair) |
| sigmoid | \(\sigma\bigl(k(z_m-c)\bigr)\) | \(k\in\mathcal K\), \(c\in\mathcal C\), all directions |
| two-sided decay | \(\exp(-\lambda\,\mathrm{softplus}(\pm z_m))\) | both orientations; one is not a reparameterization of the other |
| Fourier (optional) | \(\sin(\pi f z_m),\;\cos(\pi f z_m)\) | phase via the pair; **omitted in the compact dictionary** |

**Standardization.** Let \(\Phi^{\mathrm{raw}}_{2:B}\) be the non-constant columns. On a fixed \(41\times 41\) grid of \([-1,1]^2\), compute mean \(\boldsymbol\mu\) and std \(\boldsymbol\sigma\) (detached w.r.t. \(\mathbf D\)). Then

\[
\Phi_1=1,\qquad
\Phi_{2:B}=\frac{\Phi^{\mathrm{raw}}_{2:B}-\boldsymbol\mu}{\boldsymbol\sigma}.
\]

Penalties on coefficients are then comparable across families. When \(\mathbf D\) is a learned rotation, \(\boldsymbol\mu,\boldsymbol\sigma\) are recomputed on that grid for the current axes (still detached, so scale does not absorb rotation).

**Two dictionaries.**

*Overcomplete (default `BasisConfig`).*  
\(\mathcal K=(4,10)\), \(\mathcal C=(-0.35,0,0.35)\), \(\lambda\in\{1.5,4\}\), \(f\in\{0.5,1\}\), four directions (axes + diagonals). \(B=59\). Useful as a smoother; named families are only weakly identifiable.

*Compact (`compact_basis_config`, used with learned axes).*  
\(\mathcal K=(6)\), \(\mathcal C=(-0.4,0,0.4)\), \(\lambda=2\), no Fourier, no diagonals, two axes only.

\[
B=1+2+2\cdot(3+2)=13.
\]

Families: constant, linear \(u,v\), three sigmoids and two decays on \(u\), same on \(v\). This is the biologically motivated bank: compartment baseline, monotone gradient, switch, source–sink decay.

### 3.4 Low-rank gene-specific experts

For domain \(p\) and gene \(g\),

\[
E_{pg}(\mathbf s)=\beta_{pg}+\sum_{b=1}^{B-1}\Theta_{pgb}\,\Phi_{b+1}(\mathbf s),
\]

with a rank-\(R\) factorization of the non-constant coefficients

\[
\Theta_{p}\in\mathbb R^{G\times(B-1)},\qquad
\Theta_{pg\cdot}=U_{pg\cdot}\,V_p,
\]

i.e.

\[
\Theta_p=U_p V_p,\qquad
U_p\in\mathbb R^{G\times R},\;
V_p\in\mathbb R^{R\times(B-1)}.
\]

Parameters: `intercepts` \(\beta\in\mathbb R^{P\times G}\), `gene_loadings` \(U\in\mathbb R^{P\times G\times R}\), `profile_atoms` \(V\in\mathbb R^{P\times R\times(B-1)}\).

- \(V_p\)'s rows are **profile archetypes** (mixtures of dictionary columns) shared by all genes in domain \(p\).
- \(U_{pg}\) is the gene-specific mixture of those archetypes.
- \(R=1\): one spatial shape per domain; genes differ by scale and intercept.
- \(R\to\min(G,B-1)\): essentially unconstrained per-gene coefficients.
- Typical: \(R=3\) (toys), \(R=8\) (cortex PCs).

This is **not** “all genes share one curve.” It is “a domain has few spatial programs, and every gene is a combination of them.”

There is **no** neural network inside \(E_p\).

### 3.5 Shared learned angle (current last step)

The model cannot know how a slice was placed on the slide. It can learn two orthonormal tissue axes and evaluate the compact dictionary in that frame.

One scalar \(\theta\in\mathbb R\), **shared by every expert**:

\[
\mathbf u(\theta)=(\cos\theta,\sin\theta),\qquad
\mathbf v(\theta)=(-\sin\theta,\cos\theta).
\]

Then \(\mathbf D(\theta)=[\mathbf u(\theta),\;\mathbf v(\theta)]\) and \(\Phi_\theta(\mathbf s)=\Phi(\mathbf s;\mathbf D(\theta))\). Linear terms still span \(\mathbb R^2\) for any \(\theta\); sigmoids and decays do not. Group lasso is applied **per family and per axis** (e.g. `sigmoid:u` vs `sigmoid:v`), so a true \(u\)-sigmoid is cheaper if \(\theta\) aligns than if both banks stay active.

**Identifiability of \(\theta\).** Undirected unlabeled frames are identified only modulo \(\pi/2\): swapping axes or flipping a sign is not an error. The reported residual is

\[
\mathrm{err}(\hat\theta,\theta^\star)
=\min_{k\in\mathbb Z}\bigl|(\hat\theta-\theta^\star+\tfrac{\pi}{4})\bmod\tfrac{\pi}{2}-\tfrac{\pi}{4}\bigr|
\in\bigl[0,\tfrac{\pi}{4}\bigr].
\]

**Why not a different \(\theta_p\) per expert.** A slice has one anatomy. Coefficient fusion and consensus compare \(\Theta_p\) and \(\Theta_q\) on the **same** \(\Phi\). Per-expert rotations mix “the gene changed” with “the coordinate system changed,” and recreate a soft isodepth. A hierarchical residual \(\theta_p=\theta+\delta_p\) was implemented and **rejected**: it is not a nested relaxation of the shared model (gauge \(\theta\leftarrow\theta+c,\;\delta\leftarrow\delta-c\)), fusion becomes invalid as soon as \(\delta\neq 0\), and a large \(\lambda_\delta\) does not recover `learned_axes` without offsets (extra Adam parameters, zero gradient of \(1-\cos\delta\) at \(0\), grad-clip domination). Do not use it.

**Envelope gradient.** Joint Adam on \((\theta,U,V)\) lets \(U,V\) absorb a wrong frame, so \(\partial\mathcal L/\partial\theta\approx 0\). Periodically ridge-refitting experts at the current \(\theta\) (`refit_experts_every`) puts coefficients near \(\arg\min_{U,V}\mathcal L_{\mathrm{data}}\), and the remaining \(\partial\mathcal L/\partial\theta\) is the envelope derivative. This is an EM-style inner step, not a second network.

Initialize \(\theta\) from the leading SVD axis of centered coordinates when the cloud is elongated (`init_learned_axes_from_coords`). A square cloud has no unique PCA axis; leave \(\theta=0\) and let expression rotate it.

---

## 4. Losses

All terms are implemented in `consensus_moe_loss`. Let \(W\) be `LossWeights`. The scalar objective is

\[
\mathcal L=\mathcal L_{\mathrm{data}}
+\lambda_{\mathrm{TV}}\mathcal L_{\mathrm{TV}}
+\lambda_H\mathcal L_H
+\lambda_B\mathcal L_{\mathrm{balance}}
+\lambda_R\|\Theta\|_F^2\text{-style ridge}
+\lambda_F\mathcal L_{\mathrm{family}}
+\lambda_{\mathrm{fuse}}\mathcal L_{\mathrm{fuse}}
+\lambda_C\mathcal L_{\mathrm{consensus}}.
\]

Code multiplies each named term by the corresponding field of `LossWeights` (defaults in parentheses below). Reconstruction is **not** multiplied by an extra weight.

### 4.1 Reconstruction \(\mathcal L_{\mathrm{data}}\)

**Homoscedastic Gaussian (toys without known noise):**

\[
\mathcal L_{\mathrm{data}}=\frac{1}{NG}\sum_{i,g}(\hat y_{ig}-y_{ig})^2.
\]

**Heteroscedastic Gaussian** (known \(\sigma_{ig}^2\), as in the noisy checkerboard):

\[
\mathcal L_{\mathrm{data}}=\frac{1}{NG}\sum_{i,g}\frac12\left(
\frac{(\hat y_{ig}-y_{ig})^2}{\sigma_{ig}^2}+\log\sigma_{ig}^2
\right).
\]

**Negative binomial (UMI counts).** Expert output is treated as log relative mean. With library size \(L_i\) (default: total UMI),

\[
\log\mu_{ig}=\hat y_{ig}+\log\frac{L_i}{\bar L},\qquad
\alpha_g=\mathrm{softplus}(\xi_g),
\]

and NB2 NLL averaged over entries (`negative_binomial_nll`). \(\xi\in\mathbb R^G\) is `log_inverse_dispersion`. This is the intended likelihood for raw counts; Gaussian MSE is for continuous toys and whitened PCs.

### 4.2 Spatial graph

Radius graph (`spatial_edges`): radius \(=1.05\times\) smallest positive pairwise distance (4-neighbour on a grid).  
kNN graph (`spatial_knn_edges`): symmetric \(k\)-NN, used for irregular spots.

Undirected edges \((i,j)\in\mathcal E\). Write \(G_i=G(\mathbf s_i)\).

### 4.3 Boundary TV \(\mathcal L_{\mathrm{TV}}\) (`spatial_boundary`, default \(0.03\))

Disagreement of neighbouring gates:

\[
b_{ij}=1-\langle G_i,G_j\rangle,\qquad
\mathcal L_{\mathrm{TV}}=\frac{1}{|\mathcal E|}\sum_{(i,j)\in\mathcal E}b_{ij}.
\]

If both spots have the same one-hot domain, \(b_{ij}=0\). This is a soft length penalty on the cut. It must stay **weak**: a checkerboard has many true boundaries.

### 4.4 Gate entropy \(\mathcal L_H\) (`gate_entropy`, default \(0.01\))

\[
\mathcal L_H=\frac{1}{N}\sum_i\sum_p -G_{ip}\log G_{ip}.
\]

Together with decreasing \(\tau\), this sharpens assignments.

### 4.5 Weak usage balance \(\mathcal L_{\mathrm{balance}}\) (`gate_balance`, default \(0.02\))

\[
\bar G_p=\frac1N\sum_i G_{ip},\qquad
\mathcal L_{\mathrm{balance}}=\frac1P\sum_p\Bigl(\bar G_p-\frac1P\Bigr)^2.
\]

Prevents expert collapse. It must stay weak: biological domains need not have equal area. This is milder than Mix's CV load-balancing.

### 4.6 Coefficient ridge (`coefficient_ridge`, default \(2\cdot 10^{-4}\))

\[
\mathcal L_R=\mathrm{mean}(\Theta^{\odot 2}).
\]

Stabilizes the factorization; intercepts are not included.

### 4.7 Group lasso by family **and axis** (`basis_group_lasso`, default \(2\cdot 10^{-3}\))

Partition non-constant columns into groups such as `linear:u`, `sigmoid:v`, `exponential:u`. For each group \(\mathcal G\),

\[
\mathcal L_{\mathrm{family}}=\sum_{\mathcal G}
\frac{1}{PG}\sum_{p,g}
\sqrt{\sum_{b\in\mathcal G}\Theta_{pgb}^2+\varepsilon}.
\]

A gene that uses sigmoids on **both** axes pays more than a gene that uses only `sigmoid:u`. That is the inductive bias that makes a shared \(\theta\) identifiable.

### 4.8 Fused genes \(\mathcal L_{\mathrm{fuse}}\) (`fused_genes`, default \(4\cdot 10^{-3}\))

For each unordered pair of domains \((p,q)\) and each edge \((i,j)\), crossing mass

\[
m_{ij}^{pq}=G_{ip}G_{jq}+G_{iq}G_{jp}.
\]

If total crossing mass of \((p,q)\) is negligible, skip. Otherwise L1 on coefficient and intercept differences, weighted by mass, then normalized by total crossing mass:

\[
\mathcal L_{\mathrm{fuse}}
=\frac{
\sum_{p<q}\Bigl(\sum_{(i,j)}m_{ij}^{pq}\Bigr)
\cdot\mathrm{mean}_{g}\bigl(\|\Theta_{pg\cdot}-\Theta_{qg\cdot}\|_1+\lvert\beta_{pg}-\beta_{qg}\rvert\bigr)
}{
\sum_{p<q}\sum_{(i,j)}m_{ij}^{pq}}.
\]

(Implementation uses mean absolute value over the basis index for each gene, then mean over genes.) Many genes are encouraged to be **exactly unchanged** across a touching pair.

This term is only valid if \(\Phi\) is the same in \(p\) and \(q\).

### 4.9 Consensus hinge \(\mathcal L_{\mathrm{consensus}}\) (`boundary_consensus`, default \(0.08\))

At the edge midpoint \(\mathbf m_{ij}=(\mathbf s_i+\mathbf s_j)/2\), evaluate both experts. For gene \(g\),

\[
J_{ij,g}^{pq}=\frac{\bigl|E_{pg}(\mathbf m_{ij})-E_{qg}(\mathbf m_{ij})\bigr|}{\mathrm{std}_i(y_{\cdot g})}.
\]

Soft “this gene supports the boundary”:

\[
c_{ij,g}=\sigma\bigl((J_{ij,g}^{pq}-t)/T\bigr),
\qquad
\bar c_{ij}=\frac1G\sum_g c_{ij,g},
\]

with defaults \(t=0.6\), \(T=0.15\). Require at least a fraction \(\rho=0.25\) of genes to change:

\[
\mathcal L_{\mathrm{consensus}}
=\frac{
\sum_{p<q}\sum_{(i,j)} m_{ij}^{pq}\,\bigl[\rho-\bar c_{ij}\bigr]_+^2
}{
\sum_{p<q}\sum_{(i,j)}m_{ij}^{pq}}.
\]

One noisy marker cannot open a domain; fusion still allows most genes to stay put. Reconstruction decides *which* genes change.

---

## 5. Why this combination

| Term | Role |
|---|---|
| data | fit expression / PCs / counts |
| TV | few, spatially coherent cuts |
| entropy + \(\tau\downarrow\) | discrete domains at the end |
| balance | no unused experts |
| ridge + group lasso | simple, axis-aligned named behaviours |
| fusion | most genes may ignore a boundary |
| consensus hinge | enough genes must jump to justify a boundary |

Without fusion, every gene wants its own partition. Without the hinge, fusion erases all boundaries. Without a **fixed shared** \(\Phi\), fusion and the hinge compare incomparable coefficients.

---

## 6. Optimization

`fit_consensus_moe`, preceded by two warm starts.

### 6.1 Boundary-blind initialization

**Do not** use true domains, tile IDs, or boundary coordinates.

`graph_expression_initialization`:

1. Optionally residualize a global spatial trend (ridge on a small `SpatialBasis`); skip this when the domain split *is* that trend (half-planes).
2. Average features over a spatial \(k\)-NN (including self).
3. Standardize columns; \(k\)-means into \(P\) clusters.

The only spatial prior is local averaging.

Then `pretrain_gate`: Adam on gate weights only, cross-entropy to those labels (optional extra TV on the gate).

Then `initialize_experts_from_ridge`: for each domain, weighted ridge of \(\mathbf y\) on \(\Phi_\theta(\mathbf s)\) (intercept unpenalized), SVD of the non-constant coefficient matrix, keep rank \(R\):

\[
\Theta=U\Sigma V^\top,\qquad
U_R\leftarrow U_{:R}\Sigma_{:R}^{1/2},\quad
V_R\leftarrow\Sigma_{:R}^{1/2}V_{:R}.
\]

### 6.2 Joint training

Adam, three parameter groups:

| Group | Default contents | Typical LR |
|---|---|---|
| experts | \(\beta,U,V,\xi\) | \(10^{-3}\) |
| gate | gate MLP | \(2\cdot 10^{-4}\) |
| axis | \(\theta\) if `learned_axes` | \(5\cdot 10^{-2}\) |

- `freeze_gate_epochs`: zero gate grads for the first \(T_{\mathrm{freeze}}\) epochs (experts, then \(\theta\), adapt first).
- `freeze_axes_epochs`: same for \(\theta\).
- Gradient clip: \(\|\nabla\|_2\le 5\).
- Temperature schedule, \(t=0,\ldots,T-1\):

\[
\tau(t)=\tau_0\left(\frac{\tau_T}{\tau_0}\right)^{t/(T-1)}.
\]

Typical: \(\tau_0=0.8\), \(\tau_T=0.15\) or \(0.2\).

- `refit_experts_every=k`: every \(k\) epochs, hard-assign spots and re-run ridge SVD at the current \(\theta\). Required for a usable \(\partial\mathcal L/\partial\theta\). Skip a refit if any domain has fewer than \(B\) spots.

Progress is a tqdm bar (reconstruction, total loss, \(\tau\)).

### 6.3 Hard labels

After training, `hard_domains` uses \(\arg\max\) of **logits**, not sampled softmax.

---

## 7. Real-data protocol (MERFISH)

Fitting \(G\sim 250\) genes directly is unnecessary for domain discovery. Protocol in the notebook:

1. Load one section (`load_merfish_slice`).
2. \(\log(1+y)\), spatial \(k\)-NN smooth of log-expression (dropout suppression).
3. Standardize genes; PCA to \(K\) components (typically \(K=32\)); **whiten** each PC to unit variance.
4. Fit Consensus on the \(N\times K\) whitened PC matrix (Gaussian MSE). Coordinates are **not** concatenated into PCA: geometry must come from expression.
5. Inverse map: unwhiten \(\to\) PCA inverse \(\to\) unscale \(\to\) predicted log-expression (`SpatialExpressionPCA.inverse`). Marker genes (Cux2, Rorb, …) are reconstituted; they are not training targets by name.

Cell-type layers are plotting annotations only.

---

## 8. What was tried and discarded

| Idea | Why not |
|---|---|
| Unconstrained neural field per domain × gene | Internal discontinuities; overparameterized; boundaries unidentifiable |
| Full per-gene coefficient vector, no rank | Interpretable but \(P\cdot G\cdot B\) unrelated profiles |
| Overcomplete Fourier+diagonal dictionary as the *scientific* model | Too rich; names are not mechanisms; kept only as a smoother |
| Mix hard top-1 during Consensus training | Gate gets no useful reconstruction gradient |
| Tile-average initialization | Leaks synthetic tile geometry |
| Per-expert angle \(\theta_p\) | Soft isodepth; fusion incomparable |
| Hierarchical \(\theta_p=\theta+\delta_p\) | Not nested in optimization; fusion invalid; high \(\lambda_\delta\) \(\neq\) shared model |

Current recommended expert: **compact 13-function dictionary + one shared \(\theta\) + rank-\(R\) loadings**.

---

## 9. Code map

| Path | Role |
|---|---|
| `src/gastonmix/run_moe_script.py` | Published GASTON-Mix (`GASTON_MoE`, top-1, isodepth MLPs) |
| `src/gastonmix/segmented_fit.py` | Mix post-hoc piecewise Poisson vs isodepth |
| `gaston_improvement_support.py` | Consensus model, losses, init, toys, MERFISH/PCA helpers |
| `gaston_mieux.ipynb` | Design notes and experiments (conda env `gaston-mix`) |

Consensus is **not** wired into the Mix CLI.

### Minimal Consensus fit

```python
from gaston_improvement_support import (
    ConsensusSpatialMoE, LossWeights, compact_basis_config,
    graph_expression_initialization, pretrain_gate,
    initialize_experts_from_ridge, fit_consensus_moe,
    spatial_knn_edges, init_learned_axes_from_coords,
)

model = ConsensusSpatialMoE(
    n_genes=G, n_domains=P, rank=8,
    gate_hidden=(64, 64),
    gate_fourier_frequencies=(1, 2, 3),  # partition only
    basis_config=compact_basis_config(), # 13 functions, no Fourier experts
    learned_axes=True,
)
init_learned_axes_from_coords(model, coords)  # skip on square clouds
labels = graph_expression_initialization(coords, expression, n_domains=P)
pretrain_gate(model, coords, labels)
initialize_experts_from_ridge(model, coords, expression, labels)
fit_consensus_moe(
    model, coords, expression,
    edges=spatial_knn_edges(coords),
    refit_experts_every=1,               # needed for θ
    weights=LossWeights(),
)
z = model.hard_domains(coords)
```

---

## Installation (GASTON-Mix package)

```
conda env create -f environment.yml
conda activate gaston-mix
pip install -e .
```

CPU or GPU. Consensus extras are imported from the repo root module, not from `gastonmix`.

## Citations

GASTON-Mix: [biorXiv 2025.01.31.635955](https://www.biorxiv.org/content/10.1101/2025.01.31.635955v1).

```
@article{Chitra2025,
  author = {Chitra, Uthsav and Dan, Shu and Krienen, Fenna and Raphael, Benjamin J.},
  title = {GASTON-Mix: a unified model of spatial gradients and domains using spatial mixture-of-experts},
  elocation-id = {2025.01.31.635955},
  year = {2025},
  doi = {10.1101/2025.01.31.635955},
  journal = {bioRxiv}
}
```

GASTON-Consensus is unpublished prototype code in this fork.
