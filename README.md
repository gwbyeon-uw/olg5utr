# olg5utr

Design synthetic 5' UTR DNA sequences that support **overlapping genes (OLGs)** — two proteins encoded from the same DNA stretch but read from different start codons. The system optimizes sequences to maximize **mRNA translation efficiency** (MRL — Mean Ribosome Load) for *both* reading frames simultaneously, while preserving a required amino acid sequence in the overlapping region.

## Architecture: Three Components

### 1. Optimus 5' Prime Model

**What**: A CNN that predicts MRL (translation efficiency) from a 5' UTR DNA sequence.

**Origin**: PyTorch reimplementation of [Sample et al.'s Optimus 5-Prime](https://github.com/pjsample/human_5utr_modeling), originally in Keras.

**Architecture**:
- Input: one-hot DNA `(B, 4, 100)`
- 3 × Conv1d layers (120 filters, kernel 8, `same` padding) with ReLU + dropout
- Flatten → Linear(120×100, 40) → ReLU → Dropout(0.2) → Linear(40, 1)
- **Multi-head modification**: shared conv backbone with separate dense heads per dataset (`egfp_unmod`, `egfp_pseudo`, `egfp_m1pseudo`, `mcherry_unmod`)

**Training details** (`optimus_retrain.ipynb`):
- Data: MPRA datasets (CSV files with UTR sequences + ribosome load measurements + read counts)
- Preprocessing: quantile cutoff by read count, quantile normalization of RL binned by read count, z-score scaling, deduplication by k-mer priority, k-mer distance filtering between train/val/test splits
- Loss: MSE on z-scored MRL
- Optimizer: AdamW (lr=0.001, weight_decay=0.01), grad clip 0.5

### 2. Differentiable Translator

**What**: A learned CNN that mimics biological codon-to-amino-acid translation, but is **differentiable** so gradients can flow through it during sequence optimization.

**Why not just use the genetic code lookup?** Because argmax over a lookup table has zero gradient. This network learns the same mapping but provides gradients via straight-through estimation.

**Architecture**:
- Input: one-hot DNA `(B, 4, L)`
- Conv1d(4, 1024, kernel=3, stride=1) — extracts "codon features" from every 3-nt window
- 3 frame shifts (+0, +1, +2) to get all reading frames
- Conv1d(1024, 21, kernel=1, stride=3) — maps codon features → 21 amino acids (20 AA + stop), separate weights for forward vs reverse strand
- `STArgmaxSoftmaxGeneric` — straight-through estimator: argmax in forward pass (discrete AA), softmax gradients in backward pass

**Training** (`translator_train.ipynb`): random DNA → Biopython translation → cross-entropy loss. Converges to 0 loss in <2 epochs (it's learning a deterministic mapping).

### 3. OLG Sequence Optimizer

**What**: The core optimization engine. Given a target protein sequence for the overlapping region, it finds DNA sequences that:
1. Encode the correct protein (from the alternative start codon)
2. Maximize MRL for *both* the full-length UTR and the truncated UTR (from the alt start)
3. Optionally stay close to a seed sequence (edit distance constraint)

## Algorithmic Details

### OptimusOLG Forward Pass

The sequence is parameterized as **learnable logits** `(B, 4, L)`:

1. **Layer normalization** — applied separately to left (before alt start) and right (after alt start) regions, stabilizing logit scale
2. **Seed biasing** — if a seed sequence is provided, its log-probabilities are added to the normalized logits
3. **Right overhang masking** — positions beyond the sequence get masked with `-1e8` to force specific nucleotides at the tail
4. **Straight-through sampling** — `STArgmaxSoftmaxGeneric`: softmax → argmax in forward (discrete one-hot), gradients flow through softmax in backward
5. **Start codon forcing** — ATG is hard-written at position `alt_start`
6. **Two MRL predictions**:
   - `mrl1`: Optimus predicts MRL on the full sequence (upstream start)
   - `mrl2`: Optimus predicts MRL on left-padded slice up to alt_start (downstream start)
7. **Protein translation** — the right portion (from alt_start onward) is translated via the Translator to get the amino acid sequence

### Two-Phase Optimization

#### Phase 1: Gradient Descent

Three loss terms with **independent gradient computation and per-batch clipping**:

1. **MRL loss**: `-(w_mrl1 * mrl1 + (1 - w_mrl1) * mrl2)` — maximize weighted mean MRL across both reading frames. `w_mrl1` (default 0.5) controls the balance: higher values prioritize the upstream start, lower values prioritize the downstream start
2. **Protein loss**: cross-entropy between predicted AA sequence and target AA sequence — enforces the protein constraint
3. **Edit loss**: smooth L1 distance to seed sequence (masked where seed is zero/padding)

Key gradient trick:
- Each loss is backpropagated separately
- Protein and edit gradients are **clipped relative to the MRL gradient norm** — this prevents the protein constraint from overwhelming the MRL objective
- Gradients are then summed and applied

Runs for `min_steps` to `max_steps` iterations. A result is "acceptable" only when protein loss = 0 (exact AA match).

#### Phase 2: Simulated Annealing / Greedy Search

Takes the best sequences from Phase 1 and refines them:

- **Mutation strategy**: randomly decides whether to mutate in the left region (free mutations) or the right region (synonymous mutations only via `SynonymousMutator`)
- **SynonymousMutator**: precomputes all synonymous codon alternatives from the standard genetic code, then swaps codons to synonymous variants — preserving the protein while changing the DNA
- **Acceptance**: Metropolis criterion with exponential cooling. Default config sets `tau0 ≈ 0`, making it effectively **greedy** (only accept improvements)
- Tracks best per-batch results where protein constraint is satisfied

### Supporting Infrastructure

- **`STArgmaxSoftmaxGeneric`**: `ret = argmax - softmax.detach() + softmax` — the classic straight-through trick. Forward gives discrete one-hot, backward gives softmax gradients.
- **`KmerCounter`**: GPU-accelerated k-mer counting using base-5 encoding (A=0,C=1,G=2,T=3,N=4). Used for train/test split filtering by sequence similarity.
- **`quantile_normalize_binned`**: normalizes MRL values stratified by read count bins — corrects for the read-count-dependent noise in MPRA measurements.
- **`select_onehot_by_priority`**: deduplicates sequences, keeping the one with highest read count.
- **`stratified_split`**: train/test split stratified by sequence length, with high-read-count sequences prioritized for the test set.
- **`ProportionalMultiDatasetSampler`**: batches from multiple datasets proportional to their size — enables joint training across eGFP (unmod/pseudo/m1pseudo) and mCherry datasets with shared conv weights.

## Configuration

Optimization is configured via YAML files. A base config (`src/olg5utr/config_base.yaml`) provides defaults; experiment-specific overrides are layered on top:

```python
from olg5utr import BASE_CONFIG_PATH, OptimizationConfig
config = OptimizationConfig.from_yaml(BASE_CONFIG_PATH, "my_experiment.yaml")
```

Multiple files are merged left-to-right — later files override earlier ones. Only specify the fields you want to change.

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `seq_length` | 100 | Total nucleotide sequence length |
| `alt_start_pos` | 50 | 0-based position of the A in the alternative ATG start codon |
| `fix_aa_seq` | null | Target protein for the overlapping ORF. Single letters for exact residues, `[KR]` for degenerate (K or R), `X` for wildcard |
| `right_overhang` | null | Fixed DNA nucleotides past the ORF end (e.g. `"A"`, `"TC"`) |
| `seed_seq` | null | Optional seed DNA sequence (5' side, before alt_start) |
| `w_mrl1` | 0.5 | Weight for MRL1 in loss (0–1). 0.5 = equal, 1.0 = MRL1 only |
| `n_batch` | 10 | Number of sequences optimized in parallel |
| `device` | cuda:0 | Compute device |

### Reading Frame

The reading frame of the overlapping ORF relative to the main CDS is determined by `alt_start_pos` and `seq_length`:

```
frame_offset = (alt_start_pos % 3 - seq_length % 3) % 3
```

For example, with `seq_length=100` (`100 % 3 = 1`):
- `alt_start_pos=72` → `(0 - 1) % 3 = 2` → **+2 frame**
- `alt_start_pos=71` → `(2 - 1) % 3 = 1` → **+1 frame**

### GD / SA Sub-configs

Gradient descent (`gd:`) and simulated annealing (`sa:`) each have their own block:

```yaml
gd:
  min_steps: 1000       # minimum GD iterations
  max_steps: 1000       # maximum GD iterations
  learning_rate: 0.001
  w_prot: 1.0           # protein gradient scale (relative to MRL grad norm)
  w_edit: 0.5           # edit gradient scale (relative to MRL grad norm)
sa:
  steps: 2000           # SA iterations
  max_mutations: 1      # mutations per step
  tau0: 1.0e-08         # initial temperature (≈0 = greedy)
```

### Example Override

```yaml
# 8x[GS] degenerate protein, +2 frame, 100nt
seq_length: 100
alt_start_pos: 72
fix_aa_seq: "[GS][GS][GS][GS][GS][GS][GS][GS]"
right_overhang: A
n_batch: 50
```

## Data Flow

```
Random/Seed DNA logits (B, 4, 100)
    → LayerNorm → ST-Argmax → discrete one-hot DNA
    → Force ATG at alt_start
    → Split into two UTR views
    → Optimus CNN → MRL₁, MRL₂
    → Translator CNN → predicted protein
    → Loss = -weighted_mean(MRL) + protein_CE + edit_distance
    → Gradient descent (clipped per-component)
    → Greedy/SA refinement with synonymous mutations
    → Output: optimized DNA sequence
```
