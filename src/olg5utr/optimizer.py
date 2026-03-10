from Bio.Data.CodonTable import standard_dna_table
import numpy as np
import torch
import torch.nn.functional as F

from olg5utr.config import OptimizationConfig
from olg5utr.encoding import (
    AA_ALPHABET,
    DNA_ALPHABET,
    build_aa_mask,
    build_right_overhang_mask,
    build_seed_onehot,
    dna_to_onehot,
)
from olg5utr.models import Optimus, OptimusOLG, Translator


def load_translator(
    channels: int = 1024,
    state_dict_path: str | None = None,
    device: torch.device | None = None,
) -> torch.nn.Module:
    """Load and initialize the Translator model from saved weights.

    Args:
        channels: Number of intermediate channels for codon features.
        state_dict_path: Path to the saved state dict file.
        device: Device to load the model onto.

    Returns:
        Translator model with loaded weights in eval mode.
    """
    translator = Translator(channels)
    translator = translator.to(device)
    translator.load_state_dict(torch.load(state_dict_path, map_location=device, weights_only=True))
    return translator


def load_optimus_model(
    device: torch.device | None = None,
    state_dict_path: str | None = None,
) -> torch.nn.Module:
    """Load and initialize the Optimus MRL prediction model from saved weights.

    Args:
        device: Device to load the model onto.
        state_dict_path: Path to the saved checkpoint file (must contain
            a ``model_state_dict`` key).

    Returns:
        Optimus model with loaded weights in eval mode.
    """
    model_mrl = Optimus(
        inp_len=100,
        nbr_filters=120,
        filter_len=8,
        border_mode="same",
        dropout1=0.0,
        dropout2=0.0,
        dropout3=0.2,
        nodes=40,
        out_kw=["egfp_unmod", "egfp_pseudo", "egfp_m1pseudo", "mcherry_unmod"],
        n_out_col=1,
    )
    model_mrl = model_mrl.to(device)
    checkpoint = torch.load(state_dict_path, map_location=device, weights_only=True)
    model_mrl.load_state_dict(checkpoint["model_state_dict"])
    return model_mrl


class OptimizationResult:
    """Container for optimization results across gradient and SA phases.

    Attributes:
        best_results: List of best ``[sampled, mrl1, mrl2, step]`` per batch index,
            or None if no acceptable result was found.
        history: List of per-step records ``[mrl1, mrl2, loss_prot, num_diff, loss_edit]``.
        min_losses: List of minimum MRL losses per batch index.
        acceptable_batch: Tensor of batch indices that met protein constraints.
    """

    def __init__(self) -> None:
        self.best_results: list = []
        self.history: list = []
        self.min_losses: list = []
        self.acceptable_batch: torch.Tensor | None = None

    def update_best(
        self,
        idx: int,
        sampled: torch.Tensor,
        mrl1: torch.Tensor,
        mrl2: torch.Tensor,
        step: int,
        loss: torch.Tensor,
    ) -> None:
        """Update the best result for a batch index if loss improved.

        Args:
            idx: Batch index.
            sampled: One-hot nucleotide sequence for this batch element.
            mrl1: MRL prediction for the full sequence.
            mrl2: MRL prediction from the alternative start.
            step: Optimization step number.
            loss: MRL loss value for comparison.
        """
        if loss < self.min_losses[idx]:
            self.min_losses[idx] = loss.detach().clone()
            self.best_results[idx] = [
                sampled.detach().clone(),
                mrl1.detach().clone(),
                mrl2.detach().clone(),
                step,
            ]

    def add_history(
        self,
        mrl1: torch.Tensor,
        mrl2: torch.Tensor,
        loss_prot: torch.Tensor,
        num_diff: torch.Tensor,
        loss_edit: torch.Tensor,
    ) -> None:
        """Append a step's metrics to the history.

        Args:
            mrl1: MRL predictions for the full sequence.
            mrl2: MRL predictions from the alternative start.
            loss_prot: Protein constraint loss.
            num_diff: Number of nucleotide changes from previous step.
            loss_edit: Edit distance loss from seed sequence.
        """
        self.history.append(
            [
                mrl1.detach().clone(),
                mrl2.detach().clone(),
                loss_prot.detach().clone(),
                num_diff.detach().clone(),
                loss_edit.detach().clone(),
            ]
        )

    def filter_acceptable(self) -> None:
        """Filter results to keep only batches that met protein constraints.

        Sets ``acceptable_batch`` to indices where ``best_results[i]`` is not
        None, and slices history to match. If no acceptable results exist,
        sets ``best_results`` and ``history`` to None.
        """
        self.acceptable_batch = torch.tensor(
            [i for i in range(len(self.best_results)) if self.best_results[i] is not None]
        )

        if self.acceptable_batch.shape[0] > 0:
            self.history = [
                [h[i][self.acceptable_batch] for i in range(len(h))] for h in self.history
            ]
            self.best_results = [r for r in self.best_results if r is not None]
        else:
            self.history = None
            self.best_results = None


def calculate_mrl_loss(
    mrl1: torch.Tensor,
    mrl2: torch.Tensor,
    w_mrl1: float = 0.5,
) -> torch.Tensor:
    """Calculate MRL loss: maximize weighted mean of MRL1 and MRL2.

    Loss = -(w_mrl1 * mrl1 + (1 - w_mrl1) * mrl2)

    ``w_mrl1`` controls the balance: 0.5 weights both equally,
    higher values prioritize MRL1, lower values prioritize MRL2.

    Args:
        mrl1: MRL predictions for the first reading frame.
        mrl2: MRL predictions for the second reading frame.
        w_mrl1: MRL1 weight fraction (0 to 1). Default 0.5 = equal.

    Returns:
        Per-element loss tensor (lower is better).
    """
    return -(w_mrl1 * mrl1 + (1.0 - w_mrl1) * mrl2)


def calculate_protein_loss(
    fix_aa: torch.Tensor,
    prot: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Calculate protein constraint loss from acceptable-residue masks.

    Measures whether predicted probability mass falls on acceptable residues.
    Returns 0 when all predicted probability is on acceptable residues.

    Supports three kinds of positions in ``fix_aa``:
    - **Standard** (one-hot): exactly one residue acceptable → ``-log(prot_k)``
    - **Degenerate** (multi-hot): multiple residues acceptable →
      ``-log(prot_k + prot_j + ...)``
    - **Wildcard** (all ones): any residue acceptable → ``-log(1) = 0``

    Args:
        fix_aa: Acceptable-residue mask ``(batch, 21, aa_len)``. Binary values:
            1 where a residue is acceptable, 0 otherwise.
        prot: Predicted amino acid probabilities ``(batch, 21, aa_len)``.
        eps: Small constant for numerical stability in log.

    Returns:
        Per-batch loss tensor.
    """
    acceptable_prob = torch.sum(fix_aa * prot, dim=1)
    return (-1.0 * torch.log(acceptable_prob + eps)).mean(-1)


def calculate_edit_loss(
    seed_seq: torch.Tensor,
    seq: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Calculate edit loss (smooth L1) between seed and current sequences.

    Padding positions (all zeros in the channel dimension of seed_seq) are
    masked out and do not contribute to the loss.

    Args:
        seed_seq: Seed/reference one-hot sequence ``(batch, 4, length)``.
        seq: Current one-hot sequence ``(batch, 4, length)``.
        eps: Unused, kept for API compatibility.

    Returns:
        Per-batch loss tensor.
    """
    l1 = F.smooth_l1_loss(seq, seed_seq, reduction="none").sum(1)
    mask = torch.all(seed_seq == 0, dim=1)
    l1[mask] = 0
    return l1.mean(1)


class GradientOptimizer:
    """Gradient-based optimization phase for the OLG pipeline.

    Optimizes nucleotide sequences by computing per-task gradients (MRL,
    protein, edit) and rescaling protein/edit gradient norms relative to
    the MRL gradient norm before combining them.

    Args:
        model: An ``OptimusOLG`` model instance with learnable weight parameters.
        config: Optimization configuration.
    """

    def __init__(self, model: torch.nn.Module, config: OptimizationConfig) -> None:
        self.model = model
        self.config = config
        self.optimizer = self._create_optimizer()

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create the AdamW optimizer from config parameters."""
        gd = self.config.gd
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=gd.learning_rate,
            betas=(gd.beta1, gd.beta2),
            weight_decay=gd.weight_decay,
        )

    def _backward_and_save(
        self,
        loss: torch.Tensor,
        retain: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Backpropagate loss and return a detached copy of the weight gradient.

        Args:
            loss: Scalar loss to backpropagate.
            retain: Whether to retain the computation graph.

        Returns:
            Tuple of (gradient copy, per-batch gradient L2 norm).
        """
        loss.backward(retain_graph=retain)
        grad = self.model.weight.grad.data
        per_batch_norm = grad.flatten(1).norm(p=2, dim=1)
        return grad.detach().clone(), per_batch_norm

    def _compute_gradients(
        self,
        loss_mrl: torch.Tensor,
        loss_prot: torch.Tensor,
        loss_edit: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute per-task gradients with norm rescaling.

        Protein and edit gradients are L2-normalized then scaled to
        ``factor * mrl_grad_norm``, where factor is ``w_prot`` or ``w_edit``.
        The MRL gradient is kept at its raw scale.

        Args:
            loss_mrl: MRL loss tensor (per-batch).
            loss_prot: Protein constraint loss tensor (per-batch).
            loss_edit: Edit distance loss tensor (per-batch).

        Returns:
            Tuple of (mrl_gradients, protein_gradients, edit_gradients),
            each with shape matching ``model.weight``.
        """
        self.optimizer.zero_grad()
        grads_mrl, mrl_grad_norm = self._backward_and_save(loss_mrl.sum(), retain=True)

        self.optimizer.zero_grad()
        grads_prot, _ = self._backward_and_save(loss_prot.sum(), retain=True)

        self.optimizer.zero_grad()
        grads_edit, _ = self._backward_and_save(loss_edit.sum(), retain=False)

        self.optimizer.zero_grad()

        # Rescale protein/edit gradient norms to be proportional to MRL gradient norm
        for grad, factor in [
            (grads_prot, self.config.gd.w_prot),
            (grads_edit, self.config.gd.w_edit),
        ]:
            grad_norm = grad.flatten(1).norm(p=2, dim=1).view(-1, 1, 1) + self.config.eps
            grad.mul_((factor * mrl_grad_norm).view(-1, 1, 1) / grad_norm)

        return grads_mrl, grads_prot, grads_edit

    def optimize(
        self,
        fix_aa: torch.Tensor,
        seed_onehot: torch.Tensor | None,
    ) -> OptimizationResult:
        """Run gradient-based optimization.

        Args:
            fix_aa: Target amino acid one-hot tensor ``(batch, 21, aa_len)``.
            seed_onehot: Optional seed sequence for edit loss. If None,
                the initial random sequence is used as the seed.

        Returns:
            Optimization results with best sequences and training history.
        """
        result = OptimizationResult()
        result.min_losses = [float("inf")] * self.config.n_batch
        result.best_results = [None] * self.config.n_batch

        with torch.inference_mode():
            sampled, mrl1, mrl2, prot = self.model()
            prev_sample = sampled.detach().clone()

        found_result = False
        step = 0

        while (not found_result) or (step < self.config.gd.min_steps):
            self.optimizer.zero_grad()
            sampled, mrl1, mrl2, prot = self.model()

            loss_mrl = calculate_mrl_loss(mrl1, mrl2, self.config.w_mrl1)
            loss_prot = calculate_protein_loss(fix_aa, prot, self.config.eps)
            if seed_onehot is None:
                seed_onehot = sampled.detach().clone()
            loss_edit = calculate_edit_loss(seed_onehot, sampled, self.config.eps)

            # Compute and apply gradients (weights applied inside via norm rescaling)
            grads_mrl, grads_prot, grads_edit = self._compute_gradients(
                loss_mrl, loss_prot, loss_edit
            )
            self.model.weight.grad = grads_mrl + grads_prot + grads_edit
            self.optimizer.step()

            num_diff = ((prev_sample - sampled) ** 2).sum([1, 2]) * 0.5
            prev_sample = sampled.detach().clone()

            for i in range(self.config.n_batch):
                if (loss_prot[i] == 0.0) and (loss_mrl[i] < result.min_losses[i]):
                    result.update_best(i, sampled[i], mrl1[i], mrl2[i], step, loss_mrl[i])

            acceptable_count = sum(1 for r in result.best_results if r is not None)
            if acceptable_count == self.config.n_batch:
                found_result = True

            result.add_history(mrl1, mrl2, loss_prot, num_diff, loss_edit)

            step += 1
            if step >= self.config.gd.max_steps:
                break

        return result


class SimulatedAnnealer:
    """Simulated annealing optimization phase.

    Operates on discrete one-hot sequences using random mutations and
    Metropolis acceptance criterion. Typically used as greedy search with
    ``tau0 ~ 0`` and ``max_mutations = 1``.

    Args:
        model: An ``OptimusOLG`` model instance (used in passthrough mode).
        config: Optimization configuration.
    """

    def __init__(self, model: torch.nn.Module, config: OptimizationConfig) -> None:
        self.model = model
        self.config = config
        self.syn_mutator = SynonymousMutator(self.config.device)

    def _calculate_temperature(self, step: int) -> float:
        """Calculate annealing temperature for the current step.

        Args:
            step: Current SA step number.

        Returns:
            Temperature value (clamped to ``min_temp``).
        """
        sa = self.config.sa
        return np.maximum(
            sa.tau0 * np.exp(-sa.anneal_rate * step),
            sa.min_temp,
        )

    @staticmethod
    def _anneal(
        loss: float | torch.Tensor,
        loss_perturbed: float | torch.Tensor,
        tau: float | torch.Tensor,
    ) -> bool:
        """Metropolis acceptance criterion for simulated annealing.

        Always accepts improvements. Accepts deteriorations with probability
        ``exp(-(loss_perturbed - loss) / tau)``.

        Args:
            loss: Current loss value (lower is better).
            loss_perturbed: Proposed new loss value.
            tau: Temperature parameter. High = more exploration, low = greedy.

        Returns:
            True if the proposed state should be accepted.
        """
        if loss_perturbed < loss:
            return True
        ap = torch.exp((loss - loss_perturbed) / tau)
        return ap.item() > torch.rand(1).item()

    @staticmethod
    def mutate_inplace(
        sequences: torch.Tensor,
        num_mutations: int,
        target_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply random point mutations to one-hot encoded sequences in-place.

        Each mutated position is changed to a different nucleotide (never the
        same). Positions are sampled without replacement per batch element.

        Args:
            sequences: One-hot tensor of shape ``(batch, 4, length)``.
                Modified in-place.
            num_mutations: Number of positions to mutate per sequence.
            target_positions: Optional pre-selected positions tensor of shape
                ``(batch, num_mutations)``. If None, positions are randomly sampled.

        Returns:
            The same tensor (modified in-place).
        """
        if num_mutations <= 0:
            return sequences

        B, C, L = sequences.shape
        device = sequences.device

        if target_positions is None:
            num_mutations = min(num_mutations, L)
            positions = torch.stack(
                [torch.randperm(L, device=device)[:num_mutations] for _ in range(B)]
            )
        else:
            positions = target_positions

        batch_idx = torch.arange(B, device=device).unsqueeze(1)
        # Gathered shape is (B, num_mutations, C)
        current_nucs = torch.argmax(sequences[batch_idx, :, positions], dim=-1)
        new_nucs = (current_nucs + torch.randint(1, C, (B, num_mutations), device=device)) % C
        sequences[batch_idx, :, positions] = 0
        sequences[batch_idx, new_nucs, positions] = 1

        return sequences

    def optimize(
        self,
        result: OptimizationResult,
        fix_aa: torch.Tensor,
        initial_step: int = 0,
        seed_onehot: torch.Tensor | None = None,
    ) -> OptimizationResult:
        """Run simulated annealing on the best results from gradient optimization.

        Left-side mutations use random nucleotide changes. Right-side mutations
        (downstream of alt start) use synonymous codon substitutions to preserve
        the encoded protein.

        Args:
            result: Optimization results from the gradient phase.
            fix_aa: Target amino acid one-hot tensor for acceptable batches.
            initial_step: Step offset for history tracking.
            seed_onehot: Optional seed sequence for edit loss computation.

        Returns:
            Updated optimization results.
        """
        if not result.best_results:
            return result

        current_input = torch.stack([r[0] for r in result.best_results])
        B = current_input.shape[0]

        with torch.inference_mode():
            sampled, mrl1, mrl2, prot = self.model(input_onehot=current_input)

            loss_mrl = calculate_mrl_loss(mrl1, mrl2, self.config.w_mrl1)
            loss_prot = calculate_protein_loss(fix_aa, prot, self.config.eps)
            if seed_onehot is None:
                seed_onehot = sampled.detach().clone()
            loss_edit = calculate_edit_loss(seed_onehot, sampled, self.config.eps)

            loss_total = (
                loss_mrl + self.config.sa.w_prot * loss_prot + self.config.sa.w_edit * loss_edit
            )
            prev_sample = sampled.detach().clone()

            last_loss_prot = loss_prot.detach().clone()
            last_loss_total = loss_total.detach().clone()

            for sa_step in range(self.config.sa.steps):
                last_input = current_input.clone()

                num_mut = torch.randint(1, self.config.sa.max_mutations + 1, (1, 1)).item()
                mutate_left = torch.rand(B) * self.model.seq_length < self.model.alt_start

                if mutate_left.any():
                    left_idx = mutate_left.nonzero(as_tuple=True)[0]
                    SimulatedAnnealer.mutate_inplace(
                        current_input[left_idx, :, : self.model.alt_start], num_mut
                    )
                if (~mutate_left).any():
                    right_idx = (~mutate_left).nonzero(as_tuple=True)[0]
                    right_end = self.model.alt_start + 3 * self.model.aa_len
                    self.syn_mutator.mutate_batch(
                        current_input[right_idx, :, self.model.alt_start : right_end],
                        num_mut,
                    )

                sampled, mrl1, mrl2, prot = self.model(input_onehot=current_input)

                loss_mrl = calculate_mrl_loss(mrl1, mrl2, self.config.w_mrl1)
                loss_prot = calculate_protein_loss(fix_aa, prot, self.config.eps)
                loss_edit = calculate_edit_loss(seed_onehot, sampled, self.config.eps)

                loss_total = (
                    loss_mrl
                    + self.config.sa.w_prot * loss_prot
                    + self.config.sa.w_edit * loss_edit
                )

                tau = self._calculate_temperature(sa_step)

                for i in range(len(result.best_results)):
                    is_swapped = False

                    if loss_prot[i] <= last_loss_prot[i]:
                        is_swapped = SimulatedAnnealer._anneal(
                            last_loss_total[i], loss_total[i], tau
                        )

                    if is_swapped:
                        last_loss_prot[i] = loss_prot[i].detach().clone()
                        last_loss_total[i] = loss_total[i].detach().clone()

                        if (loss_prot[i] == 0.0) and (loss_mrl[i] < result.min_losses[i]):
                            result.update_best(
                                i,
                                sampled[i],
                                mrl1[i],
                                mrl2[i],
                                initial_step + sa_step,
                                loss_mrl[i],
                            )
                    else:
                        current_input[i] = last_input[i]

                num_diff = ((prev_sample - sampled) ** 2).sum([1, 2]) * 0.5
                prev_sample = sampled.detach().clone()

                result.add_history(mrl1, mrl2, loss_prot, num_diff, loss_edit)

        return result


class SynonymousMutator:
    """Mutator that substitutes codons with synonymous alternatives.

    Ensures mutations do not change the encoded protein sequence by only
    swapping codons that encode the same amino acid.

    Args:
        device: Compute device string.
        codon_table: Optional custom codon-to-amino-acid mapping. If None,
            uses the standard genetic code from Biopython.
    """

    def __init__(
        self,
        device: str = "cpu",
        codon_table: dict[str, str] | None = None,
    ) -> None:
        self.device = device
        self._build_lookup_tables(codon_table)

    def _build_lookup_tables(self, codon_table: dict[str, str] | None = None) -> None:
        """Build lookup tensors for synonymous codon substitutions.

        Args:
            codon_table: Codon-to-amino-acid mapping dict. If None, uses
                the standard genetic code.
        """
        if codon_table is None:
            codon_table = self._get_standard_genetic_code()

        unique_aas = AA_ALPHABET
        self.aa_to_idx = {aa: i for i, aa in enumerate(unique_aas)}
        self.idx_to_aa = {i: aa for aa, i in self.aa_to_idx.items()}

        nucleotides = DNA_ALPHABET
        codon_to_aa_flat = torch.zeros(64, dtype=torch.long, device=self.device)

        for i in range(4):
            for j in range(4):
                for k in range(4):
                    codon = nucleotides[i] + nucleotides[j] + nucleotides[k]
                    codon_idx = i * 16 + j * 4 + k

                    if codon in codon_table:
                        aa = codon_table[codon]
                        codon_to_aa_flat[codon_idx] = self.aa_to_idx[aa]
                    else:
                        if "X" not in self.aa_to_idx:
                            self.aa_to_idx["X"] = len(self.aa_to_idx)
                            self.idx_to_aa[self.aa_to_idx["X"]] = "X"
                        codon_to_aa_flat[codon_idx] = self.aa_to_idx["X"]

        # Compute max synonyms from the actual codon table being used
        aa_counts = np.unique(list(codon_table.values()), return_counts=True)[1]
        max_syns = int(aa_counts.max()) - 1
        self.syn_codons = torch.full((64, max_syns), -1, dtype=torch.long, device=self.device)
        self.syn_counts = torch.zeros(64, dtype=torch.long, device=self.device)

        num_aas = len(self.aa_to_idx)
        for aa_idx in range(num_aas):
            aa_codons = (codon_to_aa_flat == aa_idx).nonzero(as_tuple=True)[0]
            if len(aa_codons) > 1:
                for codon in aa_codons:
                    others = aa_codons[aa_codons != codon]
                    self.syn_codons[codon, : len(others)] = others
                    self.syn_counts[codon] = len(others)

        self.codon_decode = torch.zeros(64, 3, dtype=torch.long, device=self.device)
        for i in range(64):
            self.codon_decode[i] = torch.tensor([i // 16, (i // 4) % 4, i % 4])

    def _get_standard_genetic_code(self) -> dict:
        """Return the standard genetic code as a codon-to-amino-acid dict."""
        return standard_dna_table.forward_table

    def mutate_batch(
        self,
        one_hot: torch.Tensor,
        num_mutations: int,
    ) -> torch.Tensor:
        """Apply synonymous codon mutations to a batch of one-hot sequences.

        Randomly selects mutable codon positions and replaces each with a
        synonymous alternative. Operates in-place on the input tensor.

        Args:
            one_hot: One-hot encoded sequences of shape ``(batch, 4, length)``.
                Length must be divisible by 3. Modified in-place.
            num_mutations: Number of codon positions to mutate per sequence.

        Returns:
            The same tensor (modified in-place).
        """
        B, _C, _L = one_hot.shape

        nuc_indices = torch.argmax(one_hot, dim=1)  # (B, L)
        codon_indices = (
            nuc_indices[:, 0::3] * 16 + nuc_indices[:, 1::3] * 4 + nuc_indices[:, 2::3]
        )  # (B, num_codons)

        batch_syn_counts = self.syn_counts[codon_indices]
        mutable_mask = batch_syn_counts > 0

        for b in range(B):
            mutable_pos = torch.where(mutable_mask[b])[0]

            if len(mutable_pos) == 0:
                continue

            n_mut = min(num_mutations, len(mutable_pos))
            if n_mut == 0:
                continue

            perm = torch.randperm(len(mutable_pos), device=one_hot.device)
            selected_pos = mutable_pos[perm[:n_mut]]

            for pos in selected_pos:
                codon_idx = codon_indices[b, pos]
                n_syns = self.syn_counts[codon_idx]

                if n_syns > 0:
                    syn_idx = torch.randint(n_syns, (1,), device=one_hot.device)[0]
                    new_codon_idx = self.syn_codons[codon_idx, syn_idx]
                    new_codon = self.codon_decode[new_codon_idx]

                    start_pos = pos * 3
                    one_hot[b, :, start_pos : start_pos + 3] = 0
                    one_hot[
                        b,
                        new_codon,
                        torch.arange(start_pos, start_pos + 3, device=one_hot.device),
                    ] = 1

        return one_hot


class OptimusOLGPipeline:
    """Main pipeline for running the complete OLG 5' UTR optimization.

    Runs a two-phase optimization: gradient-based optimization followed by
    simulated annealing. The gradient phase optimizes learnable weight
    parameters, while SA makes discrete mutations to the best sequences.

    Args:
        config: Optimization configuration. Uses defaults if None.
    """

    def __init__(self, config: OptimizationConfig | None = None) -> None:
        self.config = config or OptimizationConfig()
        self.device = torch.device(self.config.device)

        self.translator = load_translator(
            channels=self.config.model.translator_channels,
            device=self.device,
            state_dict_path=self.config.model.translator_weights_path,
        )
        self.model_mrl = load_optimus_model(
            device=self.device,
            state_dict_path=self.config.model.optimus_weights_path,
        )

        if isinstance(self.config.start_codon, str):
            self.start_codon = (
                dna_to_onehot(self.config.start_codon, 3)
                .transpose(1, 0)
                .unsqueeze(0)
                .repeat(self.config.n_batch, 1, 1)
            )
        else:
            self.start_codon = dna_to_onehot(self.config.start_codon, 3).transpose(1, 0)

    def _resolve_inputs(
        self,
        fix_aa: torch.Tensor | None,
        right_overhang_mask: torch.Tensor | None,
        seed_onehot: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Resolve tensor inputs from arguments or config strings.

        Tensor arguments take precedence over config strings. Config strings
        are encoded and batched automatically.

        Args:
            fix_aa: Target amino acid tensor, or None to use config.
            right_overhang_mask: Overhang mask tensor, or None to use config.
            seed_onehot: Seed sequence tensor, or None to use config.

        Returns:
            Tuple of (fix_aa, right_overhang_mask, seed_onehot).

        Raises:
            ValueError: If fix_aa or right_overhang_mask cannot be resolved
                from either argument or config.
        """
        n = self.config.n_batch

        if fix_aa is None:
            if self.config.fix_aa_seq is None:
                msg = "fix_aa tensor or config.fix_aa_seq must be provided"
                raise ValueError(msg)
            fix_aa = (
                build_aa_mask(self.config.fix_aa_seq)
                .to(self.device)
                .repeat(n, 1, 1)
            )

        if right_overhang_mask is None:
            if self.config.right_overhang is None:
                msg = "right_overhang_mask tensor or config.right_overhang must be provided"
                raise ValueError(msg)
            right_overhang_mask = (
                build_right_overhang_mask(self.config.right_overhang)
                .to(self.device)
                .repeat(n, 1, 1)
            )

        if seed_onehot is None and self.config.seed_seq is not None:
            seed_onehot = (
                build_seed_onehot(self.config.seed_seq, self.config.seq_length)
                .to(torch.float32)
                .to(self.device)
                .repeat(n, 1, 1)
            )

        return fix_aa, right_overhang_mask, seed_onehot

    def run(
        self,
        fix_aa: torch.Tensor | None = None,
        right_overhang_mask: torch.Tensor | None = None,
        seed_onehot: torch.Tensor | None = None,
    ) -> OptimizationResult:
        """Run the complete two-phase optimization pipeline.

        Inputs can be provided as tensors or resolved from config strings.
        Tensor arguments take precedence over config values.

        Args:
            fix_aa: Target amino acid one-hot tensor ``(batch, 21, aa_len)``.
                Falls back to ``config.fix_aa_seq`` if None.
            right_overhang_mask: Boolean mask for the right overhang region.
                Falls back to ``config.right_overhang`` if None.
            seed_onehot: Optional seed sequence to bias optimization toward.
                Falls back to ``config.seed_seq`` if None.

        Returns:
            Optimization results containing best sequences and history.
        """
        fix_aa, right_overhang_mask, seed_onehot = self._resolve_inputs(
            fix_aa, right_overhang_mask, seed_onehot
        )

        # Phase 1: Gradient-based optimization
        print("Starting gradient-based optimization...")
        model = OptimusOLG(
            self.device,
            self.model_mrl,
            self.translator,
            self.config.n_batch,
            self.config.seq_length,
            self.config.alt_start_pos,
            self.start_codon,
            right_overhang_mask,
            seed_onehot,
        ).to(self.device)

        grad_optimizer = GradientOptimizer(model, self.config)
        result = grad_optimizer.optimize(fix_aa, seed_onehot)

        result.filter_acceptable()

        if result.best_results is None:
            print("No acceptable results found in gradient optimization")
            return result

        # Phase 2: Simulated annealing
        n_acceptable = len(result.best_results)
        print(f"Starting simulated annealing with {n_acceptable} sequences...")
        sa_seed = seed_onehot[result.acceptable_batch] if seed_onehot is not None else None
        model = OptimusOLG(
            self.device,
            self.model_mrl,
            self.translator,
            n_acceptable,
            self.config.seq_length,
            self.config.alt_start_pos,
            self.start_codon[:n_acceptable],
            right_overhang_mask[result.acceptable_batch],
            sa_seed,
        ).to(self.device)

        annealer = SimulatedAnnealer(model, self.config)
        result = annealer.optimize(
            result,
            fix_aa[result.acceptable_batch],
            self.config.gd.max_steps,
            sa_seed,
        )

        print(f"Optimization complete. Found {len(result.best_results)} optimized sequences.")
        return result
