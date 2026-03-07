import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

import numpy as np

from typing import List, Iterator, Tuple, Optional, Union, Literal

from olg5utr.encoding import DNA_ALPHABET


def quantile_normalize_binned(
    values: torch.Tensor,
    bin_variable: torch.Tensor,
    n_bins: int = 10
) -> torch.Tensor:
    """
    Quantile normalization that replaces ranks with averaged values at each rank.
    Each rank position gets the average value of all samples at that rank across all bins.

    Args:
        values: 1D tensor of values to be normalized.
        bin_variable: 1D tensor used to define bins for stratified normalization. Should have the same length as values.
        n_bins: Number of bins to create based on quantiles of bin_variable.

    Returns:
        torch.Tensor: Normalized values with the same shape and device as input values.
                     Values are replaced with averaged quantile values computed across all bins.
    """
    device = values.device

    # Create bins
    bin_edges = torch.quantile(bin_variable, torch.linspace(0, 1, n_bins + 1, device=device))
    bin_edges[0] = bin_variable.min() - 1e-8
    bin_edges[-1] = bin_variable.max() + 1e-8

    # Assign bins
    bin_indices = torch.searchsorted(bin_edges[1:], bin_variable)

    # First pass: collect all values and their percentile positions
    all_ranked_values = []
    all_percentiles = []

    for bin_idx in range(n_bins):
        mask = bin_indices == bin_idx
        if mask.sum() <= 1:
            continue

        bin_values = values[mask]
        bin_size = len(bin_values)

        # Get ranks and convert to percentiles [0, 1]
        ranks = torch.argsort(torch.argsort(bin_values)).float()
        percentiles = ranks / (bin_size - 1)

        all_ranked_values.append(bin_values)
        all_percentiles.append(percentiles)

    if not all_ranked_values:
        return values.clone()

    # Concatenate all values and percentiles
    all_values_cat = torch.cat(all_ranked_values)
    all_percentiles_cat = torch.cat(all_percentiles)

    # Create a grid of target percentiles for interpolation
    n_quantiles = 100  # Use 100 quantiles for smooth interpolation
    target_percentiles = torch.linspace(0, 1, n_quantiles, device=device)

    # For each target percentile, find the average value
    quantile_averages = torch.zeros(n_quantiles, device=device)

    for i, target_p in enumerate(target_percentiles):
        # Find values close to this percentile (within a small window)
        window = 0.02  # 2% window
        mask = torch.abs(all_percentiles_cat - target_p) <= window

        if mask.sum() > 0:
            quantile_averages[i] = all_values_cat[mask].mean()
        else:
            # If no values in window, interpolate from nearest values
            if i > 0:
                quantile_averages[i] = quantile_averages[i-1]

    # Handle any remaining zeros by forward/backward fill
    non_zero_mask = quantile_averages != 0
    if non_zero_mask.sum() > 0:
        first_non_zero = non_zero_mask.nonzero()[0].item()
        last_non_zero = non_zero_mask.nonzero()[-1].item()

        # Forward fill before first non-zero
        quantile_averages[:first_non_zero] = quantile_averages[first_non_zero]
        # Backward fill after last non-zero
        quantile_averages[last_non_zero+1:] = quantile_averages[last_non_zero]

    # Second pass: assign averaged values based on percentile positions
    normalized_values = torch.zeros_like(values)

    for bin_idx in range(n_bins):
        mask = bin_indices == bin_idx
        if mask.sum() == 0:
            continue
        elif mask.sum() == 1:
            # Single value bin - assign middle quantile
            normalized_values[mask] = quantile_averages[n_quantiles // 2]
            continue

        bin_values = values[mask]
        bin_size = len(bin_values)

        # Get percentile positions
        ranks = torch.argsort(torch.argsort(bin_values)).float()
        percentiles = ranks / (bin_size - 1)

        # Map percentiles to quantile indices
        quantile_indices = (percentiles * (n_quantiles - 1)).round().long()
        quantile_indices = torch.clamp(quantile_indices, 0, n_quantiles - 1)

        # Assign averaged values
        normalized_values[mask] = quantile_averages[quantile_indices]

    return normalized_values


def scaler(rl: torch.Tensor) -> torch.Tensor:
    """Z-score normalization"""
    m = rl.mean()
    s = rl.std()
    scaled_rl = rl - m
    scaled_rl = scaled_rl / s
    return scaled_rl


def euc_distance_min(
    counts1: torch.Tensor,
    counts2: torch.Tensor,
    chunk_size: Optional[int] = None
) -> torch.Tensor:
    """
    Args:
        counts1: 2D tensor of shape (N, D)
        counts2: 2D tensor of shape (M, D); must have the same feature dimension D as counts1.
        chunk_size: batch size for processing counts1 in chunks to manage memory.

    Returns:
        torch.Tensor: 1D tensor of shape (N,) containing the minimum distance from each point in counts1 to any point in counts2.
    """
    normalized1 = F.normalize(counts1.to(torch.float32), p=2, dim=1)
    normalized2 = F.normalize(counts2.to(torch.float32), p=2, dim=1)

    return torch.cat([ torch.cdist(normalized1[i:(i+chunk_size)], normalized2).min(1).values for i in range(0, normalized1.shape[0], chunk_size) ])


def select_onehot_by_priority(
    onehot_tensor: torch.Tensor,
    priority_vector: torch.Tensor,
    select_mode: Literal['max', 'min'] = 'max'
) -> torch.Tensor:
    """
    Select one representative index for each unique one-hot pattern based on priority values for deduplication

    Args:
        onehot_tensor: 2D tensor of shape (N, D); i.e. one-hot DNA
        priority_vector: 1D tensor of shape (N,) containing priority scores for each sample.
        select_mode: Selection criterion for choosing the best priority.

    Returns:
        torch.Tensor: 1D tensor of dtype long containing the original indices of selected samples.
                     Length equals the number of unique patterns found in onehot_tensor.
                     Indices are sorted by the order of first appearance of each unique pattern.
    """
    unique_patterns, inverse_indices = torch.unique(onehot_tensor, dim=0, return_inverse=True)
    num_unique = unique_patterns.size(0)

    best_priorities = torch.zeros(num_unique, dtype=priority_vector.dtype, device=priority_vector.device)
    if select_mode == 'max':
        best_priorities.fill_(-float('inf'))
        best_priorities.scatter_reduce_(0, inverse_indices, priority_vector, reduce='amax')
    else:  # min
        best_priorities.fill_(float('inf'))
        best_priorities.scatter_reduce_(0, inverse_indices, priority_vector, reduce='amin')

    target_priorities = best_priorities[inverse_indices]
    is_selected = (priority_vector == target_priorities)

    selected_indices = torch.zeros(num_unique, dtype=torch.long, device=onehot_tensor.device)
    for unique_idx in range(num_unique):
        mask = (inverse_indices == unique_idx) & is_selected
        candidates = torch.where(mask)[0]
        selected_indices[unique_idx] = candidates[0]

    return selected_indices


def stratified_split(
    seq: torch.Tensor,
    rl: torch.Tensor,
    tr: torch.Tensor,
    kmers: torch.Tensor,
    fraction: float = 0.2,
    rand_split: bool = False
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Perform stratified train/test split based on sequence lengths, preserving length distribution.

    Args:
        seq: 3D tensor representing sequences, (B, C, L)
        rl: MRL values, (N,)
        tr: priority (total read count), (N, )
        kmers: pass-thru for kmer counts
        fraction: fraction of data for split
        rand_split: If True, randomly split within each length stratum. Else, priortize by tr

    Returns:
        Tuple containing two lists:
        - train_data: List of 5 tensors [seq_train, rl_train, rl_train_scaled, tr_train, kmers_train]
        - test_data: List of 5 tensors [seq_test, rl_test, rl_test_scaled, tr_test, kmers_test]
    """
    lens = seq.sum([1,2]).long()
    unique_labels = torch.unique(lens).to(lens.device)

    train_indices = torch.tensor([], dtype=torch.long)
    test_indices = torch.tensor([], dtype=torch.long)
    for label in unique_labels:
        label_indices = torch.where(lens == label)[0].to(tr.device)
        if rand_split:
            perm = torch.randperm(len(label_indices))
        else:
            perm = tr[label_indices].sort(descending=True, stable=True).indices
        shuffled_indices = label_indices[perm]

        n_test = int(len(shuffled_indices) * fraction)

        train_indices = torch.cat([train_indices, shuffled_indices[n_test:]])
        test_indices = torch.cat([test_indices, shuffled_indices[:n_test]])

    return [ seq[train_indices], rl[train_indices], scaler(rl[train_indices]), tr[train_indices], kmers[train_indices] ], [ seq[test_indices], rl[test_indices], scaler(rl[test_indices]), tr[test_indices], kmers[test_indices] ]


def bin_2d_by_quantiles(
    values: torch.Tensor,
    assignment_vector: torch.Tensor,
    num_bins: int = 10
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Bin 2D values based on quantiles of assignment vector and calculate column averages per bin

    Args:
        values: [n_samples, n_features] - 2D tensor of values
        assignment_vector: [n_samples] - vector to determine bin assignments
        num_bins: number of quantile bins

    Returns:
        bin_averages: [n_bins, n_features] - average of each column per bin
        bin_counts: [n_bins] - number of samples in each bin
        thresholds: [num_bins+1] - quantile thresholds used for binning
    """
    n_samples, n_features = values.shape

    quantiles = torch.linspace(0, 1, num_bins + 1)
    thresholds = torch.quantile(assignment_vector, quantiles)

    bin_indices = torch.bucketize(assignment_vector, thresholds[1:-1])
    bin_indices = torch.clamp(bin_indices, 0, num_bins - 1)

    bin_sums = torch.zeros(num_bins, n_features, dtype=values.dtype, device=values.device)
    bin_counts = torch.zeros(num_bins, dtype=values.dtype, device=values.device)

    for i in range(num_bins):
        mask = bin_indices == i
        if mask.any():
            bin_sums[i] = values[mask].sum(dim=0)
            bin_counts[i] = mask.sum().float()

    bin_averages = torch.where(
        bin_counts.unsqueeze(-1) > 0,
        bin_sums / bin_counts.unsqueeze(-1),
        torch.zeros_like(bin_sums)
    )

    return bin_averages, bin_counts, thresholds


class KmerCounter:
    """
    Efficient k-mer counter for one-hot encoded DNA sequences.
    Handles N's (all zeros) and provides option to exclude k-mers containing N's.
    """

    def __init__(self, k: int, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.k = k
        self.device = device
        self.vocab_size = 5 ** k  # 5 possible values: A, T, G, C, N

        self.powers = torch.tensor([5 ** i for i in range(k-1, -1, -1)],
                                   dtype=torch.long, device=device)

        self._create_n_mask()

    def _create_n_mask(self):
        """Create a mask to identify which k-mer indices contain N's."""
        all_kmers = torch.arange(self.vocab_size, device=self.device)

        has_n = torch.zeros(self.vocab_size, dtype=torch.bool, device=self.device)

        for idx in range(self.vocab_size):
            base5 = []
            num = idx
            for _ in range(self.k):
                base5.append(num % 5)
                num //= 5
            has_n[idx] = 4 in base5

        self.n_mask = has_n

    def one_hot_to_indices(self, one_hot: torch.Tensor) -> torch.Tensor:
        """
        Convert one-hot encoded sequences to base-5 indices.

        Args:
            one_hot: Tensor of shape (batch, 4, length) or (4, length)

        Returns:
            Tensor of indices where A=0, T=1, G=2, C=3, N=4
        """
        indices = torch.argmax(one_hot, dim=-2 if one_hot.dim() == 3 else 0)

        is_n = (one_hot.sum(dim=-2 if one_hot.dim() == 3 else 0) == 0)

        indices[is_n] = 4

        return indices

    def extract_kmers(self, sequences: torch.Tensor) -> torch.Tensor:
        """
        Extract all k-mers from sequences using sliding window.

        Args:
            sequences: Tensor of shape (batch, 4, length) with one-hot encoding

        Returns:
            Tensor of k-mer indices of shape (batch, num_kmers)
        """
        batch_size, _, seq_len = sequences.shape
        num_kmers = seq_len - self.k + 1

        indices = self.one_hot_to_indices(sequences)  # (batch, length)

        kmers = indices.unfold(dimension=1, size=self.k, step=1)  # (batch, num_kmers, k)

        kmer_indices = (kmers * self.powers).sum(dim=-1)  # (batch, num_kmers)

        return kmer_indices

    def count_kmers(self,
                   sequences: torch.Tensor,
                   exclude_n: bool = False,
                   return_dense: bool = True) -> torch.Tensor:
        """
        Count k-mers in batch of sequences.

        Args:
            sequences: Tensor of shape (batch, 4, length) with one-hot encoding
            exclude_n: If True, exclude k-mers containing N's from counts
            return_dense: If True, return dense count matrix; if False, return sparse

        Returns:
            If return_dense: Tensor of shape (batch, 5^k) with k-mer counts
            If not return_dense: Tuple of (indices, values, shape) for sparse tensor
        """
        batch_size = sequences.shape[0]

        kmer_indices = self.extract_kmers(sequences)  # (batch, num_kmers)

        if exclude_n:
            mask = ~self.n_mask[kmer_indices]
            kmer_indices = kmer_indices * mask - 1

        if return_dense:
            counts = torch.zeros(batch_size, self.vocab_size,
                                dtype=torch.long, device=self.device)

            for b in range(batch_size):
                seq_kmers = kmer_indices[b]
                if exclude_n:
                    seq_kmers = seq_kmers[seq_kmers >= 0]

                if seq_kmers.numel() > 0:
                    bincount = torch.bincount(seq_kmers, minlength=self.vocab_size)
                    counts[b] = bincount

            return counts
        else:
            all_indices = []
            all_values = []

            for b in range(batch_size):
                seq_kmers = kmer_indices[b]
                if exclude_n:
                    seq_kmers = seq_kmers[seq_kmers >= 0]

                if seq_kmers.numel() > 0:
                    unique_kmers, counts = torch.unique(seq_kmers, return_counts=True)
                    batch_indices = torch.full_like(unique_kmers, b)
                    indices = torch.stack([batch_indices, unique_kmers])
                    all_indices.append(indices)
                    all_values.append(counts)

            if all_indices:
                indices = torch.cat(all_indices, dim=1)
                values = torch.cat(all_values)
                return torch.sparse_coo_tensor(indices, values,
                                              (batch_size, self.vocab_size))
            else:
                return torch.sparse_coo_tensor(torch.zeros(2, 0, dtype=torch.long),
                                              torch.zeros(0, dtype=torch.long),
                                              (batch_size, self.vocab_size))

    def count_kmers_chunked(self,
                           sequences: torch.Tensor,
                           chunk_size: int = 10000,
                           exclude_n: bool = False) -> torch.Tensor:
        """
        Count k-mers in very large batches using chunking for memory efficiency.

        Args:
            sequences: Tensor of shape (batch, 4, length) with one-hot encoding
            chunk_size: Number of sequences to process at once
            exclude_n: If True, exclude k-mers containing N's from counts

        Returns:
            Tensor of shape (batch, 5^k) with k-mer counts
        """
        batch_size = sequences.shape[0]
        counts = torch.zeros(batch_size, self.vocab_size,
                            dtype=torch.long, device=self.device)

        for i in range(0, batch_size, chunk_size):
            end_idx = min(i + chunk_size, batch_size)
            chunk = sequences[i:end_idx]
            counts[i:end_idx] = self.count_kmers(chunk, exclude_n=exclude_n)

        return counts

    def kmer_to_sequence(self, kmer_idx: int) -> str:
        """
        Convert k-mer index back to sequence string.

        Args:
            kmer_idx: Index of k-mer in base-5 representation

        Returns:
            String representation of k-mer
        """
        bases = ['A', 'T', 'G', 'C', 'N']
        sequence = []

        for _ in range(self.k):
            sequence.append(bases[kmer_idx % 5])
            kmer_idx //= 5

        return ''.join(reversed(sequence))

    def get_top_kmers(self,
                     counts: torch.Tensor,
                     top_k: int = 10,
                     exclude_n: bool = True) -> list:
        """
        Get top k most frequent k-mers from count matrix.

        Args:
            counts: Tensor of shape (batch, 5^k) or (5^k,) with k-mer counts
            top_k: Number of top k-mers to return
            exclude_n: If True, exclude k-mers containing N's from results

        Returns:
            List of tuples (k-mer_string, count) for each batch element
        """
        if counts.dim() == 1:
            counts = counts.unsqueeze(0)

        batch_size = counts.shape[0]
        results = []

        for b in range(batch_size):
            seq_counts = counts[b].clone()

            if exclude_n:
                seq_counts[self.n_mask] = 0

            top_counts, top_indices = torch.topk(seq_counts, min(top_k, seq_counts.shape[0]))

            batch_results = []
            for idx, count in zip(top_indices.cpu().numpy(), top_counts.cpu().numpy()):
                if count > 0:
                    batch_results.append((self.kmer_to_sequence(idx), int(count)))

            results.append(batch_results)

        return results[0] if len(results) == 1 else results


class ProportionalMultiDatasetSampler:
    """
    Samples batches from multiple datasets with probability proportional to dataset size.
    Larger datasets get selected more often for batching.
    """

    def __init__(self,
                 datasets: List[TensorDataset],
                 batch_size: int,
                 shuffle: bool = True,
                 drop_last: bool = False):
        self.datasets = datasets
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        self.dataset_sizes = [len(dataset) for dataset in datasets]
        total_size = sum(self.dataset_sizes)
        self.sampling_probs = [size / total_size for size in self.dataset_sizes]

        self.dataloaders = [
            DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                drop_last=drop_last
            ) for dataset in datasets
        ]

        self.iterators = [iter(dataloader) for dataloader in self.dataloaders]

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, ...]]:
        """
        Yields batches by randomly selecting datasets proportional to their size.
        """
        self.iterators = [iter(dataloader) for dataloader in self.dataloaders]

        active_datasets = list(range(len(self.datasets)))
        active_probs = self.sampling_probs.copy()

        while active_datasets:
            total_prob = sum(active_probs[i] for i in active_datasets)
            if total_prob == 0:
                break

            normalized_probs = [active_probs[i] / total_prob for i in active_datasets]

            dataset_idx = np.random.choice(active_datasets, p=normalized_probs)

            try:
                batch = next(self.iterators[dataset_idx])
                yield batch, dataset_idx

            except StopIteration:
                active_datasets.remove(dataset_idx)
                self.iterators[dataset_idx] = iter(self.dataloaders[dataset_idx])
