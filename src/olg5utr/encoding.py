from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F

DNA_ALPHABET = list("ACGT")
AA_ALPHABET = list("ACDEFGHIKLMNPQRSTVWYX")

DNA_ONEHOT_LOOKUP = np.zeros(128, dtype=np.int8)
DNA_ONEHOT_LOOKUP_MAPPING = np.array([65, 67, 71, 84, 97, 99, 103, 116])
DNA_ONEHOT_LOOKUP[DNA_ONEHOT_LOOKUP_MAPPING] = np.concat([np.arange(4) + 1, np.arange(4) + 1])

AA_ONEHOT_LOOKUP = np.zeros(128, dtype=np.int8)
AA_ONEHOT_LOOKUP_MAPPING = np.array(
    [65, 67, 68, 69, 70, 71, 72, 73, 75, 76, 77, 78, 80, 81, 82, 83, 84, 86, 87, 89, 88]
)
AA_ONEHOT_LOOKUP[AA_ONEHOT_LOOKUP_MAPPING] = np.arange(21) + 1


def to_onehot(
    sequences: str | list[str],
    max_pad_len: int | None,
    alphabet: Literal["DNA", "AA"] = "DNA",
) -> torch.Tensor:
    """Convert nucleotide or amino acid sequences to one-hot encoded tensors.

    Args:
        sequences: A single sequence string or list of sequence strings to encode.
        max_pad_len: Maximum length to pad sequences to. If None, uses the length
            of the longest sequence in the batch. Sequences are right-aligned
            (left-padded).
        alphabet: Type of sequence alphabet. Either 'DNA' for nucleotide sequences
            or 'AA' for amino acid sequences.

    Returns:
        One-hot encoded tensor with shape:
            - Single sequence: ``(sequence_length, num_classes)``
            - Batch of sequences: ``(batch_size, sequence_length, num_classes)``
    """
    if alphabet == "DNA":
        lookup = DNA_ONEHOT_LOOKUP
        lookup_mapping = DNA_ONEHOT_LOOKUP_MAPPING
        num_class = len(DNA_ALPHABET)
    elif alphabet == "AA":
        lookup = AA_ONEHOT_LOOKUP
        lookup_mapping = AA_ONEHOT_LOOKUP_MAPPING
        num_class = len(AA_ALPHABET)

    single_input = isinstance(sequences, str)
    if single_input:
        sequences = [sequences]

    max_len = max_pad_len if max_pad_len is not None else max(len(seq) for seq in sequences)
    batch_size = len(sequences)

    seq_bytes = np.zeros((batch_size, max_len), dtype=np.uint8)

    for i, seq in enumerate(sequences):
        seq_array = np.frombuffer(seq.encode("ascii"), dtype=np.uint8)
        seq_bytes[i, (max_len - len(seq_array)) : max_len] = seq_array

    indices = np.zeros_like(seq_bytes)
    for i in lookup_mapping:
        indices[seq_bytes == i] = lookup[i]

    # Subtract 1 for proper one-hot indexing; 0 becomes padding mask
    indices = indices - 1
    indices_tensor = torch.from_numpy(indices)
    mask = (indices_tensor >= 0) & (indices_tensor < num_class)
    indices_tensor = torch.where(mask, indices_tensor, torch.zeros_like(indices_tensor))
    one_hot = F.one_hot(indices_tensor.long(), num_classes=num_class).to(torch.float32)
    one_hot = one_hot * mask.unsqueeze(-1)

    if single_input:
        return one_hot[0]
    return one_hot


def dna_to_onehot(
    sequences: str | list[str],
    max_pad_len: int | None,
) -> torch.Tensor:
    """Convert DNA sequences to one-hot encoded tensors.

    Convenience wrapper around :func:`to_onehot` with ``alphabet='DNA'``.

    Args:
        sequences: A single DNA sequence string or list of strings.
        max_pad_len: Maximum length to pad sequences to. If None, uses the
            longest sequence length.

    Returns:
        One-hot encoded tensor. See :func:`to_onehot` for shape details.
    """
    return to_onehot(sequences, max_pad_len, alphabet="DNA")


def aa_to_onehot(
    sequences: str | list[str],
    max_pad_len: int | None,
) -> torch.Tensor:
    """Convert amino acid sequences to one-hot encoded tensors.

    Convenience wrapper around :func:`to_onehot` with ``alphabet='AA'``.

    Args:
        sequences: A single amino acid sequence string or list of strings.
        max_pad_len: Maximum length to pad sequences to. If None, uses the
            longest sequence length.

    Returns:
        One-hot encoded tensor. See :func:`to_onehot` for shape details.
    """
    return to_onehot(sequences, max_pad_len, alphabet="AA")


_AA_INDEX = {aa: i for i, aa in enumerate(AA_ALPHABET)}


def build_aa_mask(constraint: str) -> torch.Tensor:
    """Build an acceptable-residue mask from a constraint string.

    Single characters specify exact residues, brackets specify sets of
    acceptable alternatives, and ``X`` means any residue (wildcard).

    Examples::

        "MKA"      → one-hot at M, K, A (3 positions)
        "M[KR]A"   → M at pos 0, K-or-R at pos 1, A at pos 2
        "MXA"      → M at pos 0, any at pos 1, A at pos 2

    Args:
        constraint: Amino acid constraint string using single-letter codes,
            ``[...]`` for degenerate positions, and ``X`` for wildcards.

    Returns:
        Float tensor of shape ``(1, 21, n_positions)``. Values are 1.0 where
        a residue is acceptable, 0.0 otherwise.

    Raises:
        ValueError: On unmatched brackets, empty brackets, or unknown residues.
    """
    positions: list[str] = []
    i = 0
    while i < len(constraint):
        if constraint[i] == "[":
            close = constraint.find("]", i)
            if close == -1:
                msg = f"Unmatched '[' at position {i} in constraint: {constraint!r}"
                raise ValueError(msg)
            group = constraint[i + 1 : close]
            if not group:
                msg = f"Empty brackets at position {i} in constraint: {constraint!r}"
                raise ValueError(msg)
            positions.append(group)
            i = close + 1
        elif constraint[i] == "]":
            msg = f"Unmatched ']' at position {i} in constraint: {constraint!r}"
            raise ValueError(msg)
        else:
            positions.append(constraint[i])
            i += 1

    n_pos = len(positions)
    mask = torch.zeros(1, len(AA_ALPHABET), n_pos)

    for pos_idx, residues in enumerate(positions):
        if residues == "X":
            mask[0, :, pos_idx] = 1.0
        else:
            for aa in residues:
                if aa not in _AA_INDEX:
                    msg = (
                        f"Unknown amino acid {aa!r} at position {pos_idx}"
                        f" in constraint: {constraint!r}"
                    )
                    raise ValueError(msg)
                mask[0, _AA_INDEX[aa], pos_idx] = 1.0

    return mask


def build_right_overhang_mask(dna_seq: str) -> torch.Tensor:
    """Build a boolean one-hot mask for right overhang nucleotides.

    Encodes a short DNA string (the fixed nucleotides past the ORF end)
    as a boolean one-hot tensor in ``(1, 4, length)`` format.

    Args:
        dna_seq: DNA sequence for the right overhang (e.g. ``"TC"``).

    Returns:
        Bool tensor of shape ``(1, 4, len(dna_seq))``.
    """
    onehot = dna_to_onehot(dna_seq, len(dna_seq))  # (len, 4)
    return onehot.transpose(1, 0).unsqueeze(0).to(torch.bool)


def build_seed_onehot(dna_seq: str, seq_length: int) -> torch.Tensor:
    """Build a right-padded one-hot seed sequence.

    The DNA sequence is placed at the left (5' end) and zero-padded on the
    right to ``seq_length``. Used as the seed/reference for edit distance
    loss during optimization.

    Args:
        dna_seq: Seed DNA sequence (typically covers positions before
            the alternative start codon).
        seq_length: Total sequence length to pad to.

    Returns:
        Float tensor of shape ``(1, 4, seq_length)``.

    Raises:
        ValueError: If ``dna_seq`` is longer than ``seq_length``.
    """
    if len(dna_seq) > seq_length:
        msg = f"Seed sequence length {len(dna_seq)} exceeds seq_length {seq_length}"
        raise ValueError(msg)
    onehot = dna_to_onehot(dna_seq, len(dna_seq))  # (len, 4)
    onehot = onehot.transpose(1, 0)  # (4, len)
    pad_len = seq_length - len(dna_seq)
    padded = F.pad(onehot, (0, pad_len))  # (4, seq_length)
    return padded.unsqueeze(0)
