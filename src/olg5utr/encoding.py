import torch
import torch.nn.functional as F

import numpy as np

from typing import List, Optional, Union, Literal

DNA_ALPHABET = list('ACGT')
AA_ALPHABET = list('ACDEFGHIKLMNPQRSTVWYX')

DNA_ONEHOT_LOOKUP = np.zeros(128, dtype=np.int8)
DNA_ONEHOT_LOOKUP_MAPPING = np.array([65, 67, 71, 84, 97, 99, 103, 116])
DNA_ONEHOT_LOOKUP[DNA_ONEHOT_LOOKUP_MAPPING] = np.concat([np.arange(4) + 1, np.arange(4) + 1])

AA_ONEHOT_LOOKUP = np.zeros(128, dtype=np.int8)
AA_ONEHOT_LOOKUP_MAPPING = np.array([65, 67, 68, 69, 70, 71, 72, 73, 75, 76, 77, 78, 80, 81, 82, 83, 84, 86, 87, 89, 88])
AA_ONEHOT_LOOKUP[AA_ONEHOT_LOOKUP_MAPPING] = np.arange(21) + 1

def to_onehot(
    sequences: Union[str, List[str]],
    max_pad_len: Optional[int],
    alphabet: Literal['DNA', 'AA'] = 'DNA'
) -> torch.Tensor:
    """
    Args:
    sequences: A single sequence string or list of sequence strings to encode.
    max_pad_len: Maximum length to pad sequences to. If None, uses the length of the
                longest sequence in the batch. Sequences are right-aligned (left-padded).
    alphabet: Type of sequence alphabet to use. Either 'DNA' for nucleotide sequences
             or 'AA' for amino acid sequences.

    Returns:
        torch.Tensor: One-hot encoded tensor with shape:
            - Single sequence: (sequence_length, num_classes)
            - Batch of sequences: (batch_size, sequence_length, num_classes)
    """
    if alphabet == "DNA":
        lookup = DNA_ONEHOT_LOOKUP
        lookup_mapping = DNA_ONEHOT_LOOKUP_MAPPING
        num_class = len(DNA_ALPHABET)
    elif alphabet == "AA":
        lookup = AA_ONEHOT_LOOKUP
        lookup_mapping = AA_ONEHOT_LOOKUP_MAPPING
        num_class = len(AA_ALPHABET)

    # Handle single sequence case
    single_input = isinstance(sequences, str)
    if single_input:
        sequences = [sequences]

    # Convert sequences to bytes for indexing
    if max_pad_len is not None:
        max_len = max_pad_len
    else:
        max_len = max(len(seq) for seq in sequences)
    batch_size = len(sequences)

    seq_bytes = np.zeros((batch_size, max_len), dtype=np.uint8) # Create a byte matrix

    for i, seq in enumerate(sequences):
        seq_array = np.frombuffer(seq.encode('ascii'), dtype=np.uint8)
        seq_bytes[i, (max_len-len(seq_array)):max_len] = seq_array

    # Apply X_ONEHOT_LOOKUP
    indices = np.zeros_like(seq_bytes)
    for i in lookup_mapping:
        indices[seq_bytes == i] = lookup[i]

    indices = indices - 1 # Subtract 1 to get proper indexing for one-hot encoding, 0 will be padding which we'll mask out
    indices_tensor = torch.from_numpy(indices)
    mask = (indices_tensor >= 0) & (indices_tensor < num_class ) #N becomes zero everywhere
    indices_tensor = torch.where(mask, indices_tensor, torch.zeros_like(indices_tensor)) # Replace -1 indices (padding) with 0
    one_hot = F.one_hot(indices_tensor.long(), num_classes=num_class).to(torch.float32) # Generate one-hot encoding
    one_hot = one_hot * mask.unsqueeze(-1) # Apply mask to zero out padding positions

    if single_input: # Return single sequence without batch dimension if input was a string
        return one_hot[0]
    return one_hot

def dna_to_onehot(
    sequences: Union[str, List[str]],
    max_pad_len: Optional[int]
) -> torch.Tensor:
    return to_onehot(sequences, max_pad_len, alphabet='DNA')

def aa_to_onehot(
    sequences: Union[str, List[str]],
    max_pad_len: Optional[int]
) -> torch.Tensor:
    return to_onehot(sequences, max_pad_len, alphabet='AA')
