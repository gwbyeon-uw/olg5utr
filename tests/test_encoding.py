import torch
import pytest

from olg5utr.encoding import (
    DNA_ALPHABET, AA_ALPHABET,
    to_onehot, dna_to_onehot, aa_to_onehot,
)


class TestDNAOnehot:
    def test_single_sequence_shape(self):
        oh = dna_to_onehot("ACGT", 4)
        assert oh.shape == (4, 4)

    def test_single_sequence_identity(self):
        oh = dna_to_onehot("ACGT", 4)
        # Each position should be a one-hot vector matching the nucleotide index
        for i, nuc in enumerate("ACGT"):
            expected_idx = DNA_ALPHABET.index(nuc)
            assert oh[i, expected_idx] == 1.0
            assert oh[i].sum() == 1.0

    def test_batch_shape(self):
        oh = dna_to_onehot(["ACGT", "TGCA"], 4)
        assert oh.shape == (2, 4, 4)

    def test_padding_right_aligned(self):
        oh = dna_to_onehot("AC", 4)
        # "AC" padded to length 4 should be [0, 0, A, C] (right-aligned)
        assert oh.shape == (4, 4)
        # First two positions should be all zeros (padding)
        assert oh[0].sum() == 0.0
        assert oh[1].sum() == 0.0
        # Last two should be valid
        assert oh[2].sum() == 1.0
        assert oh[3].sum() == 1.0

    def test_variable_length_batch(self):
        oh = dna_to_onehot(["AC", "ACGT"], None)
        assert oh.shape == (2, 4, 4)  # Padded to max length
        # First sequence should have 2 padding positions
        assert oh[0, 0].sum() == 0.0
        assert oh[0, 1].sum() == 0.0

    def test_roundtrip(self):
        seq = "ACGTACGT"
        oh = dna_to_onehot(seq, len(seq))
        decoded = ''.join(DNA_ALPHABET[i] for i in oh.argmax(dim=1))
        assert decoded == seq

    def test_case_insensitive(self):
        oh_upper = dna_to_onehot("ACGT", 4)
        oh_lower = dna_to_onehot("acgt", 4)
        assert torch.allclose(oh_upper, oh_lower)


class TestAAOnehot:
    def test_single_sequence_shape(self):
        oh = aa_to_onehot("ACDE", 4)
        assert oh.shape == (4, 21)

    def test_roundtrip(self):
        seq = "ACDEFGHIKLMNPQRSTVWYX"
        oh = aa_to_onehot(seq, len(seq))
        decoded = ''.join(AA_ALPHABET[i] for i in oh.argmax(dim=1))
        assert decoded == seq

    def test_batch(self):
        oh = aa_to_onehot(["ACD", "EFG"], 3)
        assert oh.shape == (2, 3, 21)

    def test_dtype(self):
        oh = dna_to_onehot("ACGT", 4)
        assert oh.dtype == torch.float32
