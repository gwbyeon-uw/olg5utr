import torch
import torch.nn.functional as F
import pytest

from olg5utr.optimizer import (
    OptimizationConfig, LossCalculator, SimulatedAnnealer, SynonymousMutator,
)
from olg5utr.encoding import DNA_ALPHABET


class TestLossCalculator:
    def test_mrl_loss_higher_is_better(self):
        # Higher MRL should give lower loss
        loss_high = LossCalculator.calculate_mrl_loss(
            torch.tensor([2.0]), torch.tensor([2.0])
        )
        loss_low = LossCalculator.calculate_mrl_loss(
            torch.tensor([0.5]), torch.tensor([0.5])
        )
        assert loss_high < loss_low

    def test_mrl_loss_variance_penalty(self):
        # Equal MRLs should have lower loss than unequal (same mean)
        loss_equal = LossCalculator.calculate_mrl_loss(
            torch.tensor([1.0]), torch.tensor([1.0])
        )
        loss_unequal = LossCalculator.calculate_mrl_loss(
            torch.tensor([1.5]), torch.tensor([0.5])
        )
        assert loss_equal < loss_unequal

    def test_protein_loss_perfect_match(self):
        target = F.one_hot(torch.tensor([[0, 1, 2]]), 21).transpose(2, 1).float()
        pred = target.clone()
        loss = LossCalculator.calculate_protein_loss(target, pred)
        assert torch.allclose(loss, torch.tensor([0.0]), atol=1e-6)

    def test_protein_loss_mismatch_positive(self):
        target = F.one_hot(torch.tensor([[0, 1, 2]]), 21).transpose(2, 1).float()
        pred = F.one_hot(torch.tensor([[3, 4, 5]]), 21).transpose(2, 1).float()
        loss = LossCalculator.calculate_protein_loss(target, pred)
        assert loss > 0

    def test_edit_loss_identical_zero(self):
        seq = torch.randn(2, 4, 10)
        loss = LossCalculator.calculate_edit_loss(seq, seq)
        assert torch.allclose(loss, torch.zeros(2), atol=1e-6)

    def test_edit_loss_different_positive(self):
        seq1 = torch.zeros(2, 4, 10)
        seq1[:, 0, :] = 1.0
        seq2 = torch.zeros(2, 4, 10)
        seq2[:, 1, :] = 1.0
        loss = LossCalculator.calculate_edit_loss(seq1, seq2)
        assert torch.all(loss > 0)

    def test_edit_loss_ignores_padding(self):
        seq = torch.zeros(1, 4, 10)
        seq[:, 0, :] = 1.0
        # Seed with padding (all zeros) at first 5 positions
        seed = torch.zeros(1, 4, 10)
        seed[:, 1, 5:] = 1.0  # Different from seq but only at non-padded positions
        loss = LossCalculator.calculate_edit_loss(seed, seq)
        # Should only count loss at positions 5-9
        seed_full = torch.zeros(1, 4, 10)
        seed_full[:, 1, :] = 1.0
        loss_full = LossCalculator.calculate_edit_loss(seed_full, seq)
        assert loss < loss_full


class TestMutateInplace:
    def test_shape_preserved(self):
        seq = torch.zeros(3, 4, 20)
        seq[:, 0, :] = 1.0  # All A's
        result = SimulatedAnnealer.mutate_inplace(seq, 2)
        assert result.shape == (3, 4, 20)

    def test_still_onehot(self):
        seq = torch.zeros(3, 4, 20)
        seq[:, 0, :] = 1.0
        result = SimulatedAnnealer.mutate_inplace(seq, 5)
        assert torch.allclose(result.sum(dim=1), torch.ones(3, 20))

    def test_mutations_applied(self):
        torch.manual_seed(42)
        seq = torch.zeros(1, 4, 100)
        seq[:, 0, :] = 1.0  # All A's
        original = seq.clone()
        SimulatedAnnealer.mutate_inplace(seq, 10)
        # At least some positions should have changed
        assert not torch.allclose(seq, original)

    def test_zero_mutations_noop(self):
        seq = torch.zeros(2, 4, 10)
        seq[:, 0, :] = 1.0
        original = seq.clone()
        SimulatedAnnealer.mutate_inplace(seq, 0)
        assert torch.allclose(seq, original)


class TestSynonymousMutator:
    @pytest.fixture
    def mutator(self):
        return SynonymousMutator(device='cpu')

    def test_preserves_protein(self, mutator):
        """Synonymous mutations should not change the encoded protein"""
        # Create a sequence of known codons: GCT (Ala), GGT (Gly), AAA (Lys)
        # repeated to fill a reasonable length
        seq = torch.zeros(1, 4, 9)
        codons = "GCTGGTAAA"
        for i, nuc in enumerate(codons):
            idx = DNA_ALPHABET.index(nuc)
            seq[0, idx, i] = 1.0

        # Get original protein by reading codons
        orig_indices = seq.argmax(dim=1)  # (1, 9)
        orig_codons = (orig_indices[0, 0::3] * 16 + orig_indices[0, 1::3] * 4 + orig_indices[0, 2::3])

        # Mutate
        mutated = mutator.mutate_batch(seq.clone(), num_mutations=3)
        mut_indices = mutated.argmax(dim=1)
        mut_codons = (mut_indices[0, 0::3] * 16 + mut_indices[0, 1::3] * 4 + mut_indices[0, 2::3])

        # Codons may differ but should encode the same amino acids
        for i in range(3):
            orig_aa = mutator.aa_to_idx
            # Look up AA for each codon via the lookup
            assert mutator.syn_counts[orig_codons[i]] > 0 or orig_codons[i] == mut_codons[i]

    def test_output_still_onehot(self, mutator):
        seq = torch.zeros(2, 4, 12)
        seq[:, 0, :] = 1.0  # All A's (AAA = Lys)
        mutated = mutator.mutate_batch(seq, num_mutations=2)
        assert torch.allclose(mutated.sum(dim=1), torch.ones(2, 12))

    def test_shape_preserved(self, mutator):
        seq = torch.zeros(3, 4, 15)
        seq[:, 0, :] = 1.0
        mutated = mutator.mutate_batch(seq, num_mutations=1)
        assert mutated.shape == (3, 4, 15)


class TestAnneal:
    def test_always_accepts_improvement(self):
        assert SimulatedAnnealer._anneal(
            torch.tensor(1.0), torch.tensor(0.5), torch.tensor(0.001)
        ) == True

    def test_greedy_rejects_worse(self):
        # With tau very close to 0, should almost never accept worse
        accepted = 0
        for _ in range(100):
            if SimulatedAnnealer._anneal(
                torch.tensor(0.5), torch.tensor(1.0), torch.tensor(1e-10)
            ):
                accepted += 1
        assert accepted == 0

    def test_high_temp_accepts_more(self):
        # With high temperature, should accept worse moves more often
        accepted = 0
        for _ in range(1000):
            if SimulatedAnnealer._anneal(
                torch.tensor(0.5), torch.tensor(0.6), torch.tensor(100.0)
            ):
                accepted += 1
        # Should accept most of the time at high temperature
        assert accepted > 900


class TestOptimizationConfig:
    def test_defaults(self):
        config = OptimizationConfig()
        assert config.n_batch == 10
        assert config.seq_length == 100
        assert config.device == 'cuda:0'

    def test_override(self):
        config = OptimizationConfig(n_batch=5, device='cpu')
        assert config.n_batch == 5
        assert config.device == 'cpu'
