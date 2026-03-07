import torch
import pytest

from olg5utr.models import STArgmaxSoftmaxGeneric, Translator, Optimus, OptimusOLG
from olg5utr.encoding import dna_to_onehot


class TestSTArgmaxSoftmax:
    def test_output_is_onehot(self):
        st = STArgmaxSoftmaxGeneric(4)
        logits = torch.randn(2, 4, 10)
        out = st(logits)
        # Each position should be one-hot
        assert torch.all(out.sum(dim=1) == 1.0)
        assert set(out.unique().tolist()) == {0.0, 1.0}

    def test_shape_preserved(self):
        st = STArgmaxSoftmaxGeneric(4)
        logits = torch.randn(3, 4, 20)
        out = st(logits)
        assert out.shape == logits.shape

    def test_gradient_flows(self):
        st = STArgmaxSoftmaxGeneric(4)
        logits = torch.randn(2, 4, 10, requires_grad=True)
        out = st(logits)
        loss = out.sum()
        loss.backward()
        assert logits.grad is not None
        assert not torch.all(logits.grad == 0)


class TestTranslator:
    @pytest.fixture
    def translator(self):
        return Translator(n_channel=64)  # Small for testing

    def test_output_count(self, translator):
        x = torch.randn(2, 4, 30)
        out = translator(x)
        assert len(out) == 6  # 3 forward + 3 reverse frames

    def test_output_shapes(self, translator):
        x = torch.randn(2, 4, 30)
        out = translator(x)
        for frame_out in out:
            assert frame_out.shape[0] == 2  # batch
            assert frame_out.shape[1] == 21  # 20 AA + stop

    def test_output_is_onehot(self, translator):
        x = torch.randn(2, 4, 30)
        out = translator(x)
        for frame_out in out:
            # Each position should sum to 1 (one-hot via ST estimator)
            assert torch.allclose(frame_out.sum(dim=1), torch.ones_like(frame_out.sum(dim=1)))

    def test_gradient_flows(self, translator):
        x = torch.randn(2, 4, 30, requires_grad=True)
        out = translator(x)
        loss = sum(o.sum() for o in out)
        loss.backward()
        assert x.grad is not None


class TestOptimus:
    @pytest.fixture
    def model(self):
        return Optimus(
            inp_len=100, nbr_filters=32, filter_len=8,
            border_mode='same', dropout1=0.0, dropout2=0.0,
            dropout3=0.0, nodes=16,
            out_kw=['head_a', 'head_b'], n_out_col=1
        )

    def test_output_shape(self, model):
        x = torch.randn(4, 4, 100)
        out = model(x, final_ind=0)
        assert out.shape == (4, 1)

    def test_multiple_heads(self, model):
        x = torch.randn(4, 4, 100)
        out_a = model(x, final_ind=0)
        out_b = model(x, final_ind=1)
        # Different heads should give different outputs (different random weights)
        assert not torch.allclose(out_a, out_b)

    def test_keyword_head_selection(self, model):
        x = torch.randn(4, 4, 100)
        out_ind = model(x, final_ind=0)
        out_kw = model(x, final_ind=None, final_kw='head_a')
        assert torch.allclose(out_ind, out_kw)

    def test_gradient_flows(self, model):
        x = torch.randn(2, 4, 100, requires_grad=True)
        out = model(x, final_ind=0)
        out.sum().backward()
        assert x.grad is not None


class TestOptimusOLG:
    @pytest.fixture
    def setup(self):
        device = torch.device('cpu')
        model = Optimus(
            inp_len=100, nbr_filters=32, filter_len=8,
            border_mode='same', dropout1=0.0, dropout2=0.0,
            dropout3=0.0, nodes=16,
            out_kw=['head_a'], n_out_col=1
        )
        translator = Translator(n_channel=64)
        start_codon = dna_to_onehot("ATG", 3).transpose(1, 0).unsqueeze(0).repeat(2, 1, 1)
        right_mask = torch.ones(2, 4, 2, dtype=torch.bool)

        olg = OptimusOLG(
            device=device, model=model, translator=translator,
            num_batch=2, seq_length=100, alt_start=50,
            start_codon=start_codon, right_overhang_mask=right_mask
        )
        return olg

    def test_output_shapes(self, setup):
        sampled, mrl1, mrl2, prot = setup()
        assert sampled.shape == (2, 4, 100)
        assert mrl1.shape == (2,)
        assert mrl2.shape == (2,)

    def test_start_codon_forced(self, setup):
        sampled, _, _, _ = setup()
        # Position 50-52 should be ATG
        atg_onehot = dna_to_onehot("ATG", 3).transpose(1, 0)
        for b in range(2):
            assert torch.allclose(sampled[b, :, 50:53], atg_onehot)

    def test_sampled_is_onehot(self, setup):
        sampled, _, _, _ = setup()
        assert torch.allclose(sampled.sum(dim=1), torch.ones(2, 100))

    def test_gradient_flows_through_weight(self, setup):
        sampled, mrl1, mrl2, prot = setup()
        loss = mrl1.sum() + mrl2.sum()
        loss.backward()
        assert setup.weight.grad is not None

    def test_passthrough_mode(self, setup):
        input_oh = torch.zeros(2, 4, 100)
        input_oh[:, 0, :] = 1.0  # All A's
        input_oh[:, :, 50:53] = 0  # Will be overwritten by ATG
        sampled, mrl1, mrl2, prot = setup(input_onehot=input_oh)
        # Should use the provided input (with ATG forced)
        assert sampled.shape == (2, 4, 100)
