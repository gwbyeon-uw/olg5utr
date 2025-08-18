import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from typing import Optional, Tuple, Dict, Any, Union, List

from contextlib import contextmanager

class Translator(nn.Module):
    """
    Differentiable nucleotide-to-protein translation with convolutional filters. 
    Used to let optimizing nucleotide sequence when loss function is with translated protein sequences
    It processes all 6 possible reading frames (3 frames × 2 strands)

    1. Use 3nt convolutional filters to extract "codons"
    2. Generate 3 reading frames by shifting codon features (+0, +1, +2)
    3. Convolutional filter to map "codons" to AA both forward and reverse strands
    4. Straight-through estimation for discrete amino acid sampling
    """
    def __init__(self, n_channel: int = 1024) -> None:
        super(Translator, self).__init__()        
        self.n_channel = n_channel #High values >512 tend to make for better gradients
        self.n_nuc = 4 #ACGT
        self.n_aa = 21 #20 AA alphabet + stop
        
        self.codon = nn.Conv1d(self.n_nuc, self.n_channel, kernel_size=3, stride=1, bias=False) #Kernel size=3 for triplet codons, stride=1 for sliding window
        self.aa_withstop_f = nn.Conv1d(self.n_channel, self.n_aa, kernel_size=1, stride=3, bias=False) #Kernel size=1 for mapping each codon to aa; stride 3 for non-overlapping triplets
        self.aa_withstop_r = nn.Conv1d(self.n_channel, self.n_aa, kernel_size=1, stride=3, bias=False) #negative strand
        self.stargsoftmax_withstop = STArgmaxSoftmaxGeneric(self.n_aa) # Straight-through estimator for discrete amino acid sampling
        
    def forward(
        self, 
        input: torch.Tensor, 
        temperature: float = 1.0
    ) -> List[torch.Tensor]:
        """
        Args:
            input: One-hot encoded nucleotide sequence; [batch_size, 4, sequence_length]
            temperature: Sampling temperature
        
        Returns:
            List of 6 tensors, each representing amino acid sequences:
                - Indices 0-2: Forward strand, reading frames +0, +1, +2
                - Indices 3-5: Reverse strand, reading frames -0, -1, -2
                Each tensor shape: [batch_size, 21, protein_length]
                where protein_length = (sequence_length - frame_offset - 2) // 3
        """
        codon = self.codon(input) #[batch, 4, sequence_length] -> [batch, n_channel, sequence_length-2] ("codon" features)
        codon_frames = [ codon, codon[:, :, 1:], codon[:, :, 2:] ] #Shift +0, +1, +2 for alignment
        aa_withstop = [ self.aa_withstop_f(c) for c in codon_frames ] + [ self.aa_withstop_r(c) for c in codon_frames ] #[batch, n_channel, sequence_length] -> [batch, 21, (sequence_length-frame-2)//3]; "codon" features -> amino acids
        aa_withstop_temperature = [ aa * temperature for aa in aa_withstop ] #Temperature in case we want to use sampling
        sampled_withstop = [ self.stargsoftmax_withstop(aa) for aa in aa_withstop_temperature ] #Sample
        
        return sampled_withstop

class Optimus(nn.Module):
    """
    This is to retrain the Optimus 5' prime model.
    Original Keras implementation: 
        https://github.com/pjsample/human_5utr_modeling/blob/master/modeling/training_MRL_CNN.ipynb

    Minor architecture modifications:
    -Shared convolution weights with multiple dense heads; jointly trains on all datasets (modified RNAs, mCherry, variable sizes for each head)
    """
    def __init__(
        self,
        inp_len: int,
        nbr_filters: int,
        filter_len: int,
        border_mode: Union[int, str],
        dropout1: float,
        dropout2: float,
        dropout3: float,
        nodes: int,
        out_kw: List[str],
        n_out_col: int
    ) -> None:
        """
        Args:
            inp_len: 50 or 100 in Optimus
            nbr_filters: 120 in Optimus
            filter_len: 8 in Optimus
            border_mode: 'same' in Optimus
            dropout1: 0.0 in Optimus
            dropout2: 0.0 in Optimus
            dropout3: 0.2 in Optimus
            nodes: 40 in Optimus
            out_kw: List of output keywords/names for each prediction head.
            n_out_col: Number of output units for each prediction head 
        """
        super(Optimus, self).__init__()
        self.n_out = len(out_kw)
        self.out_ind = { k:v for v, k in enumerate(out_kw) }
        
        self.conv = nn.Sequential(
            nn.Conv1d(4, nbr_filters, filter_len, padding=border_mode),
            nn.ReLU(),
            nn.Conv1d(nbr_filters, nbr_filters, filter_len, padding=border_mode),
            nn.ReLU(),
            nn.Dropout(dropout1),
            nn.Conv1d(nbr_filters, nbr_filters, filter_len, padding=border_mode),
            nn.ReLU(),
            nn.Dropout(dropout2)
        )

        with torch.no_grad():
            dummy_input = torch.zeros(1, 4, inp_len)
            x = self.conv(dummy_input)
            self.conv_output_size  = x.shape[1] * x.shape[2]

        self.head = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(self.conv_output_size, nodes),
                    nn.ReLU(),
                    nn.Dropout(dropout3),
                    nn.Linear(nodes, n_out_col)
                ) 
                for _ in range(self.n_out)
            ]
        )

    def forward(self, x, final_ind=0, final_kw=None):
        """
        Args:
            x: One hot sequence tensor of shape (B, C=4, L)
            final_ind: Index of the prediction head; if None and final_kw is also None, uses head 0. Ignored if final_kw is provided.
            final_kw: Keyword name of the prediction head to use. Takes precedence over final_ind if both are provided.
        
        Returns:
            torch.Tensor: (batch_size, n_out_col).
        """
        x = self.conv(x)

        #Add multi head to jointly train all data with shared conv 
        if final_ind is not None:
            x = self.head[final_ind](x)
        elif final_kw is not None:
            x = self.head[self.out_ind[final_kw]](x)
        
        return x

class OptimusOLG(nn.Module):
    """
    Helper class to optimize 5'UTR sequences for dual expression from alternative start codons
    SeqProp style algorithm https://doi.org/10.1186/s12859-021-04437-5

    1. Normalized weights => logits for sampling nucleotide sequences
    2. Force start codon at the specified position
    3. Returns: 1) Two MRLs for two 5'UTRs (where one is slice of other) using Optimus 5 prime; 2) amino acid sequence for the overlapping region

    Differentiable for gradient descent optimization
    """
    def __init__(
        self,
        device: torch.device,
        model: nn.Module,
        translator: nn.Module,
        num_batch: int,
        seq_length: int,
        alt_start: int,
        start_codon: torch.Tensor,
        right_overhang_mask: torch.Tensor,
        seed_onehot: Optional[torch.Tensor] = None
#        initial_weights: Optional[torch.Tensor]
    ) -> None:
        """
        Args:
            device: CPU/GPU
            model: Optimus 5' model for MRL prediction
            translator: Nuc->AA translation module
            num_batch: Number of sequences to predict in parallel
            seq_length: Total length of each nucleotide sequence
            alt_start: Position where start codon should be inserted
            start_codon: Start codon sequence (i.e. ATG) in one-hot encoded format (1xCxL)
        """
        super(OptimusOLG, self).__init__()
        
        self.stsampler = STArgmaxSoftmaxGeneric(4) #Straight-through estimator (Softmax backward, Argmax forward)
        self.model = model
        self.translator = translator
        self.num_batch = num_batch
        self.onehot_dim = 4 #ACGT
        self.seq_length = seq_length
        self.alt_start = alt_start
        self.aa_len = (self.seq_length - self.alt_start) // 3 - 1
        self.device = device
        self.eps = 1e-8
        self.mask_val = -1e8

        self.weight = nn.Parameter(torch.zeros(self.num_batch, self.onehot_dim, self.seq_length))
        self.seed_onehot = seed_onehot
        if self.seed_onehot is None:
            torch.nn.init.xavier_normal_(self.weight) #glorot initialization
            
        self.layer_norm_seed_left = nn.LayerNorm((self.onehot_dim, self.alt_start))
        self.layer_norm_seed_right = nn.LayerNorm((self.onehot_dim, self.seq_length - self.alt_start))
        self.layer_norm_left = nn.LayerNorm((self.onehot_dim, self.alt_start))
        self.layer_norm_right = nn.LayerNorm((self.onehot_dim, self.seq_length - self.alt_start))

        self.model.eval()
        self.translator.eval()

        self.start_codon = start_codon.to(torch.float32).to(self.device)
        self.right_overhang_mask = right_overhang_mask.to(self.device)
        self.right_end = self.seq_length - self.right_overhang_mask.shape[-1]
        
    def forward(
        self, 
        input_onehot: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            input_onehot: Optional pre-computed one-hot nucleotide sequences
                         If None, sequences are generated from learned parameters
                         Shape: [batch_size, 4, seq_length]
        
        Returns:
            Tuple containing:
                - sampled: One-hot encoded nucleotide sequences [batch, 4, seq_length]
                - mrl1: MRL prediction for downstream start [batch]
                - mrl2: MRL prediction for upstream start [batch]  
                - prot_f2: AA sequence for upstream protein [batch, 21, seq_length]
        """
        if input_onehot is None:
            weight = self.weight
            normed_left = self.layer_norm_left(weight[:, :, :self.alt_start])  #Normalization to stabilize logit scaling and influences sampling entropy; separately for region that is / is not constrained by proteins; Bx4x(start_codon_pos)
            normed_right = self.layer_norm_right(weight[:, :, self.alt_start:]) #Bx4x(L-start_codon_pos)
            normed = torch.cat([ normed_left, normed_right ], dim=2)
            normed[:, :, self.right_end:] = normed[:, :, self.right_end:] + (~self.right_overhang_mask).float() * self.mask_val
            if self.seed_onehot is not None:
                seed = torch.log(self.seed_onehot + self.eps)
                normed_seed_left = self.layer_norm_seed_left(seed[:, :, :self.alt_start])
                normed_seed_right = self.layer_norm_seed_right(seed[:, :, self.alt_start:])
                normed_seed = torch.cat([ normed_seed_left, normed_seed_right ], dim=2)
                normed = normed + normed_seed
            sampled = self.stsampler(normed) #Argmax forward pass; softmax backward Bx4xL
        else: #Use provided one-hot instead of sampling
            sampled = input_onehot #Bx4xL
        sampled[:, :, self.alt_start:(self.alt_start+3)] = self.start_codon #Force start codon
        sampled_slice1 = F.pad(sampled[:, :, :self.alt_start], (self.alt_start, 0)) #Nucleotides for upstream start Bx4xL
        sampled_slice2 = sampled[:, :, self.alt_start:] #Nucleotides for downstream start Bx4xL
        withstop_slice2 = self.translator(sampled_slice2) #Translation for upstream start Bx21xL
        prot_f2 = withstop_slice2[0][:, :, 1:] #Bx21x(L-1); first is always start codon
        
        mrl1 = self.model(sampled, final_ind=0).squeeze(1) #MRL for upstream start [B]
        mrl2 = self.model(sampled_slice1, final_ind=0).squeeze(1) #MRL from downstream start [B]

        return sampled, mrl1, mrl2, prot_f2

#Argmax forward, softmax backward
class STArgmaxSoftmaxGeneric(nn.Module):
    """
    Straight-Through Estimator for Argmax with Softmax gradients
    Forward Pass:  argmax(softmax(logits))
    Backward Pass: gradients flow through softmax(logits)
    """
    def __init__(self, onehot_dim: int) -> None:
        super(STArgmaxSoftmaxGeneric, self).__init__()
        self.onehot_dim = onehot_dim
    
    def forward(self, x): #X should be logits
        softmax_seq = F.softmax(x, dim=1) #[batch_size, onehot_dim, sequence_length]
        argmax_seq = 1.0 * F.one_hot(torch.argmax(softmax_seq, 1), self.onehot_dim).permute([0, 2, 1]) #[batch_size, sequence_length]
        ret = argmax_seq - softmax_seq.detach() + softmax_seq #softmax.detach(): breaks gradient flow
        return ret