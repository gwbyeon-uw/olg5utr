import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from matplotlib import pyplot as plt

from utils import *
from optimus import *
from models import *

from Bio.Data.CodonTable import standard_dna_table

@dataclass
class OptimizationConfig:
    """Configuration for optimization parameters and the defaults"""
    # Batch and sequence parameters
    n_batch: int = 10
    seq_length: int = 100
    alt_start_pos: int = 50 #This would be the 0-based position index for A in ATG
    start_codon: Union[str, list] = 'ATG'
    
    # Gradient optimization parameters
    min_steps: int = 1000 #Minimum GD steps
    max_steps: int = 1000 #Maximum GD steps; must be >min_steps; it keeps iterating up to max_steps or until protein constraints are met
    max_fix_aa_grad_norm_factor: float = 0.5 #Clips fixed protein sequence constraint gradient to this factor times the grad norm of MRL loss
    learning_rate: float = 0.001 #Optimizer configs; AdamW
    beta1: float = 0.9
    beta2: float = 0.999
    weight_decay: float = 0.01
    
    # Simulated annealing parameters; but should operate as greedy by setting max_mutations=1 and tau0=0+eps
    sa_steps: int = 2000
    max_mutations: int = 1
    min_temp: float = 1e-8
    anneal_rate: float = 1e-8
    tau0: float = 1e-8
    
    # Other parameters
    eps: float = 1e-8
    device: str = 'cuda:0'

    #Model paths
    translator_channels: int = 1024
    translator_weights_path: str = './translator_cnn_1024ch.pth'
    optimus_weights_path: str = './optimus_mrl_multi.pth'


class ModelLoader:
    """Handles loading and initialization of models"""
    
    @staticmethod
    def load_translator(channels: int = 1024, state_dict_path: str = None, device: torch.device = None) -> torch.nn.Module:
        """Load and initialize the translator model"""
        translator = Translator(channels)
        translator = translator.to(device)
        translator.load_state_dict(torch.load(state_dict_path))
        return translator
    
    @staticmethod
    def load_optimus_model(device: torch.device = None, state_dict_path: str = None) -> torch.nn.Module:
        """Load and initialize the Optimus MRL model"""
        model_mrl = Optimus(
            inp_len=100, 
            nbr_filters=120, 
            filter_len=8, 
            border_mode='same', 
            dropout1=0.0, 
            dropout2=0.0, 
            dropout3=0.2, 
            nodes=40, 
            out_kw=['egfp_unmod', 'egfp_pseudo', 'egfp_m1pseudo', 'mcherry_unmod'], 
            n_out_col=1
        )
        model_mrl = model_mrl.to(device)
        model_mrl.load_state_dict(torch.load(state_dict_path)['model_state_dict'])
        return model_mrl


class OptimizationResult:
    """Container for optimization results"""
    
    def __init__(self):
        self.best_results = []
        self.history = []
        self.min_losses = []
        self.acceptable_batch = None
    
    def update_best(self, idx: int, sampled: torch.Tensor, mrl1: torch.Tensor, 
                   mrl2: torch.Tensor, step: int, loss: torch.Tensor):
        """Update best result for a specific batch index"""
        while len(self.best_results) <= idx:
            self.best_results.append(None)
            self.min_losses.append(float('inf'))
        
        if loss < self.min_losses[idx]:
            self.min_losses[idx] = loss.detach().clone()
            self.best_results[idx] = [
                sampled.detach().clone(),
                mrl1.detach().clone(),
                mrl2.detach().clone(),
                step
            ]
    
    def add_history(self, mrl1: torch.Tensor, mrl2: torch.Tensor, 
                   loss_prot: torch.Tensor, num_diff: torch.Tensor):
        """Add a step to the history"""
        self.history.append([
            mrl1.detach().clone(),
            mrl2.detach().clone(),
            loss_prot.detach().clone(),
            num_diff.detach().clone()
        ])
    
    def filter_acceptable(self):
        """Filter results to only include acceptable batches; acceptable = fixed protein sequence constraints met"""
        self.acceptable_batch = torch.tensor([
            i for i in range(len(self.best_results)) 
            if self.best_results[i] is not None
        ])

        if self.acceptable_batch.shape[0] > 0:        
            self.history = [
                [h[i][self.acceptable_batch] for i in range(len(h))] 
                for h in self.history
            ]
            self.best_results = [r for r in self.best_results if r is not None]
        else:
            self.history = None
            self.best_results = None


class LossCalculator:
    """Handles loss calculation operations"""
    
    @staticmethod
    def calculate_mrl_loss(mrl1: torch.Tensor, mrl2: torch.Tensor) -> torch.Tensor:
        """Calculate MRL losses"""
        #We could also make this targeted
        mean_mrl = (mrl1 + mrl2) * 0.5
        sqe_mrl = (mrl1 - mrl2) ** 2 #Variance to try to keep them similar
        return mean_mrl * -1.0 + sqe_mrl
    
    @staticmethod
    def calculate_protein_loss(fix_aa: torch.Tensor, prot: torch.Tensor, 
                               eps: float = 1e-8) -> torch.Tensor:
        """Calculate protein constraint loss"""
        return (-1.0 * torch.sum(fix_aa * torch.log(prot + eps), dim=-1)).mean(-1) #Cross entropy vs fixed target


class GradientOptimizer:
    """Handles gradient-based optimization phase"""
    
    def __init__(self, model: torch.nn.Module, config: OptimizationConfig):
        self.model = model
        self.config = config
        self.optimizer = self._create_optimizer()
        self.loss_calc = LossCalculator()
        
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create the optimizer for the model"""
        opt_params = (
            list(self.model.dense.parameters()) + 
            list(self.model.layer_norm_left.parameters()) + 
            list(self.model.layer_norm_right.parameters())
        )
        
        return torch.optim.AdamW(
            opt_params,
            lr=self.config.learning_rate,
            betas=(self.config.beta1, self.config.beta2),
            weight_decay=self.config.weight_decay
        )

    @staticmethod
    @contextmanager
    def masked_gradients(
        parameter: torch.Tensor, 
        mask: torch.Tensor
    ) -> None:
        """
        Context manager that temporarily masks gradients for selective clipping.
        
        Args:
            parameters: parameter tensor
            mask: mask, True is selected subset of gradient
        """
        try:
            original_grad = parameter.grad.data[~mask].clone() # Store original gradients for masked-out regions
            parameter.grad.data[~mask] = 0.0
            yield
                
        finally:
            parameter.grad.data[~mask] = original_grad # Restore original gradients for masked-out regions

    @staticmethod
    def clip_grad_norm_masked(
        parameter: torch.Tensor,
        max_norm: float,
        mask: torch.Tensor,
        norm_type: float = 2.0,
    ) -> torch.Tensor:
        """
        Clip gradients using torch.nn.utils.clip_grad_norm_ but only on masked regions.
        
        Args:
            parameter: parameter
            max_norm: Maximum norm value
            mask: mask, True is selected subset of gradient
            norm_type: Type of norm
            
        Returns:
            Total norm of gradients before clipping
        """
        with GradientOptimizer.masked_gradients(parameter, mask):
            return torch.nn.utils.clip_grad_norm_(
                parameter, max_norm, norm_type
            )

    @staticmethod
    def clip_layer_gradients_by_magnitude(
        layer: nn.Module, 
        bottom_percent: float = 0.1, 
        batch_index: Optional[Union[int, slice]] = None
    ) -> None:
        """
        Simple function to zero out bottom N% of gradients by magnitude for a specific layer.
        
        Args:
            layer: layer with gradients
            bottom_percent: Fraction (0-1) of gradients to zero out
        """
        if hasattr(layer, 'weight') and layer.weight.grad is not None:
            grad = layer.weight.grad[batch_index]
            grad_magnitudes = torch.abs(grad.view(-1))
            
            k = int(len(grad_magnitudes) * bottom_percent) # Find threshold (k-th smallest value)
            if k > 0:
                threshold = torch.kthvalue(grad_magnitudes, k).values
                mask = torch.abs(grad) > threshold # To zero out gradients below threshold
                layer.weight.grad[batch_index] = grad * mask.float() 
        
        if hasattr(layer, 'bias') and layer.bias is not None and layer.bias.grad is not None: # Same for bias if present
            grad = layer.bias.grad[batch_index]
            grad_magnitudes = torch.abs(grad)
            
            k = int(len(grad_magnitudes) * bottom_percent)
            if k > 0:
                threshold = torch.kthvalue(grad_magnitudes, k).values
                mask = grad_magnitudes > threshold
                layer.bias.grad[batch_index] = grad * mask.float()
    
    def _compute_gradients(self, loss_mrl: torch.Tensor, loss_prot: torch.Tensor) -> Dict:
        """Compute and scale gradients"""
        # Store MRL gradients
        grads_mrl = {
            name: torch.zeros_like(param) 
            for name, param in self.model.named_parameters()
        }
        
        # Compute MRL gradients
        self.optimizer.zero_grad()
        loss_mrl.sum().backward(retain_graph=True)
        
        mrl_grad_norm = self.model.dense.weight.grad.data.view( #Grad norm of the MRL to scale by
            (self.model.num_batch, self.model.onehot_dim, self.model.seq_length)
        ).flatten(1).norm(p=2, dim=1) 
        
        for name, param in self.model.named_parameters(): #Save the MRL gradient
            if param.grad is not None:
                grads_mrl[name] += param.grad.clone()
        
        self.optimizer.zero_grad() #Clear the gradient to calculate protein gradient separately
        loss_prot.sum().backward()
        
        mask = torch.zeros(self.model.dense.weight.shape, dtype=torch.bool, 
                          device=self.model.dense.weight.device)
        
        for b in range(self.model.num_batch): #Scaling for each batch independently
            max_grad_norm = self.config.max_fix_aa_grad_norm_factor * mrl_grad_norm[b] #Scales protein grad norm to MRL grad norm
            start_idx = b * self.model.onehot_dim * self.model.seq_length
            end_idx = (b + 1) * self.model.onehot_dim * self.model.seq_length
            
            mask[start_idx:end_idx].fill_(True)
            GradientOptimizer.clip_grad_norm_masked(self.model.dense.weight, max_grad_norm, mask)
            mask[start_idx:end_idx].fill_(False)

        for name, param in self.model.named_parameters(): #Accumulate back the saved MRL gradient atop protein gradient
            if param.grad is not None:
                param.grad += grads_mrl[name]
        
    def optimize(self, fix_aa: torch.Tensor) -> OptimizationResult:
        """Run gradient-based optimization"""
        result = OptimizationResult()
        result.min_losses = [float('inf')] * self.config.n_batch
        result.best_results = [None] * self.config.n_batch
        
        # Get initial random sequence
        with torch.inference_mode():
            sampled, mrl1, mrl2, prot = self.model()
            prev_sample = sampled.detach().clone()
        
        found_result = False
        step = 0
        
        while (not found_result) or (step < self.config.min_steps):
            self.optimizer.zero_grad()
            sampled, mrl1, mrl2, prot = self.model() # Forward pass
            
            # Calculate losses
            loss_mrl = self.loss_calc.calculate_mrl_loss(mrl1, mrl2)
            loss_prot = self.loss_calc.calculate_protein_loss(fix_aa, prot, self.config.eps)
            
            # Compute and apply gradients
            self._compute_gradients(loss_mrl, loss_prot)
            self.optimizer.step()
            
            # Track changes
            num_diff = (((prev_sample - sampled)**2).sum([1, 2]) * 0.5)
            prev_sample = sampled.detach().clone()
            
            # Update best results
            for i in range(self.config.n_batch):
                if (loss_prot[i] == 0.0) and (loss_mrl[i] < result.min_losses[i]):
                    result.update_best(i, sampled[i], mrl1[i], mrl2[i], step, loss_mrl[i])
            
            # Check acceptability = protein constraint met
            acceptable_count = sum(1 for r in result.best_results if r is not None)
            if acceptable_count == self.config.n_batch:
                found_result = True
            
            # Add to history
            result.add_history(mrl1, mrl2, loss_prot, num_diff)
            
            step += 1
            if step >= self.config.max_steps:
                break
        
        return result


class SimulatedAnnealer:
    """Handles simulated annealing optimization phase. Typically just used for greedy sampling with tau0~0 and max_mut=1"""
    
    def __init__(self, model: torch.nn.Module, config: OptimizationConfig):
        self.model = model
        self.config = config
        self.loss_calc = LossCalculator()
        self.syn_mutator = SynonymousMutator(self.config.device)
    
    def _calculate_temperature(self, step: int) -> float:
        """Calculate annealing temperature for current step"""
        return np.maximum(
            self.config.tau0 * np.exp(-self.config.anneal_rate * step), 
            self.config.min_temp
        )

    @staticmethod
    def _anneal(
        loss: Union[float, torch.Tensor], 
        loss_perturbed: Union[float, torch.Tensor], 
        tau: Union[float, torch.Tensor]
    ) -> bool:
        """
        The acceptance rule follows the Metropolis criterion:
        - Always accept improvements (loss_perturbed < loss)
        - Accept deteriorations with probability exp(-ΔE/T) where:
          * ΔE = loss_perturbed - loss (energy increase)
          * T = tau (temperature)
        
        Args:
            loss: Current loss/energy value of the existing state. Lower values represent better solutions.
            loss_perturbed: Loss/energy value of the proposed new state.
            tau: Temperature parameter controlling acceptance probability.
                 - High tau (hot): Accept bad moves more frequently (exploration)
                 - Low tau (cold): Accept bad moves less frequently (exploitation)
                 - tau → 0: Only accept improvements (greedy search)
                 - tau → ∞: Accept all moves (random walk)
        
        Returns:
            bool: True if the new state should be accepted, False otherwise.
        """
        if loss_perturbed < loss: #If decreases loss, always accept
            return True
        else: #If increases loss, accept with a probability modulated by temperature
            ap = torch.exp((loss - loss_perturbed) / tau) #Acceptance probability
            if ap > np.random.rand():
                return True
            else:
                return False #Don't accept

    @staticmethod
    def mutate_inplace(
        sequences: torch.Tensor, 
        num_mutations: int, 
        target_positions: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            sequences: torch.Tensor of shape (B, C, L)
            num_mutations: int, number of positions to mutate per sequence
        
        Returns:
            torch.Tensor: same tensor as input (modified in-place)
        """
        if num_mutations <= 0:
            return sequences
        
        B, C, L = sequences.shape
        device = sequences.device
    
        if target_positions is None:
            positions = torch.randint(0, L, (B, num_mutations), device=device) # Generate random positions and shifts
        else:
            positions = target_positions
        batch_idx = torch.arange(B, device=device).unsqueeze(1)
        current_nucs = torch.argmax(sequences[batch_idx, :, positions], dim=1) # Get current and compute new nucleotides
        new_nucs = (current_nucs + torch.randint(1, C, (B, num_mutations), device=device)) % C    
        sequences[batch_idx, :, positions] = 0 # Apply mutations in-place
        sequences[batch_idx, new_nucs, positions] = 1
        
        return sequences

    @staticmethod
    def shuffle_inplace(tensor: torch.Tensor, num_positions: int) -> torch.Tensor:
        """
        Args:
            tensor: torch.Tensor of shape (B, C, L)
            num_positions: int, number of random positions to shuffle per batch
        
        Returns:
            torch.Tensor: same tensor as input (modified in-place)
        """
        if num_positions <= 0:
            return tensor
        
        B, C, L = tensor.shape
        device = tensor.device
        
        positions = torch.rand(B, L, device=device).argsort(dim=1)[:, :num_positions]
        permutations = torch.rand(B, num_positions, C, device=device).argsort(dim=2)
        
        batch_flat = torch.arange(B, device=device).repeat_interleave(num_positions)
        pos_flat = positions.flatten()
        perm_flat = permutations.view(-1, C)
        
        values = tensor[batch_flat, :, pos_flat]
        tensor[batch_flat, :, pos_flat] = values.gather(1, perm_flat)
        
        return tensor

    def anneal(self, result: OptimizationResult, fix_aa: torch.Tensor, 
              initial_step: int = 0) -> OptimizationResult:
        """Run simulated annealing on the best results"""
        if not result.best_results:
            return result
        
        # Stack best results
        current_input = torch.stack([r[0] for r in result.best_results])
        
        with torch.inference_mode():
            sampled, mrl1, mrl2, prot = self.model(input_onehot=current_input)
            loss_mrl = self.loss_calc.calculate_mrl_loss(mrl1, mrl2)
            loss_prot = self.loss_calc.calculate_protein_loss(fix_aa, prot, self.config.eps)
            loss_total = loss_mrl + loss_prot
            prev_sample = sampled.detach().clone()
            
            last_loss_prot = loss_prot.detach().clone()
            last_loss_total = loss_total.detach().clone()
            
            for sa_step in range(self.config.sa_steps):
                last_input = current_input.clone()
                
                # Mutate one-hot matrix
                num_mut = torch.randint(1, self.config.max_mutations + 1, (1, 1)).item()
                
                if torch.rand(1) * self.model.seq_length < self.model.alt_start:
                    SimulatedAnnealer.mutate_inplace(current_input[:, :, :self.model.alt_start], num_mut)
                else:
                    self.syn_mutator.mutate_batch(current_input[:, :, self.model.alt_start:(self.model.alt_start+3*self.model.aa_len)], num_mut)
                
                # Calculate new losses
                sampled, mrl1, mrl2, prot = self.model(input_onehot=current_input)
                loss_mrl = self.loss_calc.calculate_mrl_loss(mrl1, mrl2)
                loss_prot = self.loss_calc.calculate_protein_loss(fix_aa, prot, self.config.eps)
                loss_total = loss_mrl + loss_prot
                
                # Annealing decisions
                tau = self._calculate_temperature(sa_step)
                
                for i in range(len(result.best_results)):
                    is_swapped = False
                    
                    if loss_prot[i] <= last_loss_prot[i]:
                        is_swapped = SimulatedAnnealer._anneal(last_loss_total[i], loss_total[i], tau)
                    
                    if is_swapped:
                        last_loss_prot[i] = loss_prot[i].detach().clone()
                        last_loss_total[i] = loss_total[i].detach().clone()
                    else:
                        current_input[i] = last_input[i]
                    
                    # Update best if improved
                    if (loss_prot[i] == 0.0) and (loss_mrl[i] < result.min_losses[i]):
                        result.update_best(i, sampled[i], mrl1[i], mrl2[i], 
                                         initial_step + sa_step, loss_mrl[i])
                
                # Track changes
                num_diff = (((prev_sample - sampled)**2).sum([1, 2]) * 0.5)
                prev_sample = sampled.detach().clone()
                
                # Add to history
                result.add_history(mrl1, mrl2, loss_prot, num_diff)
        
        return result


class SynonymousMutator:
    def __init__(self, device: str = 'cpu', codon_table: Optional[dict] = None):
        self.device = device
        self._build_lookup_tables(codon_table)
        
    def _build_lookup_tables(self, codon_table: Optional[dict] = None):
        """Build lookup tensors from codon table."""
        if codon_table is None:
            # Default standard genetic code
            codon_table = self._get_standard_genetic_code()
        
        # Build amino acid to index mapping
        unique_aas = AA_ALPHABET
        self.aa_to_idx = {aa: i for i, aa in enumerate(unique_aas)}
        self.idx_to_aa = {i: aa for aa, i in self.aa_to_idx.items()}
        
        # Convert codon table to tensor format
        nucleotides = DNA_ALPHABET
        codon_to_aa_flat = torch.zeros(64, dtype=torch.long, device=self.device)
        
        for i in range(4):  # First position
            for j in range(4):  # Second position  
                for k in range(4):  # Third position
                    codon = nucleotides[i] + nucleotides[j] + nucleotides[k]
                    codon_idx = i * 16 + j * 4 + k
                    
                    if codon in codon_table:
                        aa = codon_table[codon]
                        codon_to_aa_flat[codon_idx] = self.aa_to_idx[aa]
                    else:
                        # Handle missing codons - assign to unknown AA
                        if 'X' not in self.aa_to_idx:
                            self.aa_to_idx['X'] = len(self.aa_to_idx)
                            self.idx_to_aa[self.aa_to_idx['X']] = 'X'
                        codon_to_aa_flat[codon_idx] = self.aa_to_idx['X']
        
        # Pre-compute synonymous codon mappings
        # For each codon, store indices of synonymous codons
        max_syns = np.unique([ v for v in standard_dna_table.forward_table.values() ], return_counts=True)[1].max().item() - 1  # Maximum synonymous codons for any AA
        self.syn_codons = torch.full((64, max_syns), -1, dtype=torch.long, device=self.device)
        self.syn_counts = torch.zeros(64, dtype=torch.long, device=self.device)
        
        # Group by AA and build synonymous mappings
        num_aas = len(self.aa_to_idx)
        for aa_idx in range(num_aas):
            aa_codons = (codon_to_aa_flat == aa_idx).nonzero(as_tuple=True)[0]
            if len(aa_codons) > 1:
                for i, codon in enumerate(aa_codons):
                    others = aa_codons[aa_codons != codon]
                    self.syn_codons[codon, :len(others)] = others
                    self.syn_counts[codon] = len(others)
        
        # Pre-compute codon decoding tensor (64 x 3)
        self.codon_decode = torch.zeros(64, 3, dtype=torch.long, device=self.device)
        for i in range(64):
            self.codon_decode[i] = torch.tensor([i//16, (i//4)%4, i%4])

    def _get_standard_genetic_code(self) -> dict:
        """Return the standard genetic code as a dictionary."""
        return standard_dna_table.forward_table
        
    def mutate_batch(
        self,
        one_hot: torch.Tensor,
        num_mutations: int, 
    ) -> torch.Tensor:
        
        B, C, L = one_hot.shape
        num_codons = L // 3
        
        # Convert to codon indices in one vectorized operation
        nuc_indices = torch.argmax(one_hot, dim=1)  # (B, L)
        codon_indices = (nuc_indices[:, 0::3] * 16 + 
                        nuc_indices[:, 1::3] * 4 + 
                        nuc_indices[:, 2::3])  # (B, num_codons)
        
        # Find all mutable positions across batch
        batch_syn_counts = self.syn_counts[codon_indices]  # (B, num_codons)
        mutable_mask = batch_syn_counts > 0  # (B, num_codons)
        
        result = one_hot #In place!
        
        # Process all batches simultaneously
        for b in range(B):
            mutable_pos = torch.where(mutable_mask[b])[0]
            
            if len(mutable_pos) == 0:
                continue
                
            # Select random positions to mutate
            n_mut = min(num_mutations, len(mutable_pos))
            if n_mut == 0:
                continue
                
            # Vectorized random selection
            perm = torch.randperm(len(mutable_pos), device=one_hot.device)
            selected_pos = mutable_pos[perm[:n_mut]]
            
            # Vectorized mutation application
            for pos in selected_pos:
                codon_idx = codon_indices[b, pos]
                n_syns = self.syn_counts[codon_idx]
                
                if n_syns > 0:
                    # Select random synonym
                    syn_idx = torch.randint(n_syns, (1,), device=one_hot.device)[0]
                    new_codon_idx = self.syn_codons[codon_idx, syn_idx]
                    new_codon = self.codon_decode[new_codon_idx]
                    
                    # Update one-hot (vectorized)
                    start_pos = pos * 3
                    result[b, :, start_pos:start_pos+3] = 0
                    result[b, new_codon, torch.arange(start_pos, start_pos+3, device=one_hot.device)] = 1
        
        return result
        

class OptimusOLGPipeline:
    """Main pipeline for running the complete optimization"""
    
    def __init__(self, config: OptimizationConfig = None):
        self.config = config or OptimizationConfig()
        self.device = torch.device(self.config.device)
        
        # Load models
        self.translator = ModelLoader.load_translator(channels=self.config.translator_channels, 
                                                      device=self.device, state_dict_path=self.config.translator_weights_path)
        self.model_mrl = ModelLoader.load_optimus_model(device=self.device, state_dict_path=self.config.optimus_weights_path)
        
        # Initialize constants
        if isinstance(self.config.start_codon, str):
            self.start_codon = dna_to_onehot(self.config.start_codon, 3).transpose(1, 0).unsqueeze(0).repeat(self.config.n_batch, 1, 1) #Using same start for all batch samples
        else:
            self.start_codon = dna_to_onehot(self.config.start_codon, 3).transpose(1, 0)
            
    def run(self, fix_aa: torch.Tensor = None, right_overhang_mask: torch.Tensor = None) -> OptimizationResult:
        """Run the complete optimization pipeline"""       
        # Phase 1: Gradient-based optimization
        print("Starting gradient-based optimization...")
        model = OptimusOLG(
            self.device, self.model_mrl, self.translator, 
            self.config.n_batch, self.config.seq_length, 
            self.config.alt_start_pos, self.start_codon, 
            right_overhang_mask
        ).to(self.device)
        
        grad_optimizer = GradientOptimizer(model, self.config)
        result = grad_optimizer.optimize(fix_aa)
        
        # Filter acceptable results
        result.filter_acceptable()

        if result.best_results is None:
            print("No acceptable results found in gradient optimization")
            return result
        
        # Phase 2: Simulated annealing
        print(f"Starting simulated annealing with {len(result.best_results)} sequences...")
        model = OptimusOLG(
            self.device, self.model_mrl, self.translator, 
            len(result.best_results), self.config.seq_length, 
            self.config.alt_start_pos, self.start_codon, 
            right_overhang_mask
        ).to(self.device)
        
        annealer = SimulatedAnnealer(model, self.config)
        result = annealer.anneal(
            result, 
            fix_aa[result.acceptable_batch], 
            self.config.max_steps
        )
        
        print(f"Optimization complete. Found {len(result.best_results)} optimized sequences.")
        return result
    
    def plot_results(self, result: OptimizationResult):
        """Plot optimization history"""
        if not result.history:
            print("No history to plot")
            return
        
        history = torch.stack([torch.stack(h) for h in result.history])
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot MRL1
        axes[0, 0].plot(history[:, 0].cpu().numpy())
        axes[0, 0].set_title('MRL1 over iterations')
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('MRL1')
        
        # Plot MRL2
        axes[0, 1].plot(history[:, 1].cpu().numpy())
        axes[0, 1].set_title('MRL2 over iterations')
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('MRL2')
        
        # Plot Protein Loss
        axes[1, 0].plot(history[:, 2].cpu().numpy())
        axes[1, 0].set_title('Protein Loss over iterations')
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('Loss')
        
        # Plot Number of Differences
        axes[1, 1].plot(history[:, 3].cpu().numpy())
        axes[1, 1].set_title('Sequence Changes over iterations')
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('Number of differences')
        
        plt.tight_layout()
        plt.show()

