from dataclasses import asdict, dataclass, field, fields
from importlib import resources
from pathlib import Path

import yaml

BASE_CONFIG_PATH = resources.files("olg5utr") / "config_base.yaml"


@dataclass
class ModelConfig:
    """Model architecture and weight paths.

    Attributes:
        translator_channels: Number of intermediate channels for codon features.
        translator_weights_path: Path to Translator model weights.
        optimus_weights_path: Path to Optimus MRL model weights.
    """

    translator_channels: int = 1024
    translator_weights_path: str = "./weights/translator_cnn_1024ch.pth"
    optimus_weights_path: str = "./weights/optimus_mrl_multi.pth"


@dataclass
class GDConfig:
    """Gradient descent phase configuration.

    Attributes:
        min_steps: Minimum gradient descent steps before early exit.
        max_steps: Maximum gradient descent steps (hard stop).
        learning_rate: AdamW learning rate.
        beta1: AdamW beta1.
        beta2: AdamW beta2.
        weight_decay: AdamW weight decay.
        w_prot: Protein gradient norm rescaling factor. The protein gradient
            is L2-normalized then scaled to ``w_prot * mrl_grad_norm``.
        w_edit: Edit gradient norm rescaling factor. Same rescaling as w_prot.
    """

    min_steps: int = 1000
    max_steps: int = 1000
    learning_rate: float = 0.001
    beta1: float = 0.9
    beta2: float = 0.999
    weight_decay: float = 0.01
    w_prot: float = 1.0
    w_edit: float = 0.5


@dataclass
class SAConfig:
    """Simulated annealing phase configuration.

    Attributes:
        steps: Number of simulated annealing steps.
        max_mutations: Maximum mutations per SA step.
        min_temp: Minimum annealing temperature.
        anneal_rate: Exponential decay rate for temperature.
        tau0: Initial annealing temperature. Use ~0 for greedy search.
        w_prot: Protein loss weight for Metropolis acceptance criterion.
            Multiplies protein loss directly in the total loss.
        w_edit: Edit loss weight for Metropolis acceptance criterion.
            Multiplies edit loss directly in the total loss.
    """

    steps: int = 2000
    max_mutations: int = 1
    min_temp: float = 1e-8
    anneal_rate: float = 1e-8
    tau0: float = 1e-8
    w_prot: float = 1.0
    w_edit: float = 0.5


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override dict into base dict.

    Nested dicts are merged recursively. All other values in override
    replace the corresponding base values.

    Args:
        base: Base configuration dict.
        override: Override dict whose values take precedence.

    Returns:
        New merged dict.
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _dataclass_from_dict(cls: type, data: dict) -> object:
    """Recursively construct a dataclass from a nested dict.

    Args:
        cls: The dataclass type to construct.
        data: Dict of field values, possibly with nested dicts for
            sub-dataclass fields.

    Returns:
        An instance of ``cls`` populated from ``data``.
    """
    field_types = {f.name: f.type for f in fields(cls)}
    kwargs = {}
    for key, value in data.items():
        if key in field_types and isinstance(value, dict):
            ft = field_types[key]
            if isinstance(ft, str):
                ft = globals()[ft]
            kwargs[key] = _dataclass_from_dict(ft, value)
        else:
            kwargs[key] = value
    return cls(**kwargs)


@dataclass
class OptimizationConfig:
    """Configuration for the OLG 5' UTR optimization pipeline.

    Composed of sub-configs for model, gradient descent, and simulated
    annealing phases. Supports YAML serialization via ``to_yaml`` and
    ``from_yaml``.

    Attributes:
        n_batch: Number of sequences to optimize in parallel.
        seq_length: Total nucleotide sequence length.
        alt_start_pos: 0-based position index for A in the alternative ATG.
        start_codon: Start codon sequence string or list of strings.
        fix_aa_seq: Protein constraint string for the overlapping ORF.
            Single letters for exact residues, ``[KR]`` for degenerate
            positions (K or R acceptable), ``X`` for wildcard (any residue).
            Example: ``"M[KR]AXC"``.
        right_overhang: Fixed DNA nucleotides past the ORF end (e.g. ``"TC"``).
            If provided, the pipeline builds the overhang mask automatically.
        seed_seq: Optional seed DNA sequence (5' side, before alt_start).
            Right-padded with zeros to ``seq_length``.
        w_mrl1: MRL1 weight fraction (0 to 1). Controls the balance between
            MRL1 and MRL2 in the loss. Default 0.5 weights both equally.
        eps: Epsilon for numerical stability.
        device: Compute device string.
        model: Model architecture and weight configuration.
        gd: Gradient descent phase configuration.
        sa: Simulated annealing phase configuration.
    """

    n_batch: int = 10
    seq_length: int = 100
    alt_start_pos: int = 50
    start_codon: str | list[str] = "ATG"
    fix_aa_seq: str | None = None
    right_overhang: str | None = None
    seed_seq: str | None = None
    w_mrl1: float = 0.5
    eps: float = 1e-8
    device: str = "cuda:0"
    model: ModelConfig = field(default_factory=ModelConfig)
    gd: GDConfig = field(default_factory=GDConfig)
    sa: SAConfig = field(default_factory=SAConfig)

    def to_yaml(self, path: str | Path) -> None:
        """Write configuration to a YAML file.

        Args:
            path: Output file path.
        """
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, *paths: str | Path) -> "OptimizationConfig":
        """Load configuration from one or more YAML files.

        When multiple paths are given, files are merged left to right:
        later files override earlier ones. Nested sections (``model``,
        ``gd``, ``sa``) are merged recursively so you only need to
        specify the fields you want to override. Missing fields use
        dataclass defaults.

        Example::

            # base.yaml sets lab defaults, experiment.yaml overrides a few fields
            config = OptimizationConfig.from_yaml("base.yaml", "experiment.yaml")

        Args:
            *paths: One or more paths to YAML config files.

        Returns:
            An ``OptimizationConfig`` instance populated from the merged files.
        """
        merged: dict = {}
        for path in paths:
            with open(path) as f:
                data = yaml.safe_load(f)
            if data:
                merged = _deep_merge(merged, data)
        return _dataclass_from_dict(cls, merged)
