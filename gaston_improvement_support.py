"""Consensus-boundary mixture-of-experts utilities.

This module is an experimental extension of GASTON-Mix.  A spatial gate is
shared by all genes, while each domain expert is a low-rank, gene-specific
linear combination of interpretable spatial basis functions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from tqdm.auto import tqdm

DeviceLike = Union[str, torch.device]


def _resolve_device(
    device: Optional[DeviceLike] = None,
    model: Optional[nn.Module] = None,
    coords: Optional[Tensor] = None,
) -> torch.device:
    """Return ``device``, else CUDA when available, else the model's device."""
    if device is not None:
        return torch.device(device)
    if coords is not None and coords.is_cuda:
        return coords.device
    if model is not None:
        model_device = next(model.parameters()).device
        if model_device.type == "cuda":
            return model_device
        if torch.cuda.is_available():
            return torch.device("cuda")
        return model_device
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _to_device(tensor: Optional[Tensor], device: torch.device) -> Optional[Tensor]:
    if tensor is None:
        return None
    if tensor.device == device:
        return tensor
    return tensor.to(device, non_blocking=device.type == "cuda")


def _configure_cuda() -> None:
    """Prefer fast GPU matmuls; TF32 is a tiny precision trade for large GEMMs."""
    if not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except (AttributeError, RuntimeError):
        pass


def set_seed(seed: int = 7) -> None:
    """Set NumPy and PyTorch random seeds."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_coordinates(coords: Tensor) -> Tensor:
    """Map each spatial axis to [-1, 1] while preserving tissue geometry."""
    minimum = coords.amin(dim=0, keepdim=True)
    span = (coords.amax(dim=0, keepdim=True) - minimum).clamp_min(1e-8)
    return 2.0 * (coords - minimum) / span - 1.0


@dataclass(frozen=True)
class BasisConfig:
    """Configuration for an identifiable, fixed spatial function dictionary."""

    sigmoid_slopes: Tuple[float, ...] = (4.0, 10.0)
    sigmoid_offsets: Tuple[float, ...] = (-0.35, 0.0, 0.35)
    exp_rates: Tuple[float, ...] = (1.5, 4.0)
    frequencies: Tuple[float, ...] = (0.5, 1.0)
    include_diagonals: bool = True


def compact_basis_config() -> BasisConfig:
    """13-function dictionary along two axes: constant, linear, sigmoid, decay.

    No diagonals and no Fourier terms. Intended to be used with learned
    orthonormal axes so the two directions can align to the tissue.
    """
    return BasisConfig(
        sigmoid_slopes=(6.0,),
        sigmoid_offsets=(-0.4, 0.0, 0.4),
        exp_rates=(2.0,),
        frequencies=(),
        include_diagonals=False,
    )


class SpatialBasis(nn.Module):
    """Constant, linear, sigmoid, exponential-decay and Fourier basis bank.

    Coordinates are expected in ``[-1, 1]^2``. Non-constant columns are
    standardized on a fixed reference grid, which makes coefficient penalties
    comparable across basis families and evaluations.
    """

    def __init__(self, config: BasisConfig = BasisConfig()) -> None:
        super().__init__()
        directions = [[1.0, 0.0], [0.0, 1.0]]
        direction_names = ["u", "v"]
        if config.include_diagonals:
            r = 2.0**-0.5
            directions += [[r, r], [r, -r]]
            direction_names += ["diag+", "diag-"]
        self.register_buffer("directions", torch.tensor(directions, dtype=torch.float32))
        self.config = config
        self.register_buffer(
            "sigmoid_slopes",
            torch.tensor(config.sigmoid_slopes, dtype=torch.float32),
        )
        self.register_buffer(
            "sigmoid_offsets",
            torch.tensor(config.sigmoid_offsets, dtype=torch.float32),
        )
        self.register_buffer(
            "exp_rates",
            torch.tensor(config.exp_rates, dtype=torch.float32),
        )
        self.register_buffer(
            "frequencies",
            torch.tensor(config.frequencies, dtype=torch.float32),
        )

        names, families = ["constant"], ["constant"]
        names += [f"linear:{direction_names[0]}", f"linear:{direction_names[1]}"]
        families += ["linear", "linear"]
        for direction_name in direction_names:
            for slope in config.sigmoid_slopes:
                for offset in config.sigmoid_offsets:
                    names.append(f"sigmoid:{direction_name}:k={slope:g}:c={offset:g}")
                    families.append("sigmoid")
            for rate in config.exp_rates:
                names.extend(
                    [
                        f"exp_decay:+{direction_name}:lambda={rate:g}",
                        f"exp_decay:-{direction_name}:lambda={rate:g}",
                    ]
                )
                families.extend(["exponential", "exponential"])
            for frequency in config.frequencies:
                names.extend(
                    [
                        f"sin:{direction_name}:f={frequency:g}",
                        f"cos:{direction_name}:f={frequency:g}",
                    ]
                )
                families.extend(["sinusoidal", "sinusoidal"])
        self.names = names
        self.families = families
        reference_axis = torch.linspace(-1.0, 1.0, 41)
        reference_y, reference_x = torch.meshgrid(
            reference_axis, reference_axis, indexing="ij"
        )
        reference = torch.stack(
            [reference_x.flatten(), reference_y.flatten()], dim=1
        )
        self.register_buffer("reference_coords", reference)
        reference_basis = self._raw_basis(reference, self.directions)[:, 1:]
        self.register_buffer(
            "basis_mean", reference_basis.mean(dim=0, keepdim=True)
        )
        self.register_buffer(
            "basis_scale",
            reference_basis.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-4),
        )

    @property
    def n_basis(self) -> int:
        return len(self.names)

    def _raw_basis(self, coords: Tensor, directions: Tensor) -> Tensor:
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError("coords must have shape [n_spots, 2]")
        if directions.ndim != 2 or directions.shape[1] != 2:
            raise ValueError("directions must have shape [n_axes, 2]")
        projections = coords @ directions.T
        n_spots, n_axes = projections.shape
        per_axis: List[Tensor] = []
        # Keep the original layout: for each axis, sigmoids then decays then Fourier.
        if self.sigmoid_slopes.numel() and self.sigmoid_offsets.numel():
            sigmoid = torch.sigmoid(
                self.sigmoid_slopes[None, None, :, None]
                * (
                    projections[:, :, None, None]
                    - self.sigmoid_offsets[None, None, None, :]
                )
            )
            per_axis.append(sigmoid.reshape(n_spots, n_axes, -1))
        if self.exp_rates.numel():
            z = projections[:, :, None]
            rates = self.exp_rates[None, None, :]
            plus = torch.exp(-rates * F.softplus(z))
            minus = torch.exp(-rates * F.softplus(-z))
            per_axis.append(torch.stack((plus, minus), dim=-1).reshape(n_spots, n_axes, -1))
        if self.frequencies.numel():
            angle = torch.pi * self.frequencies[None, None, :] * projections[:, :, None]
            per_axis.append(
                torch.stack((torch.sin(angle), torch.cos(angle)), dim=-1).reshape(
                    n_spots, n_axes, -1
                )
            )
        columns = [torch.ones_like(coords[:, :1]), projections[:, :1], projections[:, 1:2]]
        if per_axis:
            columns.append(torch.cat(per_axis, dim=2).reshape(n_spots, -1))
        return torch.cat(columns, dim=1)

    def _standardization_stats(self, directions: Tensor) -> Tuple[Tensor, Tensor]:
        reference = self._raw_basis(self.reference_coords, directions.detach())[:, 1:]
        mean = reference.mean(dim=0, keepdim=True).detach()
        scale = reference.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-4).detach()
        return mean, scale

    def forward(
        self, coords: Tensor, directions: Optional[Tensor] = None
    ) -> Tensor:
        axes = self.directions if directions is None else directions
        coords = coords.to(device=self.directions.device, dtype=self.directions.dtype)
        axes = axes.to(device=self.directions.device, dtype=self.directions.dtype)
        phi = self._raw_basis(coords, axes)
        mean, scale = self._standardization_stats(axes)
        nonconstant = (phi[:, 1:] - mean) / scale
        return torch.cat([phi[:, :1], nonconstant], dim=1)


class ConsensusSpatialMoE(nn.Module):
    """Spatial MoE with a common gate and low-rank gene-specific experts.

    For domain ``p`` and gene ``g``,

    ``E_p(s)_g = intercept[p,g] + sum_b theta[p,g,b] phi_b(s)``,

    with ``theta[p,g,:] = gene_loadings[p,g,:] @ profile_atoms[p,:,:]``.
    The rank limits the number of distinct profile archetypes per domain while
    still allowing every gene to use its own mixture of those archetypes.

    If ``learned_axes`` is true, a single angle shared by every expert rotates
    an orthonormal pair ``(u, v)``; all compact basis functions are evaluated
    in that tissue-aligned frame. The gate still sees raw coordinates.
    """

    def __init__(
        self,
        n_genes: int,
        n_domains: int,
        rank: int = 3,
        gate_hidden: Sequence[int] = (32, 32),
        gate_fourier_frequencies: Sequence[float] = (),
        basis_config: BasisConfig = BasisConfig(),
        temperature: float = 1.0,
        learned_axes: bool = False,
    ) -> None:
        super().__init__()
        if n_genes < 1 or n_domains < 1 or rank < 1:
            raise ValueError("n_genes, n_domains and rank must be positive")
        self.n_genes = n_genes
        self.n_domains = n_domains
        self.rank = rank
        self.temperature = temperature
        self.learned_axes = learned_axes
        if learned_axes and basis_config.include_diagonals:
            basis_config = replace(basis_config, include_diagonals=False)
        self.basis = SpatialBasis(basis_config)
        self.axis_angle = nn.Parameter(torch.zeros(())) if learned_axes else None

        gate_directions = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [2.0**-0.5, 2.0**-0.5],
                [2.0**-0.5, -(2.0**-0.5)],
            ]
        )
        self.register_buffer("gate_directions", gate_directions)
        self.gate_fourier_frequencies = tuple(gate_fourier_frequencies)
        gate_layers: List[nn.Module] = []
        in_features = 2 + 2 * len(self.gate_fourier_frequencies) * len(gate_directions)
        for width in gate_hidden:
            gate_layers.extend([nn.Linear(in_features, width), nn.Tanh()])
            in_features = width
        gate_layers.append(nn.Linear(in_features, n_domains))
        self.gate_network = nn.Sequential(*gate_layers)

        n_nonconstant = self.basis.n_basis - 1
        self.intercepts = nn.Parameter(torch.zeros(n_domains, n_genes))
        self.gene_loadings = nn.Parameter(
            0.08 * torch.randn(n_domains, n_genes, rank)
        )
        self.profile_atoms = nn.Parameter(
            0.08 * torch.randn(n_domains, rank, n_nonconstant)
        )
        # Used only by the negative-binomial likelihood.
        self.log_inverse_dispersion = nn.Parameter(torch.full((n_genes,), 2.0))
        group_keys = _basis_group_keys(self)
        unique_keys = sorted(set(group_keys))
        key_to_id = {key: idx for idx, key in enumerate(unique_keys)}
        self.register_buffer(
            "coefficient_group_ids",
            torch.tensor([key_to_id[key] for key in group_keys], dtype=torch.long),
        )
        self.n_coefficient_groups = len(unique_keys)

    def axis_directions(self) -> Tensor:
        """Return the orthonormal axes used by the expert dictionary."""
        if self.axis_angle is None:
            return self.basis.directions
        cosine = torch.cos(self.axis_angle)
        sine = torch.sin(self.axis_angle)
        return torch.stack(
            (torch.stack((cosine, sine)), torch.stack((-sine, cosine)))
        )

    def coefficients(self) -> Tensor:
        """Return non-constant coefficients with shape [domain, gene, basis]."""
        return torch.einsum("pgr,prb->pgb", self.gene_loadings, self.profile_atoms)

    def _prepare_coords(self, coords: Tensor) -> Tensor:
        """Copy coordinates onto the module device without mutating the input."""
        target = self.intercepts
        if coords.device == target.device and coords.dtype == target.dtype:
            return coords
        return coords.to(device=target.device, dtype=target.dtype)

    def _match_device(self, tensor: Tensor, device: torch.device) -> Tensor:
        if tensor.device == device:
            return tensor
        return tensor.to(device)

    def gate_features(self, coords: Tensor) -> Tensor:
        """Raw coordinates plus fixed multiscale Fourier gate features."""
        coords = self._prepare_coords(coords)
        if not self.gate_fourier_frequencies:
            return coords
        projections = coords @ self.gate_directions.T
        frequencies = coords.new_tensor(self.gate_fourier_frequencies)
        angle = torch.pi * frequencies[None, :, None] * projections[:, None, :]
        sincos = torch.stack((torch.sin(angle), torch.cos(angle)), dim=2)
        return torch.cat([coords, sincos.reshape(coords.shape[0], -1)], dim=1)

    def gate_logits(self, coords: Tensor) -> Tensor:
        return self.gate_network(self.gate_features(coords))

    def gates(self, coords: Tensor, temperature: Optional[float] = None) -> Tensor:
        source_device = coords.device
        tau = self.temperature if temperature is None else temperature
        gates = F.softmax(self.gate_logits(coords) / max(float(tau), 1e-3), dim=1)
        return self._match_device(gates, source_device)

    def expert_outputs(
        self, coords: Tensor, coefficients: Optional[Tensor] = None
    ) -> Tensor:
        """Return all expert values with shape [spot, domain, gene]."""
        source_device = coords.device
        coords = self._prepare_coords(coords)
        if coefficients is None:
            coefficients = self.coefficients()
        phi = self.basis(coords, directions=self.axis_directions())
        varying = torch.einsum("nb,pgb->npg", phi[:, 1:], coefficients)
        experts = varying + self.intercepts.unsqueeze(0)
        return self._match_device(experts, source_device)

    def forward(
        self,
        coords: Tensor,
        temperature: Optional[float] = None,
        coefficients: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Return mixed expression, soft gates and every expert's expression."""
        source_device = coords.device
        coords = self._prepare_coords(coords)
        if coefficients is None:
            coefficients = self.coefficients()
        gates = self.gates(coords, temperature)
        experts = self.expert_outputs(coords, coefficients=coefficients)
        prediction = torch.einsum("np,npg->ng", gates, experts)
        return (
            self._match_device(prediction, source_device),
            self._match_device(gates, source_device),
            self._match_device(experts, source_device),
        )

    @torch.no_grad()
    def hard_domains(self, coords: Tensor) -> Tensor:
        source_device = coords.device
        labels = self.gate_logits(coords).argmax(dim=1)
        return self._match_device(labels, source_device)


@dataclass
class LossWeights:
    """Weights for the regularizers in :func:`consensus_moe_loss`."""

    spatial_boundary: float = 0.03
    gate_entropy: float = 0.01
    gate_balance: float = 0.02
    coefficient_ridge: float = 2e-4
    basis_group_lasso: float = 2e-3
    fused_genes: float = 4e-3
    boundary_consensus: float = 0.08


def spatial_edges(coords: Tensor, radius_factor: float = 1.05) -> Tensor:
    """Build a radius graph using the smallest non-zero spot distance.

    This gives four-neighbour edges on a regular square grid and remains useful
    for irregular spots when coordinates have approximately uniform spacing.
    """
    distances = torch.cdist(coords, coords)
    positive = distances[distances > 1e-7]
    if positive.numel() == 0:
        raise ValueError("at least two distinct coordinates are required")
    radius = positive.min() * radius_factor
    edge_mask = torch.triu((distances <= radius) & (distances > 1e-7), diagonal=1)
    return edge_mask.nonzero(as_tuple=False).T


def spatial_knn_edges(coords: Tensor, n_neighbors: int = 6) -> Tensor:
    """Build a symmetric kNN graph for irregularly sampled coordinates."""
    if not 1 <= n_neighbors < coords.shape[0]:
        raise ValueError("n_neighbors must be between 1 and n_spots - 1")
    distances = torch.cdist(coords, coords)
    neighbors = distances.topk(n_neighbors + 1, largest=False).indices[:, 1:]
    source = torch.arange(coords.shape[0], device=coords.device)
    source = source[:, None].expand_as(neighbors)
    pairs = torch.stack([source.flatten(), neighbors.flatten()], dim=1)
    pairs = torch.sort(pairs, dim=1).values
    pairs = torch.unique(pairs, dim=0)
    return pairs.T


def negative_binomial_nll(
    log_relative_mean: Tensor,
    counts: Tensor,
    log_inverse_dispersion: Tensor,
    library_size: Optional[Tensor] = None,
) -> Tensor:
    """Mean NB2 negative log likelihood with an optional library-size offset."""
    if library_size is None:
        library_size = counts.sum(dim=1, keepdim=True).clamp_min(1.0)
    if library_size.ndim == 1:
        library_size = library_size[:, None]
    offset = torch.log(library_size / library_size.mean().clamp_min(1.0))
    mu = torch.exp((log_relative_mean + offset).clamp(-12.0, 12.0))
    inv_dispersion = F.softplus(log_inverse_dispersion).clamp_min(1e-4)[None, :]
    log_prob = (
        torch.lgamma(counts + inv_dispersion)
        - torch.lgamma(inv_dispersion)
        - torch.lgamma(counts + 1.0)
        + inv_dispersion * (torch.log(inv_dispersion) - torch.log(inv_dispersion + mu))
        + counts * (torch.log(mu.clamp_min(1e-8)) - torch.log(inv_dispersion + mu))
    )
    return -log_prob.mean()


def _basis_group_keys(model: ConsensusSpatialMoE) -> List[str]:
    """Group coefficients by function family and axis, not by the whole family.

    A linear field along a diagonal is cheaper if the shared frame rotates onto
    it than if both ``linear:u`` and ``linear:v`` stay active. The same logic
    applies to sigmoids and decays, which is what makes a shared angle
    identifiable.
    """
    keys: List[str] = []
    for name, family in zip(model.basis.names[1:], model.basis.families[1:]):
        parts = name.split(":")
        if family == "linear":
            keys.append(f"linear:{parts[1]}")
        elif family == "exponential":
            keys.append(f"exponential:{parts[1].lstrip('+-')}")
        elif family in {"sigmoid", "sinusoidal"}:
            keys.append(f"{family}:{parts[1]}")
        else:
            keys.append(family)
    return keys


def _family_group_penalty(
    model: ConsensusSpatialMoE, coefficients: Optional[Tensor] = None
) -> Tensor:
    if coefficients is None:
        coefficients = model.coefficients()
    grouped = coefficients.new_zeros(
        coefficients.shape[0], coefficients.shape[1], model.n_coefficient_groups
    )
    grouped.scatter_add_(
        -1,
        model.coefficient_group_ids.view(1, 1, -1).expand_as(coefficients),
        coefficients.square(),
    )
    return torch.sqrt(grouped + 1e-8).mean(dim=(0, 1)).sum()


def _boundary_regularizers(
    model: ConsensusSpatialMoE,
    coords: Tensor,
    gates: Tensor,
    edges: Tensor,
    gene_scale: Tensor,
    min_changed_fraction: float,
    change_threshold: float,
    change_temperature: float,
    coefficients: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Return spatial TV, fused-gene and consensus-support penalties."""
    source, target = edges
    gates_source = gates[source]
    gates_target = gates[target]
    spatial_tv = (1.0 - (gates_source * gates_target).sum(dim=1)).mean()

    if coefficients is None:
        coefficients = model.coefficients()
    midpoints = 0.5 * (coords[source] + coords[target])
    if coords.is_cuda:
        from torch.utils.checkpoint import checkpoint

        midpoint_experts = checkpoint(
            model.expert_outputs,
            midpoints,
            coefficients,
            use_reentrant=False,
        )
    else:
        midpoint_experts = model.expert_outputs(midpoints, coefficients=coefficients)

    pair_p, pair_q = torch.triu_indices(
        model.n_domains,
        model.n_domains,
        offset=1,
        device=coords.device,
    )
    n_pairs = int(pair_p.numel())
    if n_pairs == 0:
        zero = coords.new_zeros(())
        return spatial_tv, zero, zero

    crossing = (
        gates_source[:, pair_p] * gates_target[:, pair_q]
        + gates_source[:, pair_q] * gates_target[:, pair_p]
    )
    mass = crossing.sum(dim=0)
    coefficient_jump = (coefficients[pair_p] - coefficients[pair_q]).abs().mean(dim=-1)
    intercept_jump = (model.intercepts[pair_p] - model.intercepts[pair_q]).abs()
    gene_jump = (coefficient_jump + intercept_jump).mean(dim=-1)
    fused = (mass * gene_jump).sum()

    scale = gene_scale.clamp_min(1e-4)
    n_edges, n_genes = midpoint_experts.shape[0], midpoint_experts.shape[2]
    bytes_per_pair = max(n_edges * n_genes * midpoint_experts.element_size(), 1)
    if coords.is_cuda:
        try:
            free_bytes, _ = torch.cuda.mem_get_info(coords.device)
        except TypeError:
            free_bytes, _ = torch.cuda.mem_get_info()
        # Indexed copies, abs, sigmoid, and autograd each need their own slice.
        pair_chunk = max(1, int(free_bytes * 0.10) // (bytes_per_pair * 8))
        pair_chunk = max(1, min(n_pairs, pair_chunk))
    else:
        pair_chunk = max(1, min(n_pairs, 16_000_000 // max(n_edges * n_genes, 1)))
    consensus = coords.new_zeros(())
    for start in range(0, n_pairs, pair_chunk):
        sl = slice(start, start + pair_chunk)
        idx_p = pair_p[sl]
        idx_q = pair_q[sl]
        jumps = (
            midpoint_experts.index_select(1, idx_p)
            - midpoint_experts.index_select(1, idx_q)
        ).abs()
        changed = torch.sigmoid(
            (jumps / scale - change_threshold) / change_temperature
        )
        changed_fraction = changed.mean(dim=-1)
        consensus = consensus + (
            crossing[:, sl] * F.relu(min_changed_fraction - changed_fraction).square()
        ).sum()

    normalizer = mass.sum().clamp_min(1e-6)
    return spatial_tv, fused / normalizer, consensus / normalizer


def consensus_moe_loss(
    model: ConsensusSpatialMoE,
    coords: Tensor,
    expression: Tensor,
    edges: Optional[Tensor] = None,
    weights: LossWeights = LossWeights(),
    likelihood: str = "gaussian",
    library_size: Optional[Tensor] = None,
    observation_variance: Optional[Tensor] = None,
    temperature: Optional[float] = None,
    min_changed_fraction: float = 0.25,
    change_threshold: float = 0.6,
    change_temperature: float = 0.15,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    """Composite objective for parsimonious profiles and consensus boundaries.

    ``min_changed_fraction`` encodes that a boundary must be supported by more
    than an isolated gene.  The fused penalty simultaneously encourages many
    genes to remain unchanged across that boundary.
    """
    if edges is None:
        edges = spatial_edges(coords)
    coefficients = model.coefficients()
    prediction, gates, _ = model(coords, temperature, coefficients=coefficients)
    if likelihood == "gaussian":
        if observation_variance is None:
            reconstruction = F.mse_loss(prediction, expression)
        else:
            variance = observation_variance.clamp_min(1e-6)
            reconstruction = 0.5 * (
                (prediction - expression).square() / variance + variance.log()
            ).mean()
    elif likelihood in {"nb", "negative_binomial"}:
        reconstruction = negative_binomial_nll(
            prediction,
            expression,
            model.log_inverse_dispersion,
            library_size,
        )
    else:
        raise ValueError("likelihood must be 'gaussian' or 'negative_binomial'")

    gene_scale = expression.std(dim=0, unbiased=False).detach().clamp_min(0.1)
    spatial_tv, fused, consensus = _boundary_regularizers(
        model,
        coords,
        gates,
        edges,
        gene_scale,
        min_changed_fraction,
        change_threshold,
        change_temperature,
        coefficients=coefficients,
    )
    entropy = -(gates.clamp_min(1e-8) * gates.clamp_min(1e-8).log()).sum(1).mean()
    usage = gates.mean(dim=0)
    balance = ((usage - 1.0 / model.n_domains) ** 2).mean()
    ridge = coefficients.square().mean()
    group_lasso = _family_group_penalty(model, coefficients)

    terms = {
        "reconstruction": reconstruction,
        "spatial_boundary": spatial_tv,
        "gate_entropy": entropy,
        "gate_balance": balance,
        "coefficient_ridge": ridge,
        "basis_group_lasso": group_lasso,
        "fused_genes": fused,
        "boundary_consensus": consensus,
    }
    total = reconstruction
    for name in terms:
        if name != "reconstruction":
            total = total + getattr(weights, name) * terms[name]
    terms["total"] = total
    return total, terms


def pretrain_gate(
    model: ConsensusSpatialMoE,
    coords: Tensor,
    labels: Tensor,
    steps: int = 250,
    learning_rate: float = 2e-2,
    edges: Optional[Tensor] = None,
    spatial_smoothness: float = 0.0,
    weight_decay: float = 0.0,
) -> None:
    """Warm-start a gate from any provisional clustering."""
    labels = labels.to(device=coords.device, dtype=torch.long)
    optimizer = torch.optim.Adam(
        model.gate_network.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    for _ in range(steps):
        optimizer.zero_grad()
        logits = model.gate_logits(coords)
        loss = F.cross_entropy(logits, labels)
        if edges is not None and spatial_smoothness > 0:
            gates = F.softmax(logits, dim=1)
            source, target = edges
            boundary = 1.0 - (gates[source] * gates[target]).sum(dim=1)
            loss = loss + spatial_smoothness * boundary.mean()
        loss.backward()
        optimizer.step()


@torch.no_grad()
def initialize_experts_from_ridge(
    model: ConsensusSpatialMoE,
    coords: Tensor,
    expression: Tensor,
    labels: Tensor,
    observation_variance: Optional[Tensor] = None,
    ridge: float = 0.1,
) -> None:
    """Warm-start low-rank experts from weighted per-domain basis regression."""
    labels = labels.to(device=coords.device, dtype=torch.long)
    design = model.basis(coords, directions=model.axis_directions())
    n_basis = design.shape[1]
    counts = torch.bincount(labels, minlength=model.n_domains)
    if int(counts.min()) < n_basis:
        raise ValueError("each initialized domain needs at least n_basis spots")

    identity = torch.eye(n_basis, device=design.device, dtype=design.dtype)
    identity[0, 0] = 0.0  # do not shrink the intercept
    mask = F.one_hot(labels, model.n_domains).to(dtype=design.dtype)

    if observation_variance is None:
        gram = torch.einsum("np,nb,nc->pbc", mask, design, design)
        rhs = torch.einsum("np,nb,ng->pbg", mask, design, expression)
        beta = torch.linalg.solve(gram + ridge * identity, rhs)
    else:
        weights = observation_variance.clamp_min(1e-8).reciprocal()
        gram = torch.einsum("np,ng,nb,nc->pgbc", mask, weights, design, design)
        rhs = torch.einsum("np,nb,ng->pgb", mask, design, weights * expression)
        beta = torch.linalg.solve(gram + ridge * identity, rhs).permute(0, 2, 1)

    model.intercepts.copy_(beta[:, 0, :])
    theta = beta[:, 1:, :].transpose(1, 2)
    left, singular_values, right = torch.linalg.svd(theta, full_matrices=False)
    retained_rank = min(model.rank, singular_values.shape[-1])
    root_singular = singular_values[:, :retained_rank].sqrt()
    model.gene_loadings.zero_()
    model.profile_atoms.zero_()
    model.gene_loadings[:, :, :retained_rank].copy_(
        left[:, :, :retained_rank] * root_singular[:, None, :]
    )
    model.profile_atoms[:, :retained_rank].copy_(
        root_singular[:, :, None] * right[:, :retained_rank]
    )


@torch.no_grad()
def init_learned_axes_from_coords(
    model: ConsensusSpatialMoE, coords: Tensor
) -> None:
    """Initialize the shared angle from the leading spatial principal axis.

    This uses only the point cloud, not expression. It is useful when a slice
    is elongated; a nearly square cloud has a weak unique axis, so leave the
    angle at zero and let expression gradients rotate it.
    """
    if model.axis_angle is None:
        raise ValueError("model was constructed without learned_axes=True")
    centered = coords - coords.mean(dim=0, keepdim=True)
    _, _, right = torch.linalg.svd(centered, full_matrices=False)
    first_axis = right[0]
    model.axis_angle.fill_(torch.atan2(first_axis[1], first_axis[0]))


def axis_pair_error(learned_angle: Tensor | float, true_angle: float) -> float:
    """Smallest angle between two unlabeled, undirected orthonormal frames.

    Swapping the two axes or flipping a sign is not an error, so the residual
    lives in ``[0, pi/4]``.
    """
    learned = float(learned_angle)
    delta = (learned - true_angle + 0.25 * math.pi) % (0.5 * math.pi) - 0.25 * math.pi
    return abs(delta)


def rotate_coordinates(
    coords: Tensor, angle: float, isotropic_rescale: bool = True
) -> Tensor:
    """Rotate row-wise coordinates counterclockwise about the origin."""
    cosine = math.cos(angle)
    sine = math.sin(angle)
    rotation = torch.tensor(
        [[cosine, -sine], [sine, cosine]],
        device=coords.device,
        dtype=coords.dtype,
    )
    rotated = coords @ rotation.T
    if isotropic_rescale:
        rotated = rotated / rotated.abs().amax().clamp_min(1e-8)
    return rotated


def fit_consensus_moe(
    model: ConsensusSpatialMoE,
    coords: Tensor,
    expression: Tensor,
    epochs: int = 1200,
    learning_rate: float = 1e-2,
    gate_learning_rate: Optional[float] = None,
    freeze_gate_epochs: int = 0,
    weights: LossWeights = LossWeights(),
    likelihood: str = "gaussian",
    library_size: Optional[Tensor] = None,
    observation_variance: Optional[Tensor] = None,
    edges: Optional[Tensor] = None,
    start_temperature: float = 1.5,
    end_temperature: float = 0.2,
    print_every: int = 200,
    axis_learning_rate: Optional[float] = None,
    freeze_axes_epochs: int = 0,
    refit_experts_every: int = 0,
    refit_ridge: float = 0.3,
    device: Optional[DeviceLike] = None,
) -> List[Dict[str, float]]:
    """Fit the model with exponential gate-temperature annealing.

    If ``learned_axes`` is enabled, set ``refit_experts_every`` to a small
    positive integer (for example 1 or 5). Closed-form ridge updates keep the
    expert coefficients near-optimal for the current frame, so the angle
    receives an envelope gradient instead of being absorbed into ``theta``.

    Tensors and the model are moved to ``device``. By default this is CUDA when
    a GPU is available, otherwise the model's current device. After fitting,
    evaluate with coordinates on that same device.
    """
    device = _resolve_device(device, model=model, coords=coords)
    if device.type == "cuda":
        _configure_cuda()
    model.to(device)
    coords = _to_device(coords, device)
    expression = _to_device(expression, device)
    if coords is None or expression is None:
        raise ValueError("coords and expression are required")
    library_size = _to_device(library_size, device)
    observation_variance = _to_device(observation_variance, device)
    if edges is None:
        edges = spatial_edges(coords)
    else:
        edges = _to_device(edges, device)
    model.train()
    gate_parameters = list(model.gate_network.parameters())
    axis_parameters = [] if model.axis_angle is None else [model.axis_angle]
    frozen_ids = {id(parameter) for parameter in gate_parameters}
    frozen_ids.update(id(parameter) for parameter in axis_parameters)
    expert_parameters = [
        parameter
        for parameter in model.parameters()
        if id(parameter) not in frozen_ids
    ]
    parameter_groups = [
        {"params": expert_parameters, "lr": learning_rate},
        {
            "params": gate_parameters,
            "lr": learning_rate
            if gate_learning_rate is None
            else gate_learning_rate,
        },
    ]
    if axis_parameters:
        parameter_groups.append(
            {
                "params": axis_parameters,
                "lr": learning_rate
                if axis_learning_rate is None
                else axis_learning_rate,
            }
        )
    optimizer = torch.optim.Adam(parameter_groups)
    history: List[Dict[str, float]] = []
    progress = tqdm(range(epochs), desc="fit_consensus_moe", leave=True)
    postfix_every = 1 if print_every <= 1 else min(print_every, 10)
    for epoch in progress:
        fraction = epoch / max(epochs - 1, 1)
        temperature = start_temperature * (end_temperature / start_temperature) ** fraction
        if (
            refit_experts_every > 0
            and epoch % refit_experts_every == 0
            and model.axis_angle is not None
        ):
            labels = model.hard_domains(coords)
            counts = torch.bincount(labels, minlength=model.n_domains)
            if int(counts.min()) >= model.basis.n_basis:
                initialize_experts_from_ridge(
                    model,
                    coords,
                    expression,
                    labels,
                    observation_variance=observation_variance,
                    ridge=refit_ridge,
                )
        optimizer.zero_grad(set_to_none=True)
        loss, terms = consensus_moe_loss(
            model,
            coords,
            expression,
            edges=edges,
            weights=weights,
            likelihood=likelihood,
            library_size=library_size,
            observation_variance=observation_variance,
            temperature=temperature,
        )
        loss.backward()
        if epoch < freeze_gate_epochs:
            for parameter in gate_parameters:
                parameter.grad = None
        if epoch < freeze_axes_epochs:
            for parameter in axis_parameters:
                parameter.grad = None
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        log_now = epoch % print_every == 0 or epoch == epochs - 1
        postfix_now = epoch % postfix_every == 0 or epoch == epochs - 1
        if log_now or postfix_now:
            reconstruction = float(terms["reconstruction"].detach())
            total = float(terms["total"].detach())
            if postfix_now:
                progress.set_postfix(
                    recon=f"{reconstruction:.3f}",
                    loss=f"{total:.3f}",
                    temp=f"{temperature:.2f}",
                    refresh=False,
                )
            if log_now:
                row = {
                    "epoch": float(epoch),
                    "temperature": float(temperature),
                    "reconstruction": reconstruction,
                    "total": total,
                }
                row.update(
                    {
                        name: float(value.detach())
                        for name, value in terms.items()
                        if name not in row
                    }
                )
                if model.axis_angle is not None:
                    row["axis_angle"] = float(model.axis_angle.detach())
                history.append(row)
    model.temperature = end_temperature
    return history


def make_continuous_tiled_toy_data(
    layout: str = "checkerboard",
    tiles_per_axis: int = 4,
    points_per_tile: int = 5,
    noise_sd: float = 0.05,
    seed: int = 7,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Generate dense continuous gene fields inside a 4x4 tile mosaic.

    Returns coordinates, expression, spot-level domains, and the tile layout.
    The same domain-specific continuous function is evaluated wherever that
    domain occurs, including disconnected tiles.
    """
    if tiles_per_axis != 4:
        raise ValueError("this stress test is defined for a 4x4 tile mosaic")
    if points_per_tile < 2:
        raise ValueError("points_per_tile must be at least 2")
    if layout not in {"checkerboard", "random_three"}:
        raise ValueError("layout must be 'checkerboard' or 'random_three'")
    set_seed(seed)

    if layout == "checkerboard":
        row, column = np.indices((tiles_per_axis, tiles_per_axis))
        tile_domains_np = (row + column) % 2
    else:
        rng = np.random.default_rng(seed)
        tile_domains_np = rng.integers(
            0, 3, size=(tiles_per_axis, tiles_per_axis)
        )
        # Guarantee that every requested expert is represented.
        tile_domains_np[0, :3] = np.arange(3)

    resolution = tiles_per_axis * points_per_tile
    axis = (torch.arange(resolution, dtype=torch.float32) + 0.5) / resolution
    axis = 2.0 * axis - 1.0
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    coords = torch.stack([xx.flatten(), yy.flatten()], dim=1)

    row_ids = torch.arange(resolution) // points_per_tile
    column_ids = torch.arange(resolution) // points_per_tile
    tile_rows, tile_columns = torch.meshgrid(row_ids, column_ids, indexing="ij")
    tile_domains = torch.as_tensor(tile_domains_np, dtype=torch.long)
    domains = tile_domains[tile_rows, tile_columns].flatten()
    x, y = coords[:, 0], coords[:, 1]

    # Each row is one domain and each column is one gene. Some functions are
    # intentionally shared so not every gene supports every boundary.
    domain_functions = torch.stack(
        [
            torch.stack(
                [
                    0.75 * x - 0.15 * y,
                    0.85 * torch.sin(torch.pi * (x + 0.15)),
                    1.2 * torch.exp(-1.4 * (x + 1.0)),
                    torch.sigmoid(7.0 * (y + 0.1)),
                ],
                dim=1,
            ),
            torch.stack(
                [
                    0.75 * x - 0.15 * y,
                    0.8 * torch.cos(torch.pi * (y - 0.1)),
                    1.1 * torch.exp(-1.4 * (1.0 - x)),
                    torch.sigmoid(7.0 * (y + 0.1)),
                ],
                dim=1,
            ),
            torch.stack(
                [
                    -0.65 * y + 0.25,
                    0.8 * torch.cos(torch.pi * (y - 0.1)),
                    torch.full_like(x, -0.2),
                    torch.sigmoid(7.0 * (x - 0.15)),
                ],
                dim=1,
            ),
        ],
        dim=1,
    )
    # Moderate baseline shifts make provisional expression clustering possible,
    # while slopes/curvature still carry most of the within-tile information.
    domain_offsets = torch.tensor(
        [[0.0, 0.0, 0.0, 0.0], [0.0, 1.25, 0.8, 0.0], [-1.5, 1.25, -0.8, 1.4]]
    )
    domain_functions = domain_functions + domain_offsets.unsqueeze(0)
    spot_indices = torch.arange(coords.shape[0])
    expression = domain_functions[spot_indices, domains]
    expression = expression + noise_sd * torch.randn_like(expression)
    expression = (expression - expression.mean(0)) / expression.std(
        0, unbiased=False
    ).clamp_min(1e-4)
    return coords, expression, domains, tile_domains


def make_random_continuous_checkerboard(
    points_per_tile: int = 128,
    noise_variance_ratio: float = 0.5,
    expression_scale: float = 4.0,
    seed: int = 19,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Sample an irregular 4x4 checkerboard with heteroscedastic gene noise.

    The five outputs are coordinates, noisy expression, spot domains, latent
    expression means, and the known observation variances. Conditional on a
    location, noise is independent across genes and follows
    ``Normal(0, noise_variance_ratio * local_mean)``.
    """
    if points_per_tile < 1:
        raise ValueError("points_per_tile must be positive")
    if noise_variance_ratio <= 0:
        raise ValueError("noise_variance_ratio must be positive")
    if expression_scale <= 0:
        raise ValueError("expression_scale must be positive")
    set_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    coordinate_blocks, domain_blocks = [], []
    tile_width = 0.5
    for row in range(4):
        for column in range(4):
            local = torch.rand((points_per_tile, 2), generator=generator)
            local[:, 0] = -1.0 + column * tile_width + tile_width * local[:, 0]
            local[:, 1] = -1.0 + row * tile_width + tile_width * local[:, 1]
            coordinate_blocks.append(local)
            domain_blocks.append(
                torch.full((points_per_tile,), (row + column) % 2, dtype=torch.long)
            )
    coords = torch.cat(coordinate_blocks)
    domains = torch.cat(domain_blocks)
    x, y = coords[:, 0], coords[:, 1]

    domain_zero = torch.stack(
        [
            3.0 + 0.75 * x - 0.15 * y,
            2.5 + 0.85 * torch.sin(torch.pi * (x + 0.15)),
            1.5 + 1.2 * torch.exp(-1.4 * (x + 1.0)),
            1.5 + 2.0 * torch.sigmoid(7.0 * (y + 0.1)),
        ],
        dim=1,
    )
    domain_one = torch.stack(
        [
            3.0 + 0.75 * x - 0.15 * y,  # deliberately unchanged
            3.75 + 0.8 * torch.cos(torch.pi * (y - 0.1)),
            2.3 + 1.1 * torch.exp(-1.4 * (1.0 - x)),
            1.5 + 2.0 * torch.sigmoid(7.0 * (y + 0.1)),  # unchanged
        ],
        dim=1,
    )
    structured_mean = torch.where(domains[:, None] == 0, domain_zero, domain_one)
    # Two nuisance genes have no spatial or domain structure, but retain the
    # same expression-dependent independent noise as every structured gene.
    constant_mean = torch.stack(
        [torch.full_like(x, 2.4), torch.full_like(x, 3.2)], dim=1
    )
    latent_mean = expression_scale * torch.cat(
        [structured_mean, constant_mean], dim=1
    )
    observation_variance = noise_variance_ratio * latent_mean
    noise = torch.randn(
        latent_mean.shape, generator=generator, dtype=latent_mean.dtype
    ) * observation_variance.sqrt()
    expression = latent_mean + noise
    return coords, expression, domains, latent_mean, observation_variance


def make_oriented_axis_toy(
    n_spots: int = 1600,
    angle: float = 0.4,
    noise_variance_ratio: float = 0.35,
    expression_scale: float = 4.0,
    seed: int = 11,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, float]:
    """Two half-plane domains whose genes vary along a rotated orthonormal frame.

    Coordinates are generated in a canonical ``(u, v)`` frame, then rotated by
    ``angle``. Every structured gene is linear, sigmoidal, or exponentially
    decaying along those true axes, so a compact dictionary can represent the
    fields only if it recovers the same frame (up to sign and axis swap).
    """
    if n_spots < 8:
        raise ValueError("n_spots must be at least 8")
    if noise_variance_ratio <= 0 or expression_scale <= 0:
        raise ValueError("noise_variance_ratio and expression_scale must be positive")
    set_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    canonical = 2.0 * torch.rand((n_spots, 2), generator=generator) - 1.0
    u, v = canonical[:, 0], canonical[:, 1]
    domains = (u > 0.0).long()

    domain_zero = torch.stack(
        [
            2.0 + 1.1 * u,
            1.3 + 2.2 * torch.sigmoid(8.0 * (u + 0.05)),
            1.5 + 1.5 * torch.exp(-2.2 * F.softplus(u)),
            2.0 + 0.85 * v,
        ],
        dim=1,
    )
    domain_one = torch.stack(
        [
            2.7 - 0.7 * u,
            1.3 + 2.2 * torch.sigmoid(8.0 * (u - 0.35)),
            2.2 + 1.3 * torch.exp(-2.2 * F.softplus(-u)),
            2.0 + 0.85 * v,
        ],
        dim=1,
    )
    structured_mean = torch.where(domains[:, None] == 0, domain_zero, domain_one)
    constant_mean = torch.stack(
        [torch.full_like(u, 2.5), torch.full_like(u, 3.1)], dim=1
    )
    latent_mean = expression_scale * torch.cat(
        [structured_mean, constant_mean], dim=1
    )
    observation_variance = noise_variance_ratio * latent_mean
    noise = torch.randn(
        latent_mean.shape, generator=generator, dtype=latent_mean.dtype
    ) * observation_variance.sqrt()
    expression = latent_mean + noise
    coords = rotate_coordinates(canonical, angle, isotropic_rescale=True)
    return coords, expression, domains, latent_mean, observation_variance, float(angle)


def graph_expression_initialization(
    coords: Tensor,
    expression: Tensor,
    n_domains: int,
    n_neighbors: int = 16,
    seed: int = 7,
    residualize_global_trends: bool = True,
    ridge: float = 1.0,
) -> Tensor:
    """Infer provisional domains from local expression without boundary labels.

    The only spatial prior is that nearby observations can be averaged. No
    tiles, true labels, boundary coordinates, or domain shapes are supplied.
    """
    from sklearn.cluster import KMeans

    features = expression
    if residualize_global_trends:
        smooth_basis = SpatialBasis(
            BasisConfig(
                sigmoid_slopes=(2.0,),
                sigmoid_offsets=(0.0,),
                exp_rates=(1.5,),
                frequencies=(0.5,),
                include_diagonals=False,
            )
        ).to(device=coords.device, dtype=coords.dtype)
        design = smooth_basis(coords)
        identity = torch.eye(
            design.shape[1], device=design.device, dtype=design.dtype
        )
        coefficients = torch.linalg.solve(
            design.T @ design + ridge * identity,
            design.T @ expression,
        )
        features = expression - design @ coefficients

    distances = torch.cdist(coords, coords)
    neighbors = distances.topk(n_neighbors + 1, largest=False).indices
    local_features = features[neighbors].mean(dim=1)
    local_features = (
        local_features - local_features.mean(dim=0, keepdim=True)
    ) / local_features.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-4)
    labels = KMeans(
        n_clusters=n_domains, n_init=50, random_state=seed
    ).fit_predict(local_features.detach().cpu().numpy())
    return torch.as_tensor(labels, device=coords.device, dtype=torch.long)


def refine_graph_assignments(
    model: ConsensusSpatialMoE,
    coords: Tensor,
    expression: Tensor,
    initial_labels: Tensor,
    observation_variance: Optional[Tensor] = None,
    n_neighbors: int = 6,
    cycles: int = 8,
    ridge: float = 0.3,
) -> Tuple[Tensor, List[float]]:
    """Mixture-of-spatial-regressions EM without known boundary locations."""
    labels = initial_labels.clone().to(device=coords.device, dtype=torch.long)
    distances = torch.cdist(coords, coords)
    neighbors = distances.topk(n_neighbors + 1, largest=False).indices
    assignment_changes: List[float] = []
    for _ in range(cycles):
        initialize_experts_from_ridge(
            model,
            coords,
            expression,
            labels,
            observation_variance=observation_variance,
            ridge=ridge,
        )
        with torch.no_grad():
            errors = (model.expert_outputs(coords) - expression[:, None, :]).square()
            if observation_variance is not None:
                errors = errors / observation_variance[:, None, :].clamp_min(1e-6)
            errors = errors.mean(dim=2)
            local_errors = errors[neighbors].mean(dim=1)
            new_labels = local_errors.argmin(dim=1)
            assignment_changes.append(float((new_labels != labels).float().mean()))
            labels = new_labels
    return labels, assignment_changes


def irregular_tile_expression_initialization(
    coords: Tensor,
    expression: Tensor,
    n_domains: int,
    tiles_per_axis: int = 4,
    seed: int = 7,
    residualize_global_trends: bool = True,
    ridge: float = 0.1,
) -> Tensor:
    """Cluster tile-average profiles for uniformly or irregularly sampled spots."""
    from sklearn.cluster import KMeans

    features = expression
    if residualize_global_trends:
        smooth_basis = SpatialBasis(
            BasisConfig(frequencies=(0.5, 1.0))
        ).to(device=coords.device, dtype=coords.dtype)
        design = smooth_basis(coords)
        identity = torch.eye(
            design.shape[1], device=design.device, dtype=design.dtype
        )
        coefficients = torch.linalg.solve(
            design.T @ design + ridge * identity,
            design.T @ expression,
        )
        features = expression - design @ coefficients

    tile_xy = torch.floor((coords + 1.0) * (tiles_per_axis / 2.0)).long()
    tile_xy = tile_xy.clamp(0, tiles_per_axis - 1)
    tile_ids = tile_xy[:, 1] * tiles_per_axis + tile_xy[:, 0]
    n_tiles = tiles_per_axis**2
    sums = torch.zeros(
        n_tiles,
        expression.shape[1],
        device=expression.device,
        dtype=expression.dtype,
    )
    counts = torch.zeros(
        n_tiles, 1, device=expression.device, dtype=expression.dtype
    )
    sums.index_add_(0, tile_ids, features)
    counts.index_add_(
        0,
        tile_ids,
        torch.ones(expression.shape[0], 1, device=expression.device),
    )
    tile_means = sums / counts.clamp_min(1.0)
    tile_labels = KMeans(
        n_clusters=n_domains, n_init=50, random_state=seed
    ).fit_predict(tile_means.detach().cpu().numpy())
    tile_labels = torch.as_tensor(tile_labels, device=coords.device, dtype=torch.long)
    return tile_labels[tile_ids]


def tilewise_expression_initialization(
    expression: Tensor,
    tiles_per_axis: int,
    points_per_tile: int,
    n_domains: int,
    seed: int = 7,
    coords: Optional[Tensor] = None,
    residualize_global_trends: bool = False,
    ridge: float = 0.1,
) -> Tensor:
    """Cluster tile-average profiles and broadcast labels back to spots.

    Residualizing a shared smooth field is particularly useful for alternating
    domains because it prevents broad tissue gradients from dominating the
    provisional clusters.
    """
    from sklearn.cluster import KMeans

    resolution = tiles_per_axis * points_per_tile
    if expression.shape[0] != resolution**2:
        raise ValueError("expression shape does not match the tile grid")
    features = expression
    if residualize_global_trends:
        if coords is None:
            raise ValueError("coords are required to residualize global trends")
        smooth_basis = SpatialBasis(
            BasisConfig(frequencies=(0.5, 1.0))
        ).to(device=coords.device, dtype=coords.dtype)
        design = smooth_basis(coords)
        identity = torch.eye(
            design.shape[1], device=design.device, dtype=design.dtype
        )
        coefficients = torch.linalg.solve(
            design.T @ design + ridge * identity,
            design.T @ expression,
        )
        features = expression - design @ coefficients

    image = features.reshape(resolution, resolution, expression.shape[1])
    tile_means = (
        image.reshape(
            tiles_per_axis,
            points_per_tile,
            tiles_per_axis,
            points_per_tile,
            expression.shape[1],
        )
        .mean(dim=(1, 3))
        .reshape(tiles_per_axis**2, expression.shape[1])
    )
    tile_labels = KMeans(
        n_clusters=n_domains, n_init=30, random_state=seed
    ).fit_predict(tile_means.detach().cpu().numpy())
    tile_labels = torch.as_tensor(
        tile_labels.reshape(tiles_per_axis, tiles_per_axis), dtype=torch.long
    )
    return tile_labels.repeat_interleave(
        points_per_tile, dim=0
    ).repeat_interleave(points_per_tile, dim=1).flatten()


def refine_tiled_assignments(
    model: ConsensusSpatialMoE,
    coords: Tensor,
    expression: Tensor,
    initial_labels: Tensor,
    tiles_per_axis: int = 4,
    points_per_tile: int = 5,
    cycles: int = 4,
    expert_epochs: int = 250,
    learning_rate: float = 8e-3,
    weights: LossWeights = LossWeights(),
) -> Tuple[Tensor, List[float]]:
    """Hard-EM warm start for repeated, disconnected tile domains.

    Expert parameters are fitted with a fixed gate, then each whole tile is
    reassigned to the expert with the smallest expression error. This is useful
    when ordinary spatial clustering cannot initialize a checkerboard gate.
    """
    resolution = tiles_per_axis * points_per_tile
    if coords.shape[0] != resolution**2:
        raise ValueError("coordinates do not match the requested tile grid")
    labels = initial_labels.clone().to(device=coords.device, dtype=torch.long)
    assignment_changes: List[float] = []

    for _ in range(cycles):
        pretrain_gate(model, coords, labels, steps=250, learning_rate=1e-2)
        for parameter in model.gate_network.parameters():
            parameter.requires_grad_(False)
        fit_consensus_moe(
            model,
            coords,
            expression,
            epochs=expert_epochs,
            learning_rate=learning_rate,
            weights=weights,
            start_temperature=0.15,
            end_temperature=0.15,
            print_every=expert_epochs,
        )
        for parameter in model.gate_network.parameters():
            parameter.requires_grad_(True)

        with torch.no_grad():
            errors = (model.expert_outputs(coords) - expression[:, None, :]).square()
            errors = errors.mean(dim=2).reshape(
                resolution, resolution, model.n_domains
            )
            tile_errors = errors.reshape(
                tiles_per_axis,
                points_per_tile,
                tiles_per_axis,
                points_per_tile,
                model.n_domains,
            ).mean(dim=(1, 3))
            tile_labels = tile_errors.argmin(dim=2)
            new_labels = tile_labels.repeat_interleave(
                points_per_tile, dim=0
            ).repeat_interleave(points_per_tile, dim=1).flatten()
            assignment_changes.append(float((new_labels != labels).float().mean()))
            labels = new_labels

    pretrain_gate(model, coords, labels, steps=350, learning_rate=8e-3)
    return labels, assignment_changes


def make_tiled_toy_data(
    grid_size: int = 4,
    noise_sd: float = 0.04,
    seed: int = 7,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Create four square domains and four genes with distinct local behaviors."""
    if grid_size < 4:
        raise ValueError("grid_size must be at least 4")
    set_seed(seed)
    axis = torch.linspace(-1.0, 1.0, grid_size)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    coords = torch.stack([xx.flatten(), yy.flatten()], dim=1)
    right = (coords[:, 0] >= 0).long()
    bottom = (coords[:, 1] >= 0).long()
    domains = bottom * 2 + right
    x, y = coords[:, 0], coords[:, 1]

    expression = torch.empty(coords.shape[0], 4)
    # Gene 0: conserved linear trend, with a shared lower-half level shift.
    expression[:, 0] = 0.35 * x + 0.8 * bottom
    # Gene 1: linear in the left tiles, oppositely sloped in the right tiles.
    expression[:, 1] = torch.where(right.bool(), 0.8 - 0.75 * y, 0.15 + 0.75 * y)
    # Gene 2: sinusoidal above, nearly constant below.
    expression[:, 2] = torch.where(
        bottom.bool(), torch.full_like(x, -0.25), 0.65 * torch.sin(torch.pi * x)
    )
    # Gene 3: directional decay changes orientation between left and right.
    left_decay = torch.exp(-2.2 * (x + 1.0))
    right_decay = torch.exp(-2.2 * (1.0 - y))
    expression[:, 3] = torch.where(right.bool(), right_decay, left_decay)
    expression = expression + noise_sd * torch.randn_like(expression)
    expression = (expression - expression.mean(0)) / expression.std(
        0, unbiased=False
    ).clamp_min(1e-4)
    return coords, expression, domains


MERFISH_META_COLUMNS = {
    "Unnamed: 0",
    "cell_name",
    "coord_X",
    "coord_Y",
    "class",
    "cell_class",
    "mouse",
    "sample_id",
    "cell_section",
    "spatial_module_l1_complete",
}

MERFISH_ABC_META_COLUMNS = {
    "cell_label",
    "coord_X",
    "coord_Y",
    "cell_class",
    "cell_section",
    "spatial_module_l1_complete",
    "Anatomical Division",
    "Anatomical Structure",
    "Anatomical Substructure",
}

def load_merfish_slice(
    csv_path: str,
    dataset: str = "MERFISH_cortex",
    section: str = "mouse1_slice212",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load one MERFISH section: coordinates, expression, gene names, cell classes."""
    import pandas as pd

    table = pd.read_csv(csv_path)
    if dataset == "MERFISH_cortex":
        if "cell_section" not in table.columns:
            raise ValueError("expected a cell_section column")
        slice_table = table.loc[table["cell_section"] == section].reset_index(drop=True)
        if slice_table.empty:
            raise ValueError(f"no cells found for section {section}")
        gene_names = np.array(
            [column for column in slice_table.columns if column not in MERFISH_META_COLUMNS]
        )
        coords = slice_table[["coord_X", "coord_Y"]].to_numpy(np.float32)
        expression = slice_table[list(gene_names)].to_numpy(np.float32)
        cell_class = slice_table["cell_class"].to_numpy()
        return coords, expression, gene_names, cell_class
    elif dataset == "MERFISH_ABC":
        if "cell_section" not in table.columns:
            raise ValueError("expected a cell_section column")
        slice_table = table.loc[table["cell_section"] == section].reset_index(drop=True)
        if slice_table.empty:
            raise ValueError(f"no cells found for section {section}")
        gene_names = np.array(
            [column for column in slice_table.columns if column not in MERFISH_ABC_META_COLUMNS]
        )
        coords = slice_table[["coord_X", "coord_Y"]].to_numpy(np.float32)
        expression = slice_table[list(gene_names)].to_numpy(np.float32)
        cell_class = slice_table["cell_class"].to_numpy()
        return coords, expression, gene_names, cell_class
    else:
        raise ValueError(f"unknown dataset: {dataset}")


def knn_smooth_features(
    coords: Tensor, features: Tensor, n_neighbors: int = 8
) -> Tensor:
    """Average features over a spatial neighborhood, including the query point."""
    distances = torch.cdist(coords, coords)
    neighbors = distances.topk(n_neighbors + 1, largest=False).indices
    return features[neighbors].mean(dim=1)


@dataclass
class SpatialExpressionPCA:
    """PCA of spatially smoothed log-expression, used as a compact MoE target."""

    coords: Tensor
    pcs: Tensor
    log_expression: Tensor
    smoothed_log_expression: Tensor
    gene_names: np.ndarray
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    pca_mean: np.ndarray
    pca_components: np.ndarray
    pc_scale: np.ndarray
    explained_variance_ratio: np.ndarray

    def inverse(self, pcs: Tensor) -> Tensor:
        """Map predicted (whitened) PCs back to log-expression space."""
        pcs_np = pcs.detach().cpu().numpy() if torch.is_tensor(pcs) else np.asarray(pcs)
        pcs_raw = pcs_np * self.pc_scale
        scaled = pcs_raw @ self.pca_components + self.pca_mean
        log_expression = scaled * self.scaler_scale + self.scaler_mean
        return torch.as_tensor(log_expression, dtype=torch.float32, device=self.coords.device)


def spatial_gene_pca(
    coords_raw: np.ndarray,
    expression: np.ndarray,
    gene_names: np.ndarray,
    n_pcs: int = 32,
    n_neighbors: int = 8,
) -> SpatialExpressionPCA:
    """PCA on kNN-smoothed log1p expression.

    Smoothing makes the leading components spatial expression programs rather
    than single-cell dropout. Coordinates themselves are not concatenated into
    the PCA, so domain geometry still has to be learned from expression.
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    coords = normalize_coordinates(torch.as_tensor(coords_raw, dtype=torch.float32))
    log_expression = torch.log1p(torch.as_tensor(expression, dtype=torch.float32))
    smoothed = knn_smooth_features(coords, log_expression, n_neighbors=n_neighbors)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(smoothed.numpy())
    n_pcs = min(n_pcs, scaled.shape[0] - 1, scaled.shape[1])
    pca = PCA(n_components=n_pcs, random_state=7)
    pcs_raw = pca.fit_transform(scaled)
    pc_scale = np.maximum(pcs_raw.std(axis=0).astype(np.float32), 1e-4)
    pcs = pcs_raw / pc_scale
    return SpatialExpressionPCA(
        coords=coords,
        pcs=torch.as_tensor(pcs, dtype=torch.float32),
        log_expression=log_expression,
        smoothed_log_expression=smoothed,
        gene_names=np.asarray(gene_names),
        scaler_mean=scaler.mean_.astype(np.float32),
        scaler_scale=scaler.scale_.astype(np.float32),
        pca_mean=pca.mean_.astype(np.float32),
        pca_components=pca.components_.astype(np.float32),
        pc_scale=pc_scale,
        explained_variance_ratio=pca.explained_variance_ratio_.astype(np.float32),
    )


def align_labels(predicted: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Align arbitrary MoE labels to reference labels using Hungarian matching."""
    from scipy.optimize import linear_sum_assignment

    predicted = np.asarray(predicted)
    truth = np.asarray(truth)
    pred_values = np.unique(predicted)
    true_values = np.unique(truth)
    confusion = np.zeros((len(pred_values), len(true_values)), dtype=int)
    for i, pred_value in enumerate(pred_values):
        for j, true_value in enumerate(true_values):
            confusion[i, j] = np.sum(
                (predicted == pred_value) & (truth == true_value)
            )
    rows, columns = linear_sum_assignment(-confusion)
    mapping = {pred_values[row]: true_values[column] for row, column in zip(rows, columns)}
    return np.array([mapping.get(value, value) for value in predicted])
