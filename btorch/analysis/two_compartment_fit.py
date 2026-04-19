"""Allen-data fitting utilities for the two-compartment GLIF neuron."""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from btorch.models import environ, functional


@dataclass
class AllenSweepBatch:
    """Preprocessed Allen sweep tensors for fitting.

    Args:
        specimen_id: Allen specimen identifier.
        sweep_number: Sweep number within the NWB file.
        dt_ms: Simulation timestep in milliseconds.
        i_soma: Somatic current tensor shaped ``(T, B, N)``.
        v_true: Recorded voltage tensor shaped ``(T, B, N)``.
        spike_true: Binary spike tensor shaped ``(T, B, N)``.
        i_apical: Optional apical input tensor shaped ``(T, B, N)``.
        metadata: Auxiliary Allen metadata retained for bookkeeping.
    """

    specimen_id: int
    sweep_number: int
    dt_ms: float
    i_soma: Tensor
    v_true: Tensor
    spike_true: Tensor
    i_apical: Tensor | None = None
    metadata: dict[str, Any] | None = None


def _require_allensdk():
    try:
        from allensdk.core.cell_types_cache import CellTypesCache
    except ImportError as exc:
        raise ImportError(
            "AllenSDK is required for Allen Brain Cell Types access. "
            "Install it with `pip install allensdk` before using this "
            "pipeline."
        ) from exc
    return CellTypesCache


def get_cell_types_cache(
    manifest_file: str | Path | None = None,
    *,
    cache: Any | None = None,
):
    """Create or reuse an AllenSDK ``CellTypesCache`` instance."""
    if cache is not None:
        return cache
    CellTypesCache = _require_allensdk()
    if manifest_file is None:
        return CellTypesCache()
    return CellTypesCache(manifest_file=str(manifest_file))


def _matches_any(value: Any, expected: Sequence[str]) -> bool:
    text = str(value).lower()
    return any(item.lower() in text for item in expected)


def filter_mouse_visp_l5_pyramidal_cells(
    cells: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Filter Allen cell metadata to mouse VISp layer-5 pyramidal candidates.

    Notes:
        Allen metadata can vary across cache versions, so this filter accepts a
        few equivalent key names and uses ``spiny`` as a practical proxy for
        pyramidal morphology when an explicit class label is unavailable.
    """
    matched = []
    for cell in cells:
        species = cell.get("species") or cell.get("donor__species")
        area = (
            cell.get("structure_area_abbrev")
            or cell.get("structure_acronym")
            or cell.get("structure_parent__acronym")
        )
        layer = cell.get("structure_layer_name") or cell.get("cortical_layer")
        dendrite = cell.get("dendrite_type") or cell.get("cell_type")

        if not _matches_any(species, ["mouse"]):
            continue
        if not _matches_any(area, ["visp", "primary visual cortex"]):
            continue
        if not _matches_any(layer, ["layer 5", "5"]):
            continue
        if not _matches_any(dendrite, ["spiny", "pyramidal"]):
            continue
        matched.append(cell)
    return matched


def query_mouse_visp_l5_pyramidal_cells(
    *,
    manifest_file: str | Path | None = None,
    cache: Any | None = None,
) -> list[dict[str, Any]]:
    """Query candidate mouse VISp layer-5 pyramidal cells from AllenSDK."""
    ctc = get_cell_types_cache(manifest_file, cache=cache)
    return filter_mouse_visp_l5_pyramidal_cells(ctc.get_cells())


def choose_current_clamp_sweeps(
    sweep_records: Sequence[dict[str, Any]],
    *,
    preferred_stimuli: Sequence[str] = ("long square",),
) -> list[dict[str, Any]]:
    """Select current-clamp sweeps appropriate for somatic fitting."""
    matched = []
    for sweep in sweep_records:
        stimulus_name = (
            sweep.get("stimulus_name")
            or sweep.get("ephys_stimulus", {}).get("description")
            or ""
        )
        stimulus_units = sweep.get("stimulus_units") or ""
        clamp_mode = sweep.get("clamp_mode") or sweep.get("stimulus_type_name") or ""

        is_current_clamp = _matches_any(stimulus_units, ["pa", "amp"]) or (
            clamp_mode and not _matches_any(clamp_mode, ["voltage clamp"])
        )
        if not is_current_clamp:
            continue
        if preferred_stimuli and not _matches_any(stimulus_name, preferred_stimuli):
            continue
        matched.append(sweep)
    return matched


def detect_spikes_from_voltage(
    voltage: Tensor,
    *,
    threshold: float = 0.0,
) -> Tensor:
    """Detect upward threshold crossings in a voltage trace."""
    above = voltage >= threshold
    shifted = torch.zeros_like(above)
    shifted[1:] = above[:-1]
    return (above & ~shifted).to(voltage.dtype)


def resample_trace(
    trace: np.ndarray | Tensor,
    *,
    source_dt_ms: float,
    target_dt_ms: float,
) -> Tensor:
    """Resample a 1D trace to a new timestep using linear interpolation."""
    if target_dt_ms <= 0.0:
        raise ValueError(f"target_dt_ms must be positive, got {target_dt_ms}.")
    trace_t = torch.as_tensor(trace, dtype=torch.float32)
    if trace_t.ndim != 1:
        raise ValueError(f"Expected a 1D trace, got shape {tuple(trace_t.shape)}.")
    if abs(source_dt_ms - target_dt_ms) < 1e-12:
        return trace_t

    duration_ms = max((trace_t.shape[0] - 1) * source_dt_ms, 0.0)
    target_steps = max(int(round(duration_ms / target_dt_ms)) + 1, 1)
    source_time = torch.linspace(0.0, duration_ms, trace_t.shape[0])
    target_time = torch.linspace(0.0, duration_ms, target_steps)

    np_resampled = np.interp(
        target_time.cpu().numpy(),
        source_time.cpu().numpy(),
        trace_t.cpu().numpy(),
    )
    return torch.from_numpy(np_resampled).to(dtype=torch.float32)


def load_allen_sweep(
    specimen_id: int,
    sweep_number: int,
    *,
    dt_ms: float = 0.5,
    manifest_file: str | Path | None = None,
    cache: Any | None = None,
    voltage_spike_threshold: float = 0.0,
) -> AllenSweepBatch:
    """Load one Allen ephys sweep and convert it to time-first torch tensors."""
    ctc = get_cell_types_cache(manifest_file, cache=cache)
    dataset = ctc.get_ephys_data(specimen_id)
    sweep = dataset.get_sweep(sweep_number)

    response = np.asarray(sweep["response"], dtype=np.float32)
    stimulus = np.asarray(sweep["stimulus"], dtype=np.float32)
    sampling_rate = float(sweep["sampling_rate"])
    index_start, index_stop = sweep["index_range"]
    sl = slice(int(index_start), int(index_stop) + 1)

    source_dt_ms = 1000.0 / sampling_rate
    voltage = resample_trace(response[sl], source_dt_ms=source_dt_ms, target_dt_ms=dt_ms)
    current = resample_trace(stimulus[sl], source_dt_ms=source_dt_ms, target_dt_ms=dt_ms)
    spike_true = detect_spikes_from_voltage(
        voltage,
        threshold=voltage_spike_threshold,
    )

    return AllenSweepBatch(
        specimen_id=specimen_id,
        sweep_number=sweep_number,
        dt_ms=dt_ms,
        i_soma=current[:, None, None],
        v_true=voltage[:, None, None],
        spike_true=spike_true[:, None, None],
        i_apical=torch.zeros_like(current)[:, None, None],
        metadata={"sampling_rate_hz": sampling_rate},
    )


def mask_post_spike_voltage_samples(
    spike_true: Tensor,
    *,
    refractory_bins: int = 3,
) -> Tensor:
    """Mask out voltage samples immediately after true spikes."""
    if refractory_bins < 0:
        raise ValueError(
            f"refractory_bins must be non-negative, got {refractory_bins}."
        )
    mask = torch.ones_like(spike_true, dtype=torch.bool)
    spike_idx = spike_true > 0
    for offset in range(refractory_bins + 1):
        if offset == 0:
            mask = mask & ~spike_idx
            continue
        shifted = torch.zeros_like(spike_idx)
        shifted[offset:] = spike_idx[:-offset]
        mask = mask & ~shifted
    return mask


def exponential_filter_spike_train(
    spike_train: Tensor,
    *,
    tau_ms: float,
    dt_ms: float,
) -> Tensor:
    """Apply a causal exponential filter to a spike train."""
    if tau_ms <= 0.0:
        raise ValueError(f"tau_ms must be positive, got {tau_ms}.")
    alpha = float(np.exp(-dt_ms / tau_ms))
    filtered = torch.zeros_like(spike_train)
    filtered[0] = spike_train[0]
    for t in range(1, spike_train.shape[0]):
        filtered[t] = alpha * filtered[t - 1] + (1.0 - alpha) * spike_train[t]
    return filtered


def two_compartment_loss(
    *,
    v_pred: Tensor,
    spike_pred: Tensor,
    v_true: Tensor,
    spike_true: Tensor,
    dt_ms: float,
    w_Ca: Tensor | None = None,
    voltage_weight: float = 1.0,
    spike_weight: float = 1.0,
    sparsity_weight: float = 1e-4,
    spike_tau_ms: float = 10.0,
    post_spike_mask_ms: float = 3.0,
) -> dict[str, Tensor]:
    """Compute a composite fitting loss for the two-compartment model."""
    refractory_bins = int(round(post_spike_mask_ms / dt_ms))
    mask = mask_post_spike_voltage_samples(
        spike_true,
        refractory_bins=refractory_bins,
    )

    if mask.any():
        voltage_loss = F.mse_loss(v_pred[mask], v_true[mask])
    else:
        voltage_loss = torch.zeros((), device=v_pred.device, dtype=v_pred.dtype)

    spike_pred_smooth = exponential_filter_spike_train(
        spike_pred,
        tau_ms=spike_tau_ms,
        dt_ms=dt_ms,
    )
    spike_true_smooth = exponential_filter_spike_train(
        spike_true,
        tau_ms=spike_tau_ms,
        dt_ms=dt_ms,
    )
    spike_loss = F.smooth_l1_loss(spike_pred_smooth, spike_true_smooth)

    if w_Ca is None:
        sparsity_loss = torch.zeros((), device=v_pred.device, dtype=v_pred.dtype)
    else:
        sparsity_loss = w_Ca.abs().mean()

    total = (
        voltage_weight * voltage_loss
        + spike_weight * spike_loss
        + sparsity_weight * sparsity_loss
    )
    return {
        "total": total,
        "voltage": voltage_loss,
        "spike": spike_loss,
        "sparsity": sparsity_loss,
    }


def rollout_two_compartment(
    model,
    i_soma: Tensor,
    i_apical: Tensor | None = None,
) -> dict[str, Tensor]:
    """Run a time-first rollout and collect fitting-relevant traces."""
    spike, voltage, state = model.multi_step_forward(
        i_soma,
        i_apical,
        return_state=True,
    )
    return {
        "spike": spike,
        "v": voltage,
        "v_pre_spike": state["v_pre_spike"],
        "i_a": state["i_a"],
        "i_bap": state["i_bap"],
    }


def fit_two_compartment_model(
    model,
    sweeps: Iterable[AllenSweepBatch],
    *,
    lr: float = 1e-3,
    epochs: int = 10,
    chunk_size: int = 500,
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    voltage_weight: float = 1.0,
    spike_weight: float = 1.0,
    sparsity_weight: float = 1e-4,
    spike_tau_ms: float = 10.0,
    post_spike_mask_ms: float = 3.0,
) -> list[dict[str, float]]:
    """Fit the model to Allen sweeps with truncated BPTT.

    Notes:
        Each sweep is reset once at the beginning, then processed in
        ``chunk_size`` timesteps. ``functional.detach_net`` is called between
        chunks so the hidden state carries over while the graph stays bounded.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history: list[dict[str, float]] = []
    initialized = False

    for epoch in range(epochs):
        for sweep in sweeps:
            i_soma = sweep.i_soma.to(device=device, dtype=dtype)
            v_true = sweep.v_true.to(device=device, dtype=dtype)
            spike_true = sweep.spike_true.to(device=device, dtype=dtype)
            i_apical = None
            if sweep.i_apical is not None:
                i_apical = sweep.i_apical.to(device=device, dtype=dtype)

            if not initialized:
                functional.init_net_state(
                    model,
                    batch_size=i_soma.shape[1],
                    device=device,
                    dtype=dtype,
                )
                initialized = True

            functional.reset_net(
                model,
                batch_size=i_soma.shape[1],
                device=device,
                dtype=dtype,
            )
            optimizer.zero_grad()

            with environ.context(dt=float(sweep.dt_ms)):
                for start in range(0, i_soma.shape[0], chunk_size):
                    if start > 0:
                        functional.detach_net(model)
                    stop = min(start + chunk_size, i_soma.shape[0])

                    rollout = rollout_two_compartment(
                        model,
                        i_soma[start:stop],
                        None if i_apical is None else i_apical[start:stop],
                    )
                    losses = two_compartment_loss(
                        v_pred=rollout["v"],
                        spike_pred=rollout["spike"],
                        v_true=v_true[start:stop],
                        spike_true=spike_true[start:stop],
                        dt_ms=sweep.dt_ms,
                        w_Ca=getattr(model, "w_Ca", None),
                        voltage_weight=voltage_weight,
                        spike_weight=spike_weight,
                        sparsity_weight=sparsity_weight,
                        spike_tau_ms=spike_tau_ms,
                        post_spike_mask_ms=post_spike_mask_ms,
                    )
                    losses["total"].backward()
                    optimizer.step()
                    optimizer.zero_grad()

                    history.append(
                        {
                            "epoch": float(epoch),
                            "specimen_id": float(sweep.specimen_id),
                            "sweep_number": float(sweep.sweep_number),
                            "chunk_start": float(start),
                            "total_loss": float(losses["total"].detach().cpu()),
                            "voltage_loss": float(losses["voltage"].detach().cpu()),
                            "spike_loss": float(losses["spike"].detach().cpu()),
                            "sparsity_loss": float(
                                losses["sparsity"].detach().cpu()
                            ),
                        }
                    )
    return history


__all__ = [
    "AllenSweepBatch",
    "choose_current_clamp_sweeps",
    "detect_spikes_from_voltage",
    "exponential_filter_spike_train",
    "filter_mouse_visp_l5_pyramidal_cells",
    "fit_two_compartment_model",
    "get_cell_types_cache",
    "load_allen_sweep",
    "mask_post_spike_voltage_samples",
    "query_mouse_visp_l5_pyramidal_cells",
    "resample_trace",
    "rollout_two_compartment",
    "two_compartment_loss",
]
