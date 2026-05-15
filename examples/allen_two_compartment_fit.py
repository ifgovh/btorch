"""Fit the two-compartment neuron to Allen Cell Types sweeps.

This example now uses a practical medium-sized dataset for one specimen:

1. query mouse VISp layer-5 pyramidal candidates,
2. pick representative long-square current-clamp sweeps,
3. split those sweeps into train and test sets,
4. fit :class:`btorch.models.neurons.TwoCompartmentGLIF` on the train set,
5. report train/test metrics and plots.

AllenSDK is required to run this script.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch

from btorch.analysis.two_compartment_fit import (
    DEFAULT_TWO_COMPARTMENT_PARAM_BOUNDS,
    choose_current_clamp_sweeps,
    derive_population_parameter_bounds,
    evaluate_fit_across_sweeps,
    extract_model_parameters,
    fit_two_compartment_model,
    load_allen_sweep,
    load_model_parameters,
    query_mouse_visp_l5_pyramidal_cells,
    save_fit_report,
)
from btorch.models.neurons import TwoCompartmentGLIF


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-file", type=str, default=None)
    parser.add_argument("--specimen-ids", type=int, nargs="*", default=None)
    parser.add_argument("--max-cells", type=int, default=1)
    parser.add_argument("--candidate-cells", type=int, default=8)
    parser.add_argument("--max-sweeps-per-cell", type=int, default=6)
    parser.add_argument("--test-fraction", type=float, default=0.33)
    parser.add_argument("--dt-ms", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--tbptt-refine-epochs", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--tbptt-refine-lr", type=float, default=2e-4)
    parser.add_argument("--voltage-weight", type=float, default=1.0)
    parser.add_argument("--spike-weight", type=float, default=5.0)
    parser.add_argument("--spike-count-weight", type=float, default=20.0)
    parser.add_argument("--spike-timing-weight", type=float, default=10.0)
    parser.add_argument("--spike-match-window-ms", type=float, default=10.0)
    parser.add_argument("--sparsity-weight", type=float, default=1e-4)
    parser.add_argument(
        "--method",
        type=str,
        choices=("hybrid", "global", "tbptt", "staged"),
        default="staged",
    )
    parser.add_argument(
        "--global-strategy",
        type=str,
        choices=("de", "cem"),
        default="de",
    )
    parser.add_argument("--global-maxiter", type=int, default=20)
    parser.add_argument("--global-popsize", type=int, default=8)
    parser.add_argument("--local-maxiter", type=int, default=50)
    parser.add_argument("--trim-to-stimulus", action="store_true", default=True)
    parser.add_argument("--no-trim-to-stimulus", action="store_false",
                        dest="trim_to_stimulus")
    parser.add_argument("--trim-pre-pad-ms", type=float, default=100.0)
    parser.add_argument("--trim-post-pad-ms", type=float, default=150.0)
    parser.add_argument("--trim-activity-threshold-pa", type=float, default=5.0)
    parser.add_argument("--prefit-cells", type=int, default=0)
    parser.add_argument(
        "--prefit-method",
        type=str,
        choices=("staged", "global", "tbptt", "hybrid"),
        default="tbptt",
    )
    parser.add_argument("--prefit-epochs", type=int, default=1)
    parser.add_argument("--prefit-global-maxiter", type=int, default=2)
    parser.add_argument("--prefit-global-popsize", type=int, default=4)
    parser.add_argument("--prefit-local-maxiter", type=int, default=10)
    parser.add_argument("--prior-margin-fraction", type=float, default=0.2)
    parser.add_argument("--prior-min-fraction-default", type=float, default=0.2)
    parser.add_argument("--zero-shot-cells", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/two_compartment_fit",
    )
    parser.add_argument("--device", type=str, default=None)
    return parser


@dataclass(frozen=True)
class SweepSummary:
    """Selection-time summary for one Allen sweep."""

    spike_count: int
    amplitude_pa: float
    stimulus_start_ms: float
    stimulus_stop_ms: float
    first_spike_ms: float | None
    last_spike_ms: float | None
    spike_span_ms: float
    onset_delay_ms: float | None
    is_onset_only: bool
    is_sustained_lowrate: bool
    is_sustained_highrate: bool


def _sample_evenly(items: list, count: int) -> list:
    if count <= 0 or not items:
        return []
    if count >= len(items):
        return list(items)
    if count == 1:
        return [items[len(items) // 2]]
    target_positions = torch.linspace(0, len(items) - 1, steps=count)
    used: set[int] = set()
    selected_indices: list[int] = []
    for position in target_positions:
        candidate = int(round(position.item()))
        if candidate not in used:
            used.add(candidate)
            selected_indices.append(candidate)
            continue
        for radius in range(1, len(items)):
            left = candidate - radius
            right = candidate + radius
            if left >= 0 and left not in used:
                used.add(left)
                selected_indices.append(left)
                break
            if right < len(items) and right not in used:
                used.add(right)
                selected_indices.append(right)
                break
    selected_indices.sort()
    return [items[index] for index in selected_indices[:count]]


def _summarize_sweep(sweep) -> SweepSummary:
    """Extract sustained-spiking features used for sweep selection."""
    current = sweep.i_soma[:, 0, 0]
    current_abs = current.abs()
    amplitude_pa = float(current_abs.max().item())
    active = current_abs >= max(5.0, 0.1 * amplitude_pa)
    active_idx = torch.nonzero(active).flatten()
    if active_idx.numel() > 0:
        start_idx = int(active_idx[0].item())
        stop_idx = int(active_idx[-1].item())
    else:
        start_idx = 0
        stop_idx = int(current.shape[0] - 1)

    spikes = torch.nonzero(sweep.spike_true[:, 0, 0] > 0.5).flatten()
    spike_count = int(spikes.numel())
    if spike_count > 0:
        first_spike_ms = float(spikes[0].item() * sweep.dt_ms)
        last_spike_ms = float(spikes[-1].item() * sweep.dt_ms)
        spike_span_ms = last_spike_ms - first_spike_ms
        onset_delay_ms = first_spike_ms - float(start_idx * sweep.dt_ms)
    else:
        first_spike_ms = None
        last_spike_ms = None
        spike_span_ms = 0.0
        onset_delay_ms = None

    is_onset_only = (
        spike_count == 1
        or (spike_count == 2 and spike_span_ms < 20.0)
        or (
            spike_count > 0
            and onset_delay_ms is not None
            and onset_delay_ms < 25.0
            and spike_span_ms < 25.0
        )
    )
    is_sustained_lowrate = spike_count >= 2 and spike_span_ms >= 25.0
    is_sustained_highrate = spike_count >= 6 or spike_span_ms >= 80.0
    return SweepSummary(
        spike_count=spike_count,
        amplitude_pa=amplitude_pa,
        stimulus_start_ms=float(start_idx * sweep.dt_ms),
        stimulus_stop_ms=float(stop_idx * sweep.dt_ms),
        first_spike_ms=first_spike_ms,
        last_spike_ms=last_spike_ms,
        spike_span_ms=spike_span_ms,
        onset_delay_ms=onset_delay_ms,
        is_onset_only=is_onset_only,
        is_sustained_lowrate=is_sustained_lowrate,
        is_sustained_highrate=is_sustained_highrate,
    )


def _cell_richness_score(loaded: list) -> tuple[float, float, float]:
    """Rank cells by how informative their long-square sweeps are."""
    summaries = [_summarize_sweep(sweep) for sweep in loaded]
    sustained_count = sum(
        summary.is_sustained_lowrate or summary.is_sustained_highrate
        for summary in summaries
    )
    highrate_count = sum(summary.is_sustained_highrate for summary in summaries)
    max_spike_count = max((summary.spike_count for summary in summaries), default=0)
    return (float(sustained_count), float(highrate_count), float(max_spike_count))


def _select_representative_sweeps(loaded: list, total_count: int) -> list:
    summaries = {id(sweep): _summarize_sweep(sweep) for sweep in loaded}
    subthreshold = [
        sweep for sweep in loaded if summaries[id(sweep)].spike_count == 0
    ]
    sustained_lowrate = [
        sweep for sweep in loaded if summaries[id(sweep)].is_sustained_lowrate
    ]
    sustained_highrate = [
        sweep
        for sweep in loaded
        if summaries[id(sweep)].is_sustained_highrate
        and sweep not in sustained_lowrate
    ]
    fallback_spiking = [
        sweep
        for sweep in loaded
        if summaries[id(sweep)].spike_count > 0
        and not summaries[id(sweep)].is_onset_only
        and sweep not in sustained_lowrate
        and sweep not in sustained_highrate
    ]

    sustained_lowrate.sort(
        key=lambda sweep: (
            summaries[id(sweep)].spike_count,
            summaries[id(sweep)].spike_span_ms,
            summaries[id(sweep)].amplitude_pa,
        )
    )
    sustained_highrate.sort(
        key=lambda sweep: (
            summaries[id(sweep)].spike_count,
            summaries[id(sweep)].spike_span_ms,
            summaries[id(sweep)].amplitude_pa,
        ),
        reverse=True,
    )
    fallback_spiking.sort(
        key=lambda sweep: (
            summaries[id(sweep)].spike_count,
            summaries[id(sweep)].spike_span_ms,
            summaries[id(sweep)].amplitude_pa,
        ),
        reverse=True,
    )
    subthreshold.sort(
        key=lambda sweep: summaries[id(sweep)].amplitude_pa,
        reverse=True,
    )

    target_subthreshold = min(len(subthreshold), max(1, total_count // 4))
    target_lowrate = min(
        len(sustained_lowrate),
        max(1 if sustained_lowrate else 0, total_count // 3),
    )
    target_highrate = min(
        len(sustained_highrate),
        max(1 if sustained_highrate else 0, total_count // 3),
    )
    selected: list = []
    selected_ids: set[int] = set()

    def extend_unique(candidates: list) -> None:
        for sweep in candidates:
            sweep_id = id(sweep)
            if sweep_id not in selected_ids:
                selected.append(sweep)
                selected_ids.add(sweep_id)

    extend_unique(_sample_evenly(subthreshold, target_subthreshold))
    extend_unique(_sample_evenly(sustained_lowrate, target_lowrate))
    extend_unique(_sample_evenly(sustained_highrate, target_highrate))

    remaining = total_count - len(selected)
    if remaining > 0:
        extend_unique(_sample_evenly(fallback_spiking, remaining))

    if len(selected) < total_count:
        extras = [
            sweep
            for sweep in loaded
            if id(sweep) not in selected_ids and not summaries[id(sweep)].is_onset_only
        ]
        extras.sort(
            key=lambda sweep: (
                summaries[id(sweep)].spike_count,
                summaries[id(sweep)].spike_span_ms,
                summaries[id(sweep)].amplitude_pa,
            ),
            reverse=True,
        )
        extend_unique(extras[: total_count - len(selected)])

    selected.sort(
        key=lambda sweep: (
            summaries[id(sweep)].is_sustained_highrate,
            summaries[id(sweep)].is_sustained_lowrate,
            summaries[id(sweep)].spike_count,
            summaries[id(sweep)].spike_span_ms,
            summaries[id(sweep)].amplitude_pa,
        ),
        reverse=True,
    )
    return selected[:total_count]


def _stratified_split_sweeps(sweeps: list, test_fraction: float) -> tuple[list, list]:
    if not sweeps:
        return [], []
    if not 0.0 < test_fraction < 1.0:
        raise ValueError(f"test_fraction must be in (0, 1), got {test_fraction}.")

    summaries = {id(sweep): _summarize_sweep(sweep) for sweep in sweeps}
    highrate = [sweep for sweep in sweeps if summaries[id(sweep)].is_sustained_highrate]
    lowrate = [
        sweep
        for sweep in sweeps
        if summaries[id(sweep)].is_sustained_lowrate
        and not summaries[id(sweep)].is_sustained_highrate
    ]
    subthreshold = [sweep for sweep in sweeps if summaries[id(sweep)].spike_count == 0]
    residual_spiking = [
        sweep
        for sweep in sweeps
        if summaries[id(sweep)].spike_count > 0
        and sweep not in highrate
        and sweep not in lowrate
    ]

    def split_group(group: list) -> tuple[list, list]:
        if len(group) <= 1:
            return group, []
        test_count = max(1, int(math.ceil(len(group) * test_fraction)))
        test = _sample_evenly(group, test_count)
        train = [sweep for sweep in group if sweep not in test]
        return train, test

    train_highrate, test_highrate = split_group(highrate)
    train_lowrate, test_lowrate = split_group(lowrate)
    train_subthreshold, test_subthreshold = split_group(subthreshold)
    train_residual, test_residual = split_group(residual_spiking)
    train = train_highrate + train_lowrate + train_subthreshold + train_residual
    test = test_highrate + test_lowrate + test_subthreshold + test_residual

    if not train and test:
        train = test[:-1]
        test = test[-1:]
    return train, test


def _ensure_spiking_coverage(
    train: list,
    test: list,
    selected: list,
) -> tuple[list, list]:
    """Ensure both splits contain spiking sweeps when available."""
    spiking_selected = [
        sweep for sweep in selected if float(sweep.spike_true.sum().item()) > 0.0
    ]
    if len(spiking_selected) < 2:
        return train, test

    train_spiking = [
        sweep for sweep in train if float(sweep.spike_true.sum().item()) > 0.0
    ]
    test_spiking = [
        sweep for sweep in test if float(sweep.spike_true.sum().item()) > 0.0
    ]

    if not train_spiking and test_spiking:
        candidate = max(
            test_spiking,
            key=lambda sweep: float(sweep.spike_true.sum().item()),
        )
        test = [sweep for sweep in test if sweep is not candidate]
        train.append(candidate)
        train_spiking.append(candidate)

    if not test_spiking and train_spiking:
        candidate = min(
            train_spiking,
            key=lambda sweep: float(sweep.spike_true.sum().item()),
        )
        if len(train) > 1:
            train = [sweep for sweep in train if sweep is not candidate]
            test.append(candidate)

    return train, test


def _ensure_lowrate_coverage(
    train: list,
    test: list,
    selected: list,
) -> tuple[list, list]:
    """Ensure both splits contain low-rate sweeps when available."""
    lowrate_selected = [
        sweep
        for sweep in selected
        if 0.0 < float(sweep.spike_true.sum().item()) <= 5.0
    ]
    if len(lowrate_selected) < 2:
        return train, test

    train_lowrate = [
        sweep for sweep in train if 0.0 < float(sweep.spike_true.sum().item()) <= 5.0
    ]
    test_lowrate = [
        sweep for sweep in test if 0.0 < float(sweep.spike_true.sum().item()) <= 5.0
    ]

    if not train_lowrate and test_lowrate:
        candidate = min(
            test_lowrate,
            key=lambda sweep: float(sweep.spike_true.sum().item()),
        )
        test = [sweep for sweep in test if sweep is not candidate]
        train.append(candidate)
        train_lowrate.append(candidate)

    if not test_lowrate and train_lowrate and len(train) > 1:
        candidate = min(
            train_lowrate,
            key=lambda sweep: float(sweep.spike_true.sum().item()),
        )
        train = [sweep for sweep in train if sweep is not candidate]
        test.append(candidate)

    return train, test


def _ensure_sustained_coverage(
    train: list,
    test: list,
    selected: list,
) -> tuple[list, list]:
    """Ensure train and test both include sustained spiking sweeps when possible."""
    summaries = {id(sweep): _summarize_sweep(sweep) for sweep in selected}
    sustained_selected = [
        sweep
        for sweep in selected
        if summaries[id(sweep)].is_sustained_lowrate
        or summaries[id(sweep)].is_sustained_highrate
    ]
    if len(sustained_selected) < 2:
        return train, test

    train_sustained = [
        sweep
        for sweep in train
        if summaries.get(id(sweep), _summarize_sweep(sweep)).is_sustained_lowrate
        or summaries.get(id(sweep), _summarize_sweep(sweep)).is_sustained_highrate
    ]
    test_sustained = [
        sweep
        for sweep in test
        if summaries.get(id(sweep), _summarize_sweep(sweep)).is_sustained_lowrate
        or summaries.get(id(sweep), _summarize_sweep(sweep)).is_sustained_highrate
    ]

    if not train_sustained and test_sustained:
        candidate = max(
            test_sustained,
            key=lambda sweep: (
                summaries.get(id(sweep), _summarize_sweep(sweep)).spike_count,
                summaries.get(id(sweep), _summarize_sweep(sweep)).spike_span_ms,
            ),
        )
        test = [sweep for sweep in test if sweep is not candidate]
        train.append(candidate)

    if not test_sustained and train_sustained and len(train) > 1:
        candidate = min(
            train_sustained,
            key=lambda sweep: (
                summaries.get(id(sweep), _summarize_sweep(sweep)).spike_count,
                summaries.get(id(sweep), _summarize_sweep(sweep)).spike_span_ms,
            ),
        )
        train = [sweep for sweep in train if sweep is not candidate]
        test.append(candidate)

    return train, test


def _build_default_model(device: str) -> TwoCompartmentGLIF:
    return TwoCompartmentGLIF(
        n_neuron=1,
        tau_s=25.0,
        R_s=0.08,
        E_L=-72.0,
        tau_a=80.0,
        tau_th=50.0,
        delta_th=2.0,
        delta_T=2.0,
        w_Ca=0.0,
        theta_Ca=3.0,
        w_sa=0.0,
        w_as=0.0,
        v_threshold=-46.0,
        v_reset=-70.0,
        trainable_param={
            "tau_s",
            "R_s",
            "E_L",
            "tau_a",
            "tau_th",
            "delta_th",
            "delta_T",
            "w_Ca",
            "theta_Ca",
            "w_sa",
            "w_as",
            "v_threshold",
            "v_reset",
        },
    ).to(device)


def _median_parameter_set(parameter_sets: list[dict[str, float]]) -> dict[str, float]:
    if not parameter_sets:
        return {}
    names = parameter_sets[0].keys()
    median_parameters = {}
    for name in names:
        values = [float(params[name]) for params in parameter_sets if name in params]
        if values:
            median_parameters[name] = float(torch.tensor(values).median().item())
    return median_parameters


def _fit_model_bundle(
    *,
    model: TwoCompartmentGLIF,
    train_sweeps: list,
    test_sweeps: list,
    output_dir: Path,
    args,
    param_bounds: dict[str, tuple[float, float]] | None = None,
    global_maxiter: int | None = None,
    global_popsize: int | None = None,
    local_maxiter: int | None = None,
    method: str | None = None,
    epochs: int | None = None,
) -> dict:
    history = fit_two_compartment_model(
        model,
        train_sweeps,
        method=args.method if method is None else method,
        lr=args.lr,
        epochs=args.epochs if epochs is None else epochs,
        chunk_size=args.chunk_size,
        voltage_weight=args.voltage_weight,
        spike_weight=args.spike_weight,
        spike_count_weight=args.spike_count_weight,
        spike_timing_weight=args.spike_timing_weight,
        spike_match_window_ms=args.spike_match_window_ms,
        sparsity_weight=args.sparsity_weight,
        global_strategy=args.global_strategy,
        global_maxiter=(
            args.global_maxiter if global_maxiter is None else global_maxiter
        ),
        global_popsize=(
            args.global_popsize if global_popsize is None else global_popsize
        ),
        local_maxiter=args.local_maxiter if local_maxiter is None else local_maxiter,
        param_bounds=param_bounds or DEFAULT_TWO_COMPARTMENT_PARAM_BOUNDS,
        seed=args.seed,
        device=model.v_threshold.device,
    )
    if args.tbptt_refine_epochs > 0:
        tbptt_history = fit_two_compartment_model(
            model,
            train_sweeps,
            method="tbptt",
            lr=args.tbptt_refine_lr,
            epochs=args.tbptt_refine_epochs,
            chunk_size=args.chunk_size,
            voltage_weight=max(args.voltage_weight, 2.0),
            spike_weight=args.spike_weight,
            spike_count_weight=max(args.spike_count_weight, 25.0),
            spike_timing_weight=max(args.spike_timing_weight, 12.0),
            spike_match_window_ms=args.spike_match_window_ms,
            sparsity_weight=args.sparsity_weight,
            device=model.v_threshold.device,
        )
        history.extend(tbptt_history)

    train_evaluations, train_aggregate = evaluate_fit_across_sweeps(
        model,
        train_sweeps,
        device=model.v_threshold.device,
        spike_count_weight=args.spike_count_weight,
        spike_timing_weight=args.spike_timing_weight,
        spike_match_window_ms=args.spike_match_window_ms,
    )
    train_paths = save_fit_report(
        model,
        train_evaluations,
        train_aggregate,
        history,
        output_dir=output_dir / "train",
    )

    test_aggregate: dict[str, float] = {}
    test_paths: dict[str, Path] = {}
    if test_sweeps:
        test_evaluations, test_aggregate = evaluate_fit_across_sweeps(
            model,
            test_sweeps,
            device=model.v_threshold.device,
            spike_count_weight=args.spike_count_weight,
            spike_timing_weight=args.spike_timing_weight,
            spike_match_window_ms=args.spike_match_window_ms,
        )
        test_paths = save_fit_report(
            model,
            test_evaluations,
            test_aggregate,
            [],
            output_dir=output_dir / "test",
        )
    else:
        test_evaluations = []

    return {
        "model": model,
        "history": history,
        "train_evaluations": train_evaluations,
        "train_aggregate": train_aggregate,
        "train_paths": train_paths,
        "test_evaluations": test_evaluations,
        "test_aggregate": test_aggregate,
        "test_paths": test_paths,
        "parameters": extract_model_parameters(model),
    }


def _evaluate_zero_shot(
    *,
    model: TwoCompartmentGLIF,
    candidate_payloads: list[tuple[int, list, tuple[float, float, float]]],
    used_specimen_ids: set[int],
    args,
    output_dir: Path,
) -> dict[str, float] | None:
    if args.zero_shot_cells <= 0:
        return None

    zero_shot_sweeps = []
    zero_shot_specimens = []
    for specimen_id, loaded, richness in candidate_payloads:
        if specimen_id in used_specimen_ids:
            continue
        selected = _select_representative_sweeps(loaded, args.max_sweeps_per_cell)
        if not selected:
            continue
        zero_shot_specimens.append(
            {
                "specimen_id": specimen_id,
                "richness": richness,
                "sweeps": [sweep.sweep_number for sweep in selected],
            }
        )
        zero_shot_sweeps.extend(selected)
        if len(zero_shot_specimens) >= args.zero_shot_cells:
            break

    if not zero_shot_sweeps:
        return None

    zero_shot_evaluations, zero_shot_aggregate = evaluate_fit_across_sweeps(
        model,
        zero_shot_sweeps,
        device=model.v_threshold.device,
        spike_count_weight=args.spike_count_weight,
        spike_timing_weight=args.spike_timing_weight,
        spike_match_window_ms=args.spike_match_window_ms,
    )
    save_fit_report(
        model,
        zero_shot_evaluations,
        zero_shot_aggregate,
        [],
        output_dir=output_dir / "zero_shot",
    )
    zero_shot_aggregate["n_zero_shot_specimens"] = float(len(zero_shot_specimens))
    with (output_dir / "zero_shot" / "selection.json").open("w", encoding="utf-8") as f:
        json.dump(zero_shot_specimens, f, indent=2)
    return zero_shot_aggregate


def _write_run_summary(
    *,
    output_dir: Path,
    train_aggregate: dict[str, float],
    test_aggregate: dict[str, float],
    zero_shot_aggregate: dict[str, float] | None,
    derived_bounds: dict[str, tuple[float, float]] | None,
    prefit_summaries: list[dict],
) -> None:
    lines = [
        "# Two-Compartment Allen Fit Summary",
        "",
        "## Final Shared Fit",
        "",
        f"- Train sweeps: {int(train_aggregate.get('n_sweeps', 0.0))}",
        f"- Train mean spike timing F1: "
        f"{train_aggregate.get('mean_spike_timing_f1', float('nan')):.4f}",
        f"- Train mean voltage RMSE: "
        f"{train_aggregate.get('mean_voltage_rmse', float('nan')):.4f}",
    ]
    if test_aggregate:
        lines.extend(
            [
                f"- Test sweeps: {int(test_aggregate.get('n_sweeps', 0.0))}",
                f"- Test mean spike timing F1: "
                f"{test_aggregate.get('mean_spike_timing_f1', float('nan')):.4f}",
                f"- Test mean spike count error: "
                f"{test_aggregate.get('mean_spike_count_error', float('nan')):.4f}",
                f"- Test mean voltage RMSE: "
                f"{test_aggregate.get('mean_voltage_rmse', float('nan')):.4f}",
            ]
        )
    if zero_shot_aggregate:
        zero_shot_count_error = zero_shot_aggregate.get(
            "mean_spike_count_error",
            float("nan"),
        )
        lines.extend(
            [
                "",
                "## Zero-Shot Evaluation",
                "",
                f"- Zero-shot sweeps: {int(zero_shot_aggregate.get('n_sweeps', 0.0))}",
                f"- Zero-shot mean spike timing F1: "
                f"{zero_shot_aggregate.get('mean_spike_timing_f1', float('nan')):.4f}",
                f"- Zero-shot mean spike count error: {zero_shot_count_error:.4f}",
                f"- Zero-shot mean voltage RMSE: "
                f"{zero_shot_aggregate.get('mean_voltage_rmse', float('nan')):.4f}",
            ]
        )
    if prefit_summaries:
        lines.extend(["", "## Prefit Cells", ""])
        lines.extend(
            f"- Specimen {summary['specimen_id']}: "
            f"test spike timing F1={summary['test_spike_timing_f1']:.4f}, "
            f"test spike count error={summary['test_spike_count_error']:.4f}"
            for summary in prefit_summaries
        )
    if derived_bounds:
        lines.extend(["", "## Derived Bounds", ""])
        lines.extend(
            f"- `{name}`: [{lower:.4f}, {upper:.4f}]"
            for name, (lower, upper) in sorted(derived_bounds.items())
        )

    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)

    cells = query_mouse_visp_l5_pyramidal_cells(manifest_file=args.manifest_file)
    if not cells:
        raise RuntimeError("No mouse VISp layer-5 pyramidal candidates were found.")
    explicit_specimen_ids = (
        {int(specimen_id) for specimen_id in args.specimen_ids}
        if args.specimen_ids
        else None
    )
    if explicit_specimen_ids is not None:
        cells = [
            cell
            for cell in cells
            if int(cell.get("specimen_id") or cell.get("id")) in explicit_specimen_ids
        ]
        if not cells:
            raise RuntimeError(
                "None of the requested specimen IDs matched the filtered "
                "mouse VISp layer-5 pyramidal candidates."
            )

    candidate_payloads = []
    if explicit_specimen_ids is not None:
        search_cells = cells
    else:
        search_cells = cells[: max(args.max_cells, args.candidate_cells)]
    from btorch.analysis.two_compartment_fit import get_cell_types_cache

    cache = get_cell_types_cache(args.manifest_file)
    for cell in search_cells:
        specimen_id = int(cell.get("specimen_id") or cell.get("id"))
        sweep_records = cache.get_ephys_sweeps(specimen_id)
        chosen = choose_current_clamp_sweeps(sweep_records)
        loaded = [
            load_allen_sweep(
                specimen_id=specimen_id,
                sweep_number=int(sweep["sweep_number"]),
                dt_ms=args.dt_ms,
                manifest_file=args.manifest_file,
                cache=cache,
                trim_to_stimulus=args.trim_to_stimulus,
                trim_pre_pad_ms=args.trim_pre_pad_ms,
                trim_post_pad_ms=args.trim_post_pad_ms,
                trim_activity_threshold_pa=args.trim_activity_threshold_pa,
            )
            for sweep in chosen
        ]
        if not loaded:
            continue
        candidate_payloads.append((specimen_id, loaded, _cell_richness_score(loaded)))

    candidate_payloads.sort(key=lambda item: item[2], reverse=True)
    if explicit_specimen_ids is not None:
        selected_payloads = candidate_payloads[: len(candidate_payloads)]
    else:
        selected_payloads = candidate_payloads[: args.max_cells]

    selected_splits = []
    train_sweeps = []
    test_sweeps = []
    for specimen_id, loaded, richness in selected_payloads:
        selected = _select_representative_sweeps(loaded, args.max_sweeps_per_cell)
        train_split, test_split = _stratified_split_sweeps(
            selected,
            args.test_fraction,
        )
        train_split, test_split = _ensure_spiking_coverage(
            train_split,
            test_split,
            selected,
        )
        train_split, test_split = _ensure_lowrate_coverage(
            train_split,
            test_split,
            selected,
        )
        train_split, test_split = _ensure_sustained_coverage(
            train_split,
            test_split,
            selected,
        )
        train_sweeps.extend(train_split)
        test_sweeps.extend(test_split)
        selected_splits.append(
            {
                "specimen_id": specimen_id,
                "richness": richness,
                "train_sweeps": train_split,
                "test_sweeps": test_split,
            }
        )
        print(
            "Selected specimen",
            specimen_id,
            "richness=",
            richness,
            "train=",
            [sweep.sweep_number for sweep in train_split],
            "test=",
            [sweep.sweep_number for sweep in test_split],
        )

    if not train_sweeps:
        raise RuntimeError("No suitable current-clamp sweeps were found.")

    prefit_count = args.prefit_cells
    if prefit_count <= 0 and args.max_cells > 1:
        prefit_count = len(selected_splits)

    prefit_parameter_sets: list[dict[str, float]] = []
    prefit_summaries: list[dict] = []
    for split in selected_splits[:prefit_count]:
        specimen_id = int(split["specimen_id"])
        specimen_model = _build_default_model(device)
        prefit_result = _fit_model_bundle(
            model=specimen_model,
            train_sweeps=list(split["train_sweeps"]),
            test_sweeps=list(split["test_sweeps"]),
            output_dir=output_dir / "prefit" / f"specimen_{specimen_id}",
            args=args,
            method=args.prefit_method,
            epochs=args.prefit_epochs,
            global_maxiter=args.prefit_global_maxiter,
            global_popsize=args.prefit_global_popsize,
            local_maxiter=args.prefit_local_maxiter,
        )
        prefit_parameter_sets.append(prefit_result["parameters"])
        prefit_summaries.append(
            {
                "specimen_id": specimen_id,
                "test_spike_timing_f1": float(
                    prefit_result["test_aggregate"].get("mean_spike_timing_f1", 0.0)
                ),
                "test_spike_count_error": float(
                    prefit_result["test_aggregate"].get("mean_spike_count_error", 0.0)
                ),
            }
        )

    derived_bounds = None
    median_parameters = {}
    if prefit_parameter_sets:
        derived_bounds = derive_population_parameter_bounds(
            prefit_parameter_sets,
            default_bounds=DEFAULT_TWO_COMPARTMENT_PARAM_BOUNDS,
            margin_fraction=args.prior_margin_fraction,
            min_fraction_of_default=args.prior_min_fraction_default,
        )
        median_parameters = _median_parameter_set(prefit_parameter_sets)

    model = _build_default_model(device)
    if median_parameters:
        load_model_parameters(model, median_parameters)
    shared_result = _fit_model_bundle(
        model=model,
        train_sweeps=train_sweeps,
        test_sweeps=test_sweeps,
        output_dir=output_dir,
        args=args,
        param_bounds=derived_bounds or DEFAULT_TWO_COMPARTMENT_PARAM_BOUNDS,
    )
    zero_shot_aggregate = _evaluate_zero_shot(
        model=model,
        candidate_payloads=candidate_payloads,
        used_specimen_ids={int(item["specimen_id"]) for item in selected_splits},
        args=args,
        output_dir=output_dir,
    )
    _write_run_summary(
        output_dir=output_dir,
        train_aggregate=shared_result["train_aggregate"],
        test_aggregate=shared_result["test_aggregate"],
        zero_shot_aggregate=zero_shot_aggregate,
        derived_bounds=derived_bounds,
        prefit_summaries=prefit_summaries,
    )

    final = shared_result["history"][-1]
    print("Finished fitting")
    print(f"  train_sweeps:   {len(train_sweeps)}")
    print(f"  test_sweeps:    {len(test_sweeps)}")
    print(f"  total_loss:    {final['total_loss']:.6f}")
    print(f"  voltage_loss:  {final['voltage_loss']:.6f}")
    print(f"  spike_loss:    {final['spike_loss']:.6f}")
    print(f"  spike_timing:  {final['spike_timing_loss']:.6f}")
    print(f"  sparsity_loss: {final['sparsity_loss']:.6f}")
    print("Train summary")
    for name, value in shared_result["train_aggregate"].items():
        print(f"  {name}: {value:.6f}")
    if shared_result["test_aggregate"]:
        print("Test summary")
        for name, value in shared_result["test_aggregate"].items():
            print(f"  {name}: {value:.6f}")
    if zero_shot_aggregate:
        print("Zero-shot summary")
        for name, value in zero_shot_aggregate.items():
            print(f"  {name}: {value:.6f}")
    print("Train artifacts")
    for name, path in shared_result["train_paths"].items():
        print(f"  {name}: {path}")
    if shared_result["test_paths"]:
        print("Test artifacts")
        for name, path in shared_result["test_paths"].items():
            print(f"  {name}: {path}")
    if zero_shot_aggregate:
        print(f"  zero_shot: {output_dir / 'zero_shot'}")
    print(f"  summary: {output_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
