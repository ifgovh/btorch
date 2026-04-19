"""Fit the two-compartment neuron to Allen Cell Types sweeps.

This example keeps the workflow intentionally small:

1. query mouse VISp layer-5 pyramidal candidates,
2. choose long-square current-clamp sweeps,
3. load one or more sweeps into time-first tensors,
4. fit :class:`btorch.models.neurons.TwoCompartmentGLIF` with truncated BPTT.

AllenSDK is required to run this script.
"""

from __future__ import annotations

import argparse

import torch

from btorch.analysis.two_compartment_fit import (
    choose_current_clamp_sweeps,
    fit_two_compartment_model,
    load_allen_sweep,
    query_mouse_visp_l5_pyramidal_cells,
)
from btorch.models.neurons import TwoCompartmentGLIF


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-file", type=str, default=None)
    parser.add_argument("--max-cells", type=int, default=1)
    parser.add_argument("--max-sweeps-per-cell", type=int, default=1)
    parser.add_argument("--dt-ms", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    cells = query_mouse_visp_l5_pyramidal_cells(manifest_file=args.manifest_file)
    if not cells:
        raise RuntimeError("No mouse VISp layer-5 pyramidal candidates were found.")

    sweeps = []
    for cell in cells[: args.max_cells]:
        specimen_id = int(cell["specimen_id"])
        from btorch.analysis.two_compartment_fit import get_cell_types_cache

        cache = get_cell_types_cache(args.manifest_file)
        sweep_records = cache.get_ephys_sweeps(specimen_id)
        chosen = choose_current_clamp_sweeps(sweep_records)
        for sweep in chosen[: args.max_sweeps_per_cell]:
            sweeps.append(
                load_allen_sweep(
                    specimen_id=specimen_id,
                    sweep_number=int(sweep["sweep_number"]),
                    dt_ms=args.dt_ms,
                    manifest_file=args.manifest_file,
                    cache=cache,
                )
            )

    if not sweeps:
        raise RuntimeError("No suitable current-clamp sweeps were found.")

    model = TwoCompartmentGLIF(
        n_neuron=1,
        tau_s=20.0,
        R_s=1.0,
        E_L=-70.0,
        tau_a=120.0,
        w_Ca=0.5,
        theta_Ca=0.5,
        w_sa=0.5,
        w_as=1.0,
        trainable_param={
            "tau_s",
            "R_s",
            "E_L",
            "tau_a",
            "w_Ca",
            "theta_Ca",
            "w_sa",
            "w_as",
            "v_threshold",
            "v_reset",
        },
    ).to(device)

    history = fit_two_compartment_model(
        model,
        sweeps,
        lr=args.lr,
        epochs=args.epochs,
        chunk_size=args.chunk_size,
        device=device,
    )
    final = history[-1]
    print("Finished fitting")
    print(f"  total_loss:    {final['total_loss']:.6f}")
    print(f"  voltage_loss:  {final['voltage_loss']:.6f}")
    print(f"  spike_loss:    {final['spike_loss']:.6f}")
    print(f"  sparsity_loss: {final['sparsity_loss']:.6f}")


if __name__ == "__main__":
    main()
