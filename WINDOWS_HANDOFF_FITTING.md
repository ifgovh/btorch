# Windows Handoff For Two-Compartment Fitting

## Current Best Checkpoints

Repository branch:

- `few-compartment-model`

Important commits:

- `579d54c`: adaptive-threshold checkpoint
- `90e724c`: spike-initiation fitting strategy checkpoint

The best fitting result so far is based on the spike-initiation direction, not
the earlier adaptive-threshold-only checkpoint.

## Current Best Artifact

Best run folder:

- [artifacts/two_compartment_fit_sustained_selection_v5_spikeinit](./artifacts/two_compartment_fit_sustained_selection_v5_spikeinit)

Best held-out result:

- specimen `566506157`
- test sweep `51`
- true spikes `17`
- predicted spikes `15`
- spike count error `2`
- spike timing F1 `0.75`
- precision `0.80`
- recall `0.7059`
- voltage RMSE `6.07 mV`
- voltage R2 `0.529`

Best held-out plot:

- [fit_specimen_566506157_sweep_51_0.png](./artifacts/two_compartment_fit_sustained_selection_v5_spikeinit/test/fit_specimen_566506157_sweep_51_0.png)

## What Actually Changed

The current fitting workflow is no longer just “fit the first VISp L5 cell”.

It now:

- queries multiple mouse VISp L5 pyramidal candidates
- ranks them by sustained-spiking richness
- rejects onset-only long-square sweeps when better sustained sweeps exist
- fits a minimally extended two-compartment model with an exponential
  spike-initiation term
- uses staged bounded fitting to get into the right firing regime

## Environment

Use Python 3.11 on Windows for AllenSDK compatibility.

Recommended environment:

- `btorch-fit-py311`

Create it with micromamba:

```powershell
micromamba create -y -n btorch-fit-py311 -c conda-forge python=3.11 pip
```

Install dependencies:

```powershell
micromamba run -n btorch-fit-py311 pip install allensdk torch torchvision jaxtyping spikingjelly pytest ruff
micromamba run -n btorch-fit-py311 pip install -e . --config-settings editable_mode=strict
```

## Verify Before Running

```powershell
micromamba run -n btorch-fit-py311 python -m ruff check btorch/models/neurons/two_compartment.py btorch/analysis/two_compartment_fit.py examples/allen_two_compartment_fit.py tests/models/neurons/test_two_compartment.py
micromamba run -n btorch-fit-py311 python -m pytest tests/models/neurons/test_two_compartment.py -q
```

## Reproduce The Best Current Run

```powershell
micromamba run -n btorch-fit-py311 python examples/allen_two_compartment_fit.py --method staged --max-cells 1 --candidate-cells 8 --max-sweeps-per-cell 4 --test-fraction 0.25 --dt-ms 0.5 --global-maxiter 1 --global-popsize 3 --local-maxiter 3 --tbptt-refine-epochs 0 --output-dir artifacts/two_compartment_fit_sustained_selection_v5_spikeinit
```

## What To Build From

If continuing the work, build from:

- commit `90e724c`
- artifact folder `artifacts/two_compartment_fit_sustained_selection_v5_spikeinit`

Not from the earlier weak-specimen runs, because those long-square sweeps were
too close to onset-only firing and gave misleading conclusions.

## What We Learned

- Data selection mattered a lot.
- Adaptive threshold alone was not enough.
- A minimal spike-initiation nonlinearity was the first change that broke the
  zero-spike failure mode.
- Later voltage-polish ideas were not yet clearly better than the current best
  spike-initiation run.

## Recommended Next Work

- preserve the spike-initiation extension
- improve voltage fit without losing the held-out spike gains
- compare across a few more sustained-spiking specimens instead of relying on
  one cell only
