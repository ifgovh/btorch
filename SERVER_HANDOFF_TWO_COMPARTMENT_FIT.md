# Server Handoff: Two-Compartment Allen Fitting

## Goal

Continue improving the Allen-data fitting workflow for
`TwoCompartmentGLIF` on a higher-compute server.

The current local work established a promising direction, but the broader
verification runs became too slow on this machine.

## Best Code Checkpoints

- `579d54c` `Add adaptive-threshold two-compartment fitting checkpoint`
- `90e724c` `Improve spike-init fitting strategy`
- `624a891` `Document current best fitting checkpoint`

If you want the strongest current fitting behavior, start from `90e724c` or
later on the same branch. `624a891` mostly updates documentation and example
defaults around that best checkpoint.

## Best Current Artifact

Best current run:

- [artifacts/two_compartment_fit_sustained_selection_v5_spikeinit](./artifacts/two_compartment_fit_sustained_selection_v5_spikeinit)

Best held-out result:

- specimen: `566506157`
- train sweeps: `52`, `64`, `53`
- test sweep: `51`

Held-out metrics from
[artifacts/two_compartment_fit_sustained_selection_v5_spikeinit/test/fit_metrics.json](./artifacts/two_compartment_fit_sustained_selection_v5_spikeinit/test/fit_metrics.json):

- `spike_count_true = 17`
- `spike_count_pred = 15`
- `spike_count_error = 2`
- `spike_timing_f1 = 0.75`
- `precision = 0.80`
- `recall = 0.7059`
- `matched_spikes = 12`
- `false_positive_spikes = 3`
- `false_negative_spikes = 5`
- `voltage_rmse = 6.07 mV`
- `voltage_r2 = 0.529`

Best held-out plot:

- [artifacts/two_compartment_fit_sustained_selection_v5_spikeinit/test/fit_specimen_566506157_sweep_51_0.png](./artifacts/two_compartment_fit_sustained_selection_v5_spikeinit/test/fit_specimen_566506157_sweep_51_0.png)

## What Was Implemented

### 1. Better data selection

The example script no longer just takes the first VISp L5 pyramidal specimen.

It now:

- queries multiple mouse VISp L5 pyramidal candidates
- ranks them by long-square sustained-spiking richness
- rejects onset-only long-square sweeps when stronger sustained-spiking sweeps
  are available
- tries to keep informative spiking sweeps across train/test

Relevant file:

- [examples/allen_two_compartment_fit.py](./examples/allen_two_compartment_fit.py)

### 2. Fitting-strategy improvements

The staged fitter now:

- uses count-first stage acceptance
- handles missing low-rate sweeps more gracefully
- calibrates spike initiation explicitly before later stages
- supports bounded global search and optional TBPTT refinement

Relevant file:

- [btorch/analysis/two_compartment_fit.py](./btorch/analysis/two_compartment_fit.py)

### 3. Minimal model changes

Two minimal model extensions were explored:

1. adaptive threshold state
2. exponential spike-initiation term

The adaptive-threshold-only variant did not solve the zero-spike failure mode.

The exponential spike-initiation term was the first change that broke the
all-silent regime and produced the current best held-out spike metrics.

Relevant file:

- [btorch/models/neurons/two_compartment.py](./btorch/models/neurons/two_compartment.py)

## What Worked

The best improvement came from the combination of:

- better sustained-spiking data selection
- staged spike-initiation fitting
- the exponential spike-initiation term (`delta_T`)

This was enough to move from:

- zero predicted spikes on sustained-spiking test sweeps

to:

- near-correct held-out spike count
- meaningful held-out spike timing overlap

## What Did Not Beat The Best Run

These were tried locally and did not clearly beat
`two_compartment_fit_sustained_selection_v5_spikeinit`:

### 1. Adaptive threshold only

Artifact:

- [artifacts/two_compartment_fit_sustained_selection_v3_dynamic_threshold](./artifacts/two_compartment_fit_sustained_selection_v3_dynamic_threshold)

Outcome:

- still collapsed to near-zero or zero spikes on sustained-spiking sweeps

### 2. Extra voltage-refine stage

Tried locally after `90e724c`.

Outcome:

- increased runtime significantly
- did not produce a clearly better overall spike-plus-voltage tradeoff

### 3. TBPTT polish from the best spike-initiation solution

Artifact:

- [artifacts/two_compartment_fit_sustained_selection_v6_tbptt_manual](./artifacts/two_compartment_fit_sustained_selection_v6_tbptt_manual)

Outcome:

- slightly changed count on held-out sweep
- worsened timing and voltage tradeoff relative to `v5_spikeinit`

## Broader Verification Result

I did attempt "more data" verification locally.

### Multi-cell refits

Tried:

- 3-specimen joint fit
- 2-specimen joint fit
- cheaper 2-specimen variants

Outcome:

- too slow to complete reliably on this machine

### Zero-shot cross-specimen evaluation

I evaluated the best fitted model from specimen `566506157` on additional
sustained-spiking specimens:

- `580537822`
- `504615116`
- `572410577`

Outcome:

- poor generalization
- the model mostly collapsed back toward `0` predicted spikes on those other
  specimens

So the current best result is promising, but still specimen-specific.

## Recommended Next Experiments On Server

### Priority 1: multi-specimen fitting with more compute

Use the current best configuration and actually complete runs across:

- top 3 to 5 sustained-spiking specimens
- 4 sweeps per specimen
- train/test split per specimen

Main question:

- does the spike-initiation direction still help when each specimen is
  independently refit with a larger compute budget?

### Priority 2: tighter priors from per-specimen fits

If the per-specimen fits work:

- collect fitted parameter ranges across 3 to 5 cells
- use those as tighter priors / bounds
- then try limited multi-cell fitting or cross-cell initialization

### Priority 3: preserve spike gains while improving voltage

Build from `v5_spikeinit`, not from older silent-regime runs.

Candidates:

- more compute on the existing staged global search
- per-stage increased polish iterations
- carefully constrained post-fit TBPTT
- only accept a refinement if spike count error stays near the current best

## Suggested Reproduction Commands

Best current baseline:

```powershell
python examples/allen_two_compartment_fit.py `
  --method staged `
  --max-cells 1 `
  --candidate-cells 8 `
  --max-sweeps-per-cell 4 `
  --test-fraction 0.25 `
  --dt-ms 0.5 `
  --global-maxiter 1 `
  --global-popsize 3 `
  --local-maxiter 3 `
  --tbptt-refine-epochs 0 `
  --output-dir artifacts/two_compartment_fit_sustained_selection_v5_spikeinit
```

Recommended stronger server validation:

```powershell
python examples/allen_two_compartment_fit.py `
  --method staged `
  --max-cells 3 `
  --candidate-cells 8 `
  --max-sweeps-per-cell 4 `
  --test-fraction 0.25 `
  --dt-ms 0.5 `
  --global-maxiter 3 `
  --global-popsize 4 `
  --local-maxiter 10 `
  --tbptt-refine-epochs 0 `
  --output-dir artifacts/two_compartment_fit_multi_verify_server
```

If that is stable, then try:

```powershell
python examples/allen_two_compartment_fit.py `
  --method staged `
  --max-cells 5 `
  --candidate-cells 12 `
  --max-sweeps-per-cell 4 `
  --test-fraction 0.25 `
  --dt-ms 0.5 `
  --global-maxiter 4 `
  --global-popsize 6 `
  --local-maxiter 15 `
  --tbptt-refine-epochs 0 `
  --output-dir artifacts/two_compartment_fit_multi_verify_server_large
```

## Files To Hand To Server Workflow

- [btorch/models/neurons/two_compartment.py](./btorch/models/neurons/two_compartment.py)
- [btorch/analysis/two_compartment_fit.py](./btorch/analysis/two_compartment_fit.py)
- [examples/allen_two_compartment_fit.py](./examples/allen_two_compartment_fit.py)
- [tests/models/neurons/test_two_compartment.py](./tests/models/neurons/test_two_compartment.py)
- [TWO_COMPARTMENT_FIT_REPORT.md](./TWO_COMPARTMENT_FIT_REPORT.md)
- [WINDOWS_HANDOFF_FITTING.md](./WINDOWS_HANDOFF_FITTING.md)

## Bottom Line

The work is no longer at the “infrastructure only” stage.

The current best configuration has already shown:

- a real held-out spike-count gain
- a real held-out spike-timing gain
- on a genuine sustained-spiking specimen

But it is not yet broadly validated across more cells.

That is the main job for the server run.
