# Two-Compartment Fitting Report

## Current Status

Real Allen-data fitting has now been executed in this repository.

The project has moved through three phases:

1. Infrastructure only
2. Real-data fitting on weak single-spike long-square sweeps
3. Real-data fitting on sustained-spiking long-square sweeps with a minimal
   spike-initiation model extension

The current best result is not perfect, but it is the first run that gets the
held-out spike count close while also recovering a meaningful fraction of spike
timing on a genuine sustained-spiking VISp L5 pyramidal specimen.

## Important Commits

- `579d54c`: adaptive-threshold checkpoint
- `90e724c`: spike-initiation fitting strategy checkpoint

The current working tree after `90e724c` keeps the spike-initiation direction
and updates the reports below.

## Best Current Real Run

Best artifact folder:

- [artifacts/two_compartment_fit_sustained_selection_v5_spikeinit](./artifacts/two_compartment_fit_sustained_selection_v5_spikeinit)

Selected specimen:

- `566506157`

Train sweeps:

- `52`, `64`, `53`

Test sweep:

- `51`

These sweeps are qualitatively much better than the earlier weak specimen
because they contain sustained firing over the long-square pulse:

- sweep `52`: `18` spikes over about `952 ms`
- sweep `64`: `11` spikes over about `904 ms`
- sweep `51` (test): `17` spikes over about `922.5 ms`

## Best Held-Out Metrics So Far

From:

- [artifacts/two_compartment_fit_sustained_selection_v5_spikeinit/test/fit_metrics.json](./artifacts/two_compartment_fit_sustained_selection_v5_spikeinit/test/fit_metrics.json)

Held-out spiking sweep metrics:

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

Test plot:

- [fit_specimen_566506157_sweep_51_0.png](./artifacts/two_compartment_fit_sustained_selection_v5_spikeinit/test/fit_specimen_566506157_sweep_51_0.png)

## Interpretation

This is the first result that clearly enters the correct firing regime on a
held-out sustained-spiking sweep.

What improved:

- the model no longer collapses to `0` spikes
- spike count is close on held-out data
- spike timing overlap is substantial

What is still weak:

- voltage fit degraded relative to the earlier non-spiking solutions
- the model still misses some spikes and adds a few extras
- training performance is less clean than the single held-out test result

So the current best model is better judged as:

- good direction for spike count and timing
- not yet a fully satisfactory joint spike-plus-voltage fit

## Model Changes That Mattered

The most important effective change was not just optimizer tuning.

The current best direction combined:

- better Allen sweep selection across multiple candidate cells
- rejection of onset-only sweeps when sustained-spiking sweeps are available
- a staged fitter that calibrates spike initiation explicitly
- a minimal exponential spike-initiation term in the somatic voltage update

The earlier adaptive-threshold-only variant did not solve the sustained-firing
problem by itself.

## Experiments Tried After The Best Run

Two follow-up ideas were tested after the best `v5_spikeinit` run:

1. voltage-focused extra staged refinement
2. TBPTT polish from the best spike-initiation solution

Neither produced a better overall tradeoff than `v5_spikeinit`, so that run is
still the recommended checkpoint to build from.

## Files To Inspect

- Model: [btorch/models/neurons/two_compartment.py](./btorch/models/neurons/two_compartment.py)
- Fitter: [btorch/analysis/two_compartment_fit.py](./btorch/analysis/two_compartment_fit.py)
- Example: [examples/allen_two_compartment_fit.py](./examples/allen_two_compartment_fit.py)
- Tests: [tests/models/neurons/test_two_compartment.py](./tests/models/neurons/test_two_compartment.py)

## Recommended Next Step

Keep the spike-initiation extension and improve voltage fit without losing the
held-out spike gains from `v5_spikeinit`.

The safest next work items are:

- tune the later fitting stages rather than reverting the model
- add a constrained voltage-polish pass only if spike-count error stays low
- compare multiple sustained-spiking specimens, not just one
