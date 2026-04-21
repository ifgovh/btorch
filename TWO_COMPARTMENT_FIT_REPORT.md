# Two-Compartment Fitting Report

## Status

The two-compartment neuron model and the Allen fitting pipeline have been
implemented on branch `few-compartment-model`, but the neuronal parameters have
not been fitted to real Allen Brain Institute data yet.

In other words:

- The code to fit parameters exists.
- The real AllenSDK download / preprocessing / optimization run has not been
  executed yet in this repository.
- No fitted parameter values or fitted checkpoints have been produced so far.

## What Was Implemented

### 1. Two-compartment neuron module

Implemented in:

- [btorch/models/neurons/two_compartment.py](./btorch/models/neurons/two_compartment.py)

This module includes:

- somatic voltage state `v`
- apical slow current state `i_a`
- delayed back-propagating current state `i_bap`
- trainable parameters:
  - `tau_s`
  - `R_s`
  - `E_L`
  - `tau_a`
  - `w_Ca`
  - `theta_Ca`
  - `w_sa`
  - `w_as`
- surrogate-gradient somatic spiking
- single-step and multi-step rollout
- `btorch`-compatible memory registration and reset behavior

### 2. Allen fitting pipeline

Implemented in:

- [btorch/analysis/two_compartment_fit.py](./btorch/analysis/two_compartment_fit.py)

This pipeline includes:

- AllenSDK-backed cell query helpers
- mouse VISp layer-5 pyramidal candidate filtering
- current-clamp sweep selection helpers
- Allen sweep loading and resampling
- spike extraction from voltage traces
- masked voltage loss
- smoothed spike-train loss
- `w_Ca` sparsity regularization
- a bounded global-search fitting path for poor initializations
- optional local polish and TBPTT refinement
- the original truncated BPTT training loop using:
  - `functional.reset_net`
  - `functional.detach_net`

### 3. Example entrypoint

Implemented in:

- [examples/allen_two_compartment_fit.py](./examples/allen_two_compartment_fit.py)

This is a runnable example script for:

- querying candidate Allen cells
- selecting sweeps
- loading them into tensors
- fitting the model with the implemented training loop

### 4. Tests

Implemented in:

- [tests/models/neurons/test_two_compartment.py](./tests/models/neurons/test_two_compartment.py)

Verified locally:

- targeted pytest passed
- compileall checks passed
- synthetic rollout / synthetic fitting smoke tests passed

## What Has NOT Been Done Yet

The following has not been completed yet:

- installing `allensdk` in the active environment
- downloading real Allen electrophysiology sweeps
- running the training script on real Allen data
- saving fitted parameters or checkpoints
- evaluating fit quality on held-out sweeps
- documenting final fitted values

## Direct Answer

No, the neuronal parameters are not fitted yet.

What exists now is the fitting infrastructure, not the final fitted model.

## Next Step To Produce Fitted Parameters

1. Install AllenSDK in the active environment:

```bash
micromamba run -n btorch pip install allensdk
```

2. Run the example fitting script:

```bash
micromamba run -n btorch python examples/allen_two_compartment_fit.py --method hybrid --max-cells 1 --max-sweeps-per-cell 1 --epochs 5 --chunk-size 500 --dt-ms 0.5
```

3. Inspect the learned parameters after training, for example:

```python
for name, param in model.named_parameters():
    print(name, param.detach().cpu())
```

## Recommended Follow-up

- run a first real Allen fitting pass
- save the trained state dict and learned parameters
- add a report comparing predicted vs recorded voltage and spike timing
- refine sweep filtering and unit normalization if the first fit is unstable
- use the hybrid method by default when the initial parameter guess is far from
  the biological regime
