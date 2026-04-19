# Windows Handoff For Two-Compartment Fitting

## Purpose

This note explains how to continue the two-compartment neuron fitting workflow
on a Windows machine.

The current macOS work has been stopped intentionally, and the temporary
environment `btorch-fit-py311` has been deleted.

## Current Repository Status

The code changes that were added locally are still present in this repository.

New or modified files relevant to the fitting workflow:

- `btorch/models/neurons/two_compartment.py`
- `btorch/models/neurons/__init__.py`
- `btorch/models/__init__.py`
- `btorch/analysis/two_compartment_fit.py`
- `btorch/analysis/__init__.py`
- `examples/allen_two_compartment_fit.py`
- `tests/models/neurons/test_two_compartment.py`
- `TWO_COMPARTMENT_FIT_REPORT.md`

Branch:

- `few-compartment-model`

## Important Status Clarification

The fitting pipeline has been implemented, but real Allen-data fitting has not
been completed yet.

That means:

- the model code exists
- the Allen preprocessing and fitting utilities exist
- the example script exists
- no final fitted biological parameters have been produced yet

## Recommended Windows Setup

Use Python 3.11 on Windows for AllenSDK compatibility.

Recommended environment name:

- `btorch-fit-py311`

Recommended tools:

- `micromamba` or `conda`
- Git

## Step 1: Move The Repository

On the Windows machine, get the repository with your current changes by using
one of these approaches:

### Option A: push branch and pull on Windows

From the current machine:

```bash
git add .
git commit -m "Add two-compartment neuron and Allen fitting pipeline"
git push origin few-compartment-model
```

Then on Windows:

```bash
git clone <your-repo-url>
cd btorch
git checkout few-compartment-model
```

### Option B: copy the working tree directly

Copy the full repository folder to Windows, including the `.git` directory if
you want to preserve branch history locally.

## Step 2: Create The Windows Environment

### Micromamba

```powershell
micromamba create -y -n btorch-fit-py311 -c conda-forge python=3.11 pip
```

### Conda

```powershell
conda create -y -n btorch-fit-py311 python=3.11 pip
```

## Step 3: Install Runtime Dependencies

Install the minimum fitting stack first:

```powershell
micromamba run -n btorch-fit-py311 pip install allensdk torch torchvision jaxtyping spikingjelly pytest
```

Then install the repository itself:

```powershell
micromamba run -n btorch-fit-py311 pip install -e . --config-settings editable_mode=strict
```

If you prefer conda instead of micromamba, replace `micromamba run -n ...` with
`conda run -n ...`.

## Step 4: Verify The Local Code

Run the focused tests first:

```powershell
micromamba run -n btorch-fit-py311 pytest tests/models/neurons/test_two_compartment.py -q
```

Optionally syntax-check the example:

```powershell
micromamba run -n btorch-fit-py311 python -m compileall examples/allen_two_compartment_fit.py
```

## Step 5: Start The First Real Fit

Run a small first-pass fit on one cell and one sweep:

```powershell
micromamba run -n btorch-fit-py311 python examples/allen_two_compartment_fit.py --max-cells 1 --max-sweeps-per-cell 1 --epochs 5 --chunk-size 500 --dt-ms 0.5
```

This is intentionally conservative so the first run is easier to debug.

## Step 6: Inspect Learned Parameters

After the first fit works, inspect the trained parameters by adding or running
something like:

```python
for name, param in model.named_parameters():
    print(name, param.detach().cpu())
```

You will likely also want to save:

- fitted parameters
- optimizer state
- selected specimen id and sweep number
- loss curves
- predicted vs recorded traces

## Recommended Next Improvements On Windows

- save checkpoints during fitting
- add plotting for `v_true` vs `v_pred`
- add plotting for true vs predicted spike trains
- tighten Allen sweep filtering
- verify current and voltage unit conventions carefully
- consider positivity constraints for `tau_s`, `tau_a`, and `R_s`

## Known Caveat

AllenSDK failed to install cleanly in the existing macOS Python 3.12 workflow
because it pulled an older NumPy path incompatible with that setup.

That is the reason Windows continuation should use Python 3.11.

## Short Summary

What is done:

- model implementation
- fitting pipeline implementation
- example script
- focused tests

What is not done:

- real Allen-data fitting results
- final fitted parameter report
- checkpointed trained model
