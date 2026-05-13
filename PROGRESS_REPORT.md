# Progress Report

## 2026-04-23

- Commit ID: `d4fa6b1`
- Purpose: Update repository agent instructions for function documentation and
  mandatory commit/report workflow.
- Key changes:
  - Added a rule requiring every generated function to document motivation,
    arguments, return values, and side effects.
  - Added a rule requiring every change set to be committed and logged in this
    Markdown progress report with the commit ID.
- Verification status:
  - Instruction files updated locally.
  - Commit created successfully.

## 2026-05-14

- Commit ID: `50ceafa66be09a5394afc45df0d9b66782478ec4`
- Purpose: Preserve logical connectome neuron ordering when a recurrent layer
  dispatches compact heterogeneous neuron sub-populations.
- Key changes:
  - Extended `MixedNeuronPopulation` so each group can optionally declare its
    original logical neuron indices.
  - Indexed groups gather soma/apical inputs from those logical indices and
    scatter spikes and per-neuron attributes back to the full network order.
  - Added regression tests for spike scattering and apical-current routing.
- Verification status:
  - `ruff check btorch/models/neurons/mixed.py tests/models/neurons/test_mixed.py`
  - `pytest tests/models/neurons/test_mixed.py tests/models/test_mixed_comprehensive.py tests/models/test_hetero_rnn.py -q`
  - Cross-repo construction check using the Fashion-MNIST 4166-neuron
    connectome confirmed btorch imported from
    `/home/guozhang/b_torch_development` and TC/L5 indices map to 762
    original-order `l5et`/`l5it` neuron rows.
