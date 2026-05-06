# Mixed-Neuron Test Report

**Date:** 2026-05-03  
**Commit:** `2fe01c0` (`few-compartment-model` branch)  
**Test scope:** `MixedNeuronPopulation` + `HeterogeneousRecurrentNN`

---

## Summary

| Metric | Value |
|--------|-------|
| Total tests run | **27** |
| Passed | **27** |
| Failed | **0** |
| Skipped | **0** |
| Lint (`ruff`) | **clean** |

All tests passed on the first run. No flaky behaviour was observed.

---

## Test Design (what we checked)

### 1. Functional equivalence
- **test_mixed_single_group_equivalence_glif** — A `MixedNeuronPopulation` that wraps a *single* `GLIF3` group must produce **exactly the same spikes** as the raw `GLIF3` neuron.
- **test_mixed_single_group_equivalence_tc** — Same check for a single `TwoCompartmentGLIF` group, including with apical input.

*Why it matters:* The wrapper must be transparent when there is only one population.

### 2. Slicing & concatenation order
- **test_mixed_concatenation_order** — Two `GLIF3` groups are prepared so that the first group is forced to spike and the second is forced to stay silent. We verify that spikes appear in the **correct neuron-index order** after concatenation.

*Why it matters:* If slicing indices are off by one, recurrent connectivity (which is flat) will wire neurons to the wrong synapses.

### 3. Standalone multi-step dispatch
- **test_mixed_standalone_multi_step_matches_loop** — `mixed.multi_step_forward(x)` is compared against a manual Python loop over `mixed.single_step_forward(x[t])`.

*Why it matters:* Ensures the internal time-loop is consistent for users who call the neuron directly (outside `RecurrentNN`).

### 4. State management
- **test_mixed_state_init** — `init_net_state` initialises memories recursively for both `GLIF3` (v, Iasc) and `TwoCompartmentGLIF` (v, i_a, i_bap, theta_th).
- **test_mixed_state_reset_preserve_batch** — `reset_net_state` without an explicit `batch_size` argument preserves the existing batch dimension.
- **test_mixed_detach** — `detach_net` removes grad-graph attachments from all sub-population memories.

### 5. Apical input routing
- **test_mixed_apical_forward** — Apical input changes the internal apical current (`i_a`) of `TwoCompartmentGLIF` while a zero-apical run leaves it unchanged.
- **test_mixed_apical_only_tc_receives_it** — A strong apical signal is injected **only** into the TC slice. We verify that the GLIF slice is unaffected and the TC slice is affected.

### 6. Gradient flow
- **test_mixed_gradient_flow** — Back-propagation reaches **both** sub-populations' trainable parameters (`tau` for GLIF3, `tau_s` for TC).
- **test_mixed_recurrent_gradient_matches_baseline** — A `RecurrentNN` with a single-group `MixedNeuronPopulation` produces the **same gradients** as a plain `RecurrentNN` with the raw neuron, confirming that the wrapper does not alter the autograd graph.
- **test_hetero_rnn_gradient_flow** — Gradients flow back through `HeterogeneousRecurrentNN` to the external input `x`, the apical input `x_apical`, and both neuron parameters.

### 7. RecurrentNN integration
- **test_hetero_rnn_forward_shape** — `HeterogeneousRecurrentNN` with mixed neurons produces the expected `[T, batch, n_neuron]` spike tensor and collects dotted state names (`group_0.v`, `group_1.v`).
- **test_mixed_inside_standard_recurrentnn** — `MixedNeuronPopulation` also works inside the plain `RecurrentNN` (no apical input required).

### 8. Resilience to RNN configs
- **test_hetero_rnn_various_configs** — Parametrised over four config combinations:
  1. `unroll=2`, no checkpointing, no offloading
  2. `unroll=2`, `chunk_size=4`, gradient-checkpoint ON
  3. `unroll=1` (fine-grained loop)
  4. `unroll=False` (pure compiled path)

*Why it matters:* The wrapper must not break the chunked/unrolled dispatch logic in `RecurrentNNAbstract`.

### 9. Edge cases
- **test_mixed_dict_vs_list_naming** — Custom dict names (`"exc"`, `"inh"`) are preserved as submodule names.
- **test_mixed_large_batch** — Works with batch size 64.
- **test_mixed_many_small_groups** — Eight groups of one neuron each; slicing must not drift.
- **test_mixed_empty_groups_raises** — Constructor rejects empty groups.
- **test_mixed_non_positive_count_raises** — Constructor rejects non-positive counts.

---

## How to run

```bash
# All mixed-neuron tests (fast — ~4 s)
pytest tests/models/neurons/test_mixed.py \
       tests/models/test_hetero_rnn.py \
       tests/models/test_mixed_comprehensive.py -v

# Lint check
ruff check btorch/models/neurons/mixed.py \
         btorch/models/rnn.py \
         tests/models/neurons/test_mixed.py \
         tests/models/test_hetero_rnn.py \
         tests/models/test_mixed_comprehensive.py
```

---

## Files under test

| File | Role |
|------|------|
| `btorch/models/neurons/mixed.py` | `MixedNeuronPopulation` adapter |
| `btorch/models/rnn.py` | `HeterogeneousRecurrentNN` subclass |
| `btorch/models/neurons/__init__.py` | Public export |

---

*Report generated automatically after test execution.*
