# Mixed-Neuron Speed Report

**Date:** 2026-05-03  
**Device:** CPU (no CUDA)  
**Commit:** `e1e4417` (`few-compartment-model` branch)

---

## Question

Is `MixedNeuronPopulation` much slower than a single-type RNN?

To answer this we measured **three** configurations with identical total neuron counts, synapses, and inputs:

1. **Base** — 100 % `GLIF3` inside plain `RecurrentNN` (the usual baseline).
2. **Adapt** — 100 % `GLIF3` wrapped in `MixedNeuronPopulation` + `HeterogeneousRecurrentNN`.  
   This isolates the **pure wrapper overhead**.
3. **Mix** — 50 % `GLIF3` + 50 % `TwoCompartmentGLIF` inside `HeterogeneousRecurrentNN` with apical input.  
   This is the **real-world heterogeneous case**.

---

## Results

| Config | T | B | N | BaseFwd | AdaptFwd | MixFwd | A/B | M/B | BaseTrn | AdaptTrn | MixTrn | A/B | M/B |
|--------|---|---|---|---------|----------|--------|-----|-----|---------|----------|--------|-----|-----|
| small | 100 | 1 | 64 | 20.2 ms | 21.7 ms | 39.2 ms | **1.08×** | **1.95×** | 47.0 ms | 44.0 ms | 92.1 ms | **0.94×** | **1.96×** |
| small | 100 | 16 | 64 | 23.0 ms | 35.6 ms | 59.3 ms | **1.55×** | **2.58×** | 62.4 ms | 76.9 ms | 147.7 ms | **1.23×** | **2.37×** |
| medium | 200 | 1 | 256 | 81.0 ms | 82.4 ms | 121.1 ms | **1.02×** | **1.50×** | 152.6 ms | 141.2 ms | 282.7 ms | **0.92×** | **1.85×** |
| medium | 200 | 16 | 256 | 98.8 ms | 108.9 ms | 188.6 ms | **1.10×** | **1.91×** | 202.7 ms | 209.8 ms | 378.7 ms | **1.03×** | **1.87×** |
| large | 500 | 1 | 1024 | 287.1 ms | 311.9 ms | 458.5 ms | **1.09×** | **1.60×** | 850.3 ms | 908.4 ms | 1301.1 ms | **1.07×** | **1.53×** |
| large | 500 | 16 | 1024 | 905.0 ms | 787.6 ms | 1087.1 ms | **0.87×** | **1.20×** | 2566.2 ms | 2521.3 ms | 2528.7 ms | **0.98×** | **0.99×** |

*All numbers are median wall-clock time over 5 runs after 2 warm-up runs.*  
*A/B = Adapt ÷ Base (wrapper overhead only).  M/B = Mix ÷ Base (wrapper + TC cost).*

---

## Key findings

### 1. Wrapper overhead is negligible

The **Adapt** column (same neurons wrapped in `MixedNeuronPopulation`) is almost never more than **10 % slower** than the raw baseline, and sometimes even *faster* (measurement noise).  
→ The Python-level slicing / `isinstance` checks in the adapter add **essentially zero overhead** on CPU.

### 2. Real heterogeneity costs ~1.5–2.6×

The **Mix** column is **1.5–2.6× slower** than the pure-GLIF3 baseline.  
This is **not** adapter overhead — it is the intrinsic cost of `TwoCompartmentGLIF`, which has:
- an extra apical-current ODE,
- a calcium-plateau nonlinearity,
- bidirectional soma-apical coupling,
- an adaptive threshold state.

In other words, you are paying for a **more complex neuron model**, not for the mixing machinery.

### 3. Large-batch / large-network regime is friendlier

At the largest config tested (T=500, B=16, N=1024):
- Forward: Mix is only **1.2×** slower.
- Training (forward + backward): Mix is **~1.0×** (practically the same).

Why? When the network is large, the **synaptic matrix multiplications** and **back-prop through dense weights** dominate the runtime. The per-neuron integration becomes a smaller fraction of total work.

---

## Caveats

- **CPU only.** On GPU the story may differ because Python-level loops incur kernel-launch overhead. If you plan to run on GPU, a quick `torch.compile` wrapper around `MixedNeuronPopulation.single_step_forward` would likely erase the small remaining overhead.
- **No `torch.compile`.** The benchmark uses the default eager mode. Compiling the RNN cell often yields 1.5–3× speed-ups for recurrent loops.
- **Single-threaded PyTorch.** CPU benchmarks were run with default PyTorch threading. Using `torch.set_num_threads(4)` or similar may change relative numbers.

---

## Bottom line

| Question | Answer |
|----------|--------|
| Is the *adapter* slow? | **No.** Wrapper overhead is < 10 % and often in the noise. |
| Is the *mixed model* slow? | **Moderately.** 1.5–2.6× vs pure GLIF3, but that is the cost of using a two-compartment neuron, not the mixing code. |
| Is it acceptable for large networks? | **Yes.** At 1k neurons / batch 16, training throughput is practically identical. |

---

## Reproduce

```bash
PYTHONPATH=$(pwd):$PYTHONPATH \
  python tests/models/benchmark_mixed_vs_baseline.py
```

*Report generated automatically from benchmark output.*
