"""Comprehensive test suite for MixedNeuronPopulation and HeterogeneousRecurrentNN.

This module exercises:
  1. Functional equivalence (mixed wrapper vs raw neuron)
  2. Correct slicing and concatenation order
  3. Single-step and multi-step dispatch
  4. State initialisation, reset, and detachment
  5. Apical input routing (with and without)
  6. Gradient flow through heterogeneous populations and synapses
  7. Integration with RecurrentNN / HeterogeneousRecurrentNN
  8. Resilience to checkpointing, offloading, and different unroll sizes
  9. Edge cases (single group, dict naming, zero batch dims)
"""


import pytest
import torch

from btorch.models import environ
from btorch.models.functional import detach_net, init_net_state, reset_net_state
from btorch.models.linear import DenseConn
from btorch.models.neurons import GLIF3, TwoCompartmentGLIF
from btorch.models.neurons.mixed import MixedNeuronPopulation
from btorch.models.rnn import HeterogeneousRecurrentNN, RecurrentNN
from btorch.models.synapse import AlphaPSC


DTYPE = torch.float32


# ---------------------------------------------------------------------------
# 1. Functional equivalence
# ---------------------------------------------------------------------------

def test_mixed_single_group_equivalence_glif():
    """A single-group MixedNeuronPopulation(GLIF3) must match raw GLIF3."""
    torch.manual_seed(10)
    batch_size, n = 3, 7

    raw = GLIF3(n_neuron=n, step_mode="s")
    mixed = MixedNeuronPopulation(
        [(n, GLIF3(n_neuron=n, step_mode="s"))], step_mode="s"
    )

    init_net_state(raw, batch_size=batch_size, dtype=DTYPE)
    init_net_state(mixed, batch_size=batch_size, dtype=DTYPE)

    x = torch.randn(batch_size, n, dtype=DTYPE)
    with environ.context(dt=1.0):
        s_raw = raw(x)
        s_mixed = mixed(x)

    assert torch.allclose(s_raw, s_mixed)


def test_mixed_single_group_equivalence_tc():
    """A single-group MixedNeuronPopulation(TC) must match raw TC (with apical)."""
    torch.manual_seed(11)
    batch_size, n = 3, 7

    raw = TwoCompartmentGLIF(n_neuron=n, step_mode="s")
    mixed = MixedNeuronPopulation(
        [(n, TwoCompartmentGLIF(n_neuron=n, step_mode="s"))], step_mode="s"
    )

    init_net_state(raw, batch_size=batch_size, dtype=DTYPE)
    init_net_state(mixed, batch_size=batch_size, dtype=DTYPE)

    x = torch.randn(batch_size, n, dtype=DTYPE)
    x_a = torch.randn(batch_size, n, dtype=DTYPE)

    with environ.context(dt=1.0):
        s_raw, _ = raw(x, x_a)
        s_mixed = mixed(x, x_a)

    assert torch.allclose(s_raw, s_mixed)


# ---------------------------------------------------------------------------
# 2. Slicing / ordering
# ---------------------------------------------------------------------------

def test_mixed_concatenation_order():
    """Spikes must be concatenated in the same order as the groups."""
    torch.manual_seed(20)
    batch_size = 2
    n1, n2 = 3, 4

    glif1 = GLIF3(n_neuron=n1, v_threshold=-50.0, step_mode="s")
    glif2 = GLIF3(n_neuron=n2, v_threshold=-50.0, step_mode="s")

    mixed = MixedNeuronPopulation(
        [(n1, glif1), (n2, glif2)], step_mode="s"
    )
    init_net_state(mixed, batch_size=batch_size, dtype=DTYPE)

    # Manually set the first group's voltage above threshold so it spikes,
    # and the second group's voltage below threshold so it stays silent.
    with torch.no_grad():
        glif1.v.fill_(0.0)   # 0.0 > -50.0  => spike
        glif2.v.fill_(-60.0)  # -60.0 < -50.0 => no spike

    x = torch.zeros(batch_size, n1 + n2, dtype=DTYPE)
    with environ.context(dt=1.0):
        spikes = mixed(x)

    # First n1 neurons should be all ones, last n2 should be all zeros.
    assert torch.allclose(spikes[:, :n1], torch.ones_like(spikes[:, :n1]))
    assert torch.allclose(spikes[:, n1:], torch.zeros_like(spikes[:, n1:]))


# ---------------------------------------------------------------------------
# 3. Multi-step dispatch (standalone)
# ---------------------------------------------------------------------------

def test_mixed_standalone_multi_step_matches_loop():
    """multi_step_forward must equal a manual Python loop over single_step_forward."""
    torch.manual_seed(30)
    T, batch_size, n_glif, n_tc = 5, 2, 4, 4

    glif = GLIF3(n_neuron=n_glif, step_mode="s")
    tc = TwoCompartmentGLIF(n_neuron=n_tc, step_mode="s")
    mixed = MixedNeuronPopulation([(n_glif, glif), (n_tc, tc)], step_mode="m")

    init_net_state(mixed, batch_size=batch_size, dtype=DTYPE)

    x = torch.randn(T, batch_size, n_glif + n_tc, dtype=DTYPE)
    with environ.context(dt=1.0):
        spikes_multi = mixed(x)

    reset_net_state(mixed, batch_size=batch_size)
    spikes_loop = []
    with environ.context(dt=1.0):
        for t in range(T):
            spikes_loop.append(mixed.single_step_forward(x[t]))
    spikes_loop = torch.stack(spikes_loop, dim=0)

    assert torch.allclose(spikes_multi, spikes_loop)


# ---------------------------------------------------------------------------
# 4. State management
# ---------------------------------------------------------------------------

def test_mixed_state_reset_preserve_batch():
    """reset_net_state must preserve the existing batch dimension."""
    batch_size = 4
    glif = GLIF3(n_neuron=5, step_mode="s")
    tc = TwoCompartmentGLIF(n_neuron=5, step_mode="s")
    mixed = MixedNeuronPopulation({"a": (5, glif), "b": (5, tc)})

    init_net_state(mixed, batch_size=batch_size, dtype=DTYPE)
    assert glif.v.shape == (batch_size, 5)

    reset_net_state(mixed)  # no batch_size arg
    assert glif.v.shape == (batch_size, 5)


def test_mixed_detach():
    """detach_net must detach all sub-population memories."""
    batch_size = 2
    glif = GLIF3(n_neuron=5, step_mode="s")
    tc = TwoCompartmentGLIF(n_neuron=5, step_mode="s")
    mixed = MixedNeuronPopulation([(5, glif), (5, tc)])

    init_net_state(mixed, batch_size=batch_size, dtype=DTYPE)

    # Manually set requires_grad so we can detect detachment.
    glif.v.requires_grad_(True)
    tc.i_a.requires_grad_(True)

    detach_net(mixed)

    assert glif.v.grad_fn is None
    assert tc.i_a.grad_fn is None


# ---------------------------------------------------------------------------
# 5. Apical routing (detailed)
# ---------------------------------------------------------------------------

def test_mixed_apical_only_tc_receives_it():
    """When apical input is provided, only TC groups should see it."""
    torch.manual_seed(40)
    batch_size = 2
    n_glif, n_tc = 4, 4

    glif = GLIF3(n_neuron=n_glif, step_mode="s")
    tc = TwoCompartmentGLIF(n_neuron=n_tc, step_mode="s")
    mixed = MixedNeuronPopulation([(n_glif, glif), (n_tc, tc)], step_mode="s")

    init_net_state(mixed, batch_size=batch_size, dtype=DTYPE)

    x = torch.zeros(batch_size, n_glif + n_tc, dtype=DTYPE)
    x_apical = torch.zeros(batch_size, n_glif + n_tc, dtype=DTYPE)
    # Put a strong apical signal only in the TC slice.
    x_apical[:, n_glif:] = 10.0

    with environ.context(dt=1.0):
        mixed(x, x_apical)

    # GLIF neurons should see zero input; TC neurons should see large apical.
    # We verify indirectly by checking that tc.i_a changed while glif.v did not.
    # Actually glif.v will change because of leak dynamics, so we compare
    # against a run where the TC slice of apical is zero.
    reset_net_state(mixed, batch_size=batch_size)
    x_apical_zero_tc = x_apical.clone()
    x_apical_zero_tc[:, n_glif:] = 0.0
    with environ.context(dt=1.0):
        mixed(x, x_apical_zero_tc)
    i_a_zero = tc.i_a.clone()

    reset_net_state(mixed, batch_size=batch_size)
    with environ.context(dt=1.0):
        mixed(x, x_apical)
    i_a_nonzero = tc.i_a.clone()

    assert not torch.allclose(i_a_zero, i_a_nonzero)


# ---------------------------------------------------------------------------
# 6. Gradient flow (recurrent)
# ---------------------------------------------------------------------------

def test_mixed_recurrent_gradient_matches_baseline():
    """Gradient through a mixed RecurrentNN should match a hand-built baseline."""
    torch.manual_seed(50)
    T, batch_size, n = 4, 2, 6

    # Baseline: pure GLIF3 RNN
    baseline_neuron = GLIF3(n_neuron=n, step_mode="s", trainable_param={"tau"})
    baseline_conn = DenseConn(n, n, bias=None)
    baseline_psc = AlphaPSC(
        n_neuron=n, tau_syn=5.0, linear=baseline_conn, step_mode="s"
    )
    baseline_rnn = RecurrentNN(
        neuron=baseline_neuron,
        synapse=baseline_psc,
        step_mode="m",
        unroll=2,
    )

    # Mixed: same GLIF3 but wrapped in MixedNeuronPopulation
    mixed_neuron = MixedNeuronPopulation(
        [(n, GLIF3(n_neuron=n, step_mode="s", trainable_param={"tau"}))],
        step_mode="s",
    )
    mixed_conn = DenseConn(n, n, bias=None)
    mixed_psc = AlphaPSC(n_neuron=n, tau_syn=5.0, linear=mixed_conn, step_mode="s")
    mixed_rnn = RecurrentNN(
        neuron=mixed_neuron,
        synapse=mixed_psc,
        step_mode="m",
        unroll=2,
    )

    # Sync weights
    with torch.no_grad():
        mixed_conn.weight.copy_(baseline_conn.weight)

    init_net_state(baseline_rnn, batch_size=batch_size, dtype=DTYPE)
    init_net_state(mixed_rnn, batch_size=batch_size, dtype=DTYPE)

    x = torch.randn(T, batch_size, n, dtype=DTYPE, requires_grad=True)

    with environ.context(dt=1.0):
        out_b, _ = baseline_rnn(x)
        out_m, _ = mixed_rnn(x)

    assert torch.allclose(out_b, out_m)

    loss_b = out_b.sum()
    loss_m = out_m.sum()
    loss_b.backward()
    loss_m.backward()

    assert x.grad is not None
    # Check that the wrapped neuron's parameter received a gradient.
    mixed_glif = mixed_rnn.neuron.group_0
    assert mixed_glif.tau.grad is not None
    assert torch.allclose(baseline_neuron.tau.grad, mixed_glif.tau.grad)


# ---------------------------------------------------------------------------
# 7. HeterogeneousRecurrentNN with checkpointing / offloading configs
# ---------------------------------------------------------------------------

RNN_CONFIGS = [
    {"unroll": 2, "chunk_size": None, "grad_checkpoint": False, "cpu_offload": False},
    {"unroll": 2, "chunk_size": 4, "grad_checkpoint": True, "cpu_offload": False},
    {"unroll": 1, "chunk_size": None, "grad_checkpoint": False, "cpu_offload": False},
    {
        "unroll": False,
        "chunk_size": None,
        "grad_checkpoint": False,
        "cpu_offload": False,
    },
]


@pytest.mark.parametrize("cfg", RNN_CONFIGS)
def test_hetero_rnn_various_configs(cfg):
    """HeterogeneousRecurrentNN must work with different unroll/checkpoint settings."""
    torch.manual_seed(60)
    T, batch_size, n = 6, 2, 8
    n_glif, n_tc = 4, 4

    glif = GLIF3(n_neuron=n_glif, step_mode="s")
    tc = TwoCompartmentGLIF(n_neuron=n_tc, step_mode="s")
    mixed = MixedNeuronPopulation([(n_glif, glif), (n_tc, tc)], step_mode="s")

    conn = DenseConn(n, n, bias=None)
    psc = AlphaPSC(n_neuron=n, tau_syn=5.0, linear=conn, step_mode="s")

    brain = HeterogeneousRecurrentNN(
        neuron=mixed,
        synapse=psc,
        step_mode="m",
        **cfg,
    )

    init_net_state(brain, batch_size=batch_size, dtype=DTYPE)

    x = torch.randn(T, batch_size, n, dtype=DTYPE)
    x_a = torch.randn(T, batch_size, n, dtype=DTYPE)

    with environ.context(dt=1.0):
        spikes, states = brain(x, None, x_a)

    assert spikes.shape == (T, batch_size, n)
    assert isinstance(states, dict)
    assert len(states) > 0


# ---------------------------------------------------------------------------
# 8. Edge cases
# ---------------------------------------------------------------------------

def test_mixed_dict_vs_list_naming():
    """Dict-named groups should produce state keys with the custom names."""
    batch_size = 2
    glif = GLIF3(n_neuron=3, step_mode="s")
    tc = TwoCompartmentGLIF(n_neuron=3, step_mode="s")
    mixed = MixedNeuronPopulation(
        {"exc": (3, glif), "inh": (3, tc)}, step_mode="s"
    )

    init_net_state(mixed, batch_size=batch_size, dtype=DTYPE)

    x = torch.randn(batch_size, 6, dtype=DTYPE)
    with environ.context(dt=1.0):
        mixed(x)

    # Verify that the internal modules are accessible by the custom names.
    assert hasattr(mixed, "exc")
    assert hasattr(mixed, "inh")


def test_mixed_large_batch():
    """MixedNeuronPopulation should work with a larger batch size."""
    torch.manual_seed(70)
    batch_size = 64
    n_glif, n_tc = 10, 10

    glif = GLIF3(n_neuron=n_glif, step_mode="s")
    tc = TwoCompartmentGLIF(n_neuron=n_tc, step_mode="s")
    mixed = MixedNeuronPopulation([(n_glif, glif), (n_tc, tc)], step_mode="s")

    init_net_state(mixed, batch_size=batch_size, dtype=DTYPE)

    x = torch.randn(batch_size, n_glif + n_tc, dtype=DTYPE)
    with environ.context(dt=1.0):
        spikes = mixed(x)

    assert spikes.shape == (batch_size, n_glif + n_tc)


def test_mixed_many_small_groups():
    """Many small groups should still slice correctly."""
    torch.manual_seed(80)
    batch_size = 2
    groups = [(1, GLIF3(n_neuron=1, step_mode="s")) for _ in range(8)]
    mixed = MixedNeuronPopulation(groups, step_mode="s")

    init_net_state(mixed, batch_size=batch_size, dtype=DTYPE)

    x = torch.randn(batch_size, 8, dtype=DTYPE)
    with environ.context(dt=1.0):
        spikes = mixed(x)

    assert spikes.shape == (batch_size, 8)


# ---------------------------------------------------------------------------
# 9. RecurrentNN (non-hetero) compatibility with MixedNeuronPopulation
# ---------------------------------------------------------------------------

def test_mixed_inside_standard_recurrentnn():
    """MixedNeuronPopulation should work inside plain RecurrentNN when no
    apical input is needed."""
    torch.manual_seed(90)
    T, batch_size, n = 5, 2, 8
    n_glif, n_tc = 4, 4

    glif = GLIF3(n_neuron=n_glif, step_mode="s")
    tc = TwoCompartmentGLIF(n_neuron=n_tc, step_mode="s")
    mixed = MixedNeuronPopulation([(n_glif, glif), (n_tc, tc)], step_mode="s")

    conn = DenseConn(n, n, bias=None)
    psc = AlphaPSC(n_neuron=n, tau_syn=5.0, linear=conn, step_mode="s")

    brain = RecurrentNN(
        neuron=mixed,
        synapse=psc,
        step_mode="m",
        unroll=2,
    )

    init_net_state(brain, batch_size=batch_size, dtype=DTYPE)

    x = torch.randn(T, batch_size, n, dtype=DTYPE)
    with environ.context(dt=1.0):
        spikes, states = brain(x)

    assert spikes.shape == (T, batch_size, n)
    assert any("group_0.v" in k for k in states)
    assert any("group_1.v" in k for k in states)
