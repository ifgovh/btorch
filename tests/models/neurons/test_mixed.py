import pytest
import torch

from btorch.models import environ
from btorch.models.functional import init_net_state, reset_net_state
from btorch.models.neurons import GLIF3, TwoCompartmentGLIF
from btorch.models.neurons.mixed import MixedNeuronPopulation


DTYPE = torch.float32


def test_mixed_forward_shape():
    """MixedNeuronPopulation returns the correct output shape."""
    torch.manual_seed(0)
    batch_size, n_glif, n_tc = 4, 6, 4

    glif = GLIF3(n_neuron=n_glif, step_mode="s")
    tc = TwoCompartmentGLIF(n_neuron=n_tc, step_mode="s")
    mixed = MixedNeuronPopulation([(n_glif, glif), (n_tc, tc)], step_mode="s")

    init_net_state(mixed, batch_size=batch_size, dtype=DTYPE)

    x = torch.randn(batch_size, n_glif + n_tc, dtype=DTYPE)
    with environ.context(dt=1.0):
        spikes = mixed(x)

    assert spikes.shape == (batch_size, n_glif + n_tc)


def test_mixed_multi_step_shape():
    """MixedNeuronPopulation multi_step_forward returns correct shape."""
    torch.manual_seed(1)
    T, batch_size, n_glif, n_tc = 10, 3, 5, 5

    glif = GLIF3(n_neuron=n_glif, step_mode="s")
    tc = TwoCompartmentGLIF(n_neuron=n_tc, step_mode="s")
    mixed = MixedNeuronPopulation([(n_glif, glif), (n_tc, tc)], step_mode="m")

    init_net_state(mixed, batch_size=batch_size, dtype=DTYPE)

    x = torch.randn(T, batch_size, n_glif + n_tc, dtype=DTYPE)
    with environ.context(dt=1.0):
        spikes = mixed(x)

    assert spikes.shape == (T, batch_size, n_glif + n_tc)


def test_mixed_apical_forward():
    """Apical input is correctly routed to TwoCompartmentGLIF sub-population."""
    torch.manual_seed(2)
    batch_size, n_glif, n_tc = 2, 4, 6

    glif = GLIF3(n_neuron=n_glif, step_mode="s")
    tc = TwoCompartmentGLIF(n_neuron=n_tc, step_mode="s")
    mixed = MixedNeuronPopulation([(n_glif, glif), (n_tc, tc)], step_mode="s")

    init_net_state(mixed, batch_size=batch_size, dtype=DTYPE)

    x = torch.randn(batch_size, n_glif + n_tc, dtype=DTYPE)
    x_apical = torch.randn(batch_size, n_glif + n_tc, dtype=DTYPE)

    with environ.context(dt=1.0):
        # With apical input — should not raise and should produce spikes.
        spikes = mixed(x, x_apical)
    assert spikes.shape == (batch_size, n_glif + n_tc)
    i_a_with = tc.i_a.clone()

    # Verify that the two-compartment population received non-zero apical drive.
    # We can do this by resetting and comparing with zero-apical output.
    reset_net_state(mixed, batch_size=batch_size)
    with environ.context(dt=1.0):
        mixed(x)
    i_a_without = tc.i_a.clone()
    assert not torch.allclose(i_a_with, i_a_without)


def test_mixed_state_init():
    """init_net_state initialises sub-population memories recursively."""
    batch_size = 3
    glif = GLIF3(n_neuron=5, step_mode="s")
    tc = TwoCompartmentGLIF(n_neuron=5, step_mode="s")
    mixed = MixedNeuronPopulation({"glif": (5, glif), "tc": (5, tc)})

    init_net_state(mixed, batch_size=batch_size, dtype=DTYPE)

    # GLIF3 memories
    assert glif.v.shape == (batch_size, 5)
    assert glif.Iasc.shape == (batch_size, 5, glif.n_Iasc)

    # TwoCompartmentGLIF memories
    assert tc.v.shape == (batch_size, 5)
    assert tc.i_a.shape == (batch_size, 5)


def test_mixed_gradient_flow():
    """Gradients flow back through all sub-populations."""
    torch.manual_seed(3)
    batch_size, n_glif, n_tc = 2, 4, 4

    glif = GLIF3(n_neuron=n_glif, step_mode="s", trainable_param={"tau"})
    tc = TwoCompartmentGLIF(
        n_neuron=n_tc, step_mode="s", trainable_param={"tau_s"}
    )
    mixed = MixedNeuronPopulation([(n_glif, glif), (n_tc, tc)], step_mode="s")

    init_net_state(mixed, batch_size=batch_size, dtype=DTYPE)

    x = torch.randn(batch_size, n_glif + n_tc, dtype=DTYPE, requires_grad=True)
    with environ.context(dt=1.0):
        spikes = mixed(x)
    loss = spikes.sum()
    loss.backward()

    assert x.grad is not None
    assert glif.tau.grad is not None
    assert tc.tau_s.grad is not None


def test_mixed_extra_repr():
    """extra_repr contains group names and counts."""
    glif = GLIF3(n_neuron=10)
    tc = TwoCompartmentGLIF(n_neuron=20)
    mixed = MixedNeuronPopulation(
        {"fast": (10, glif), "slow": (20, tc)}
    )
    repr_str = repr(mixed)
    assert "MixedNeuronPopulation" in repr_str
    assert "fast=GLIF3" in repr_str
    assert "slow=TwoCompartmentGLIF" in repr_str


def test_mixed_empty_groups_raises():
    """Constructing with empty groups must raise ValueError."""
    with pytest.raises(ValueError, match="at least one population"):
        MixedNeuronPopulation([])


def test_mixed_non_positive_count_raises():
    """Constructing with a non-positive count must raise ValueError."""
    with pytest.raises(ValueError, match="count must be positive"):
        MixedNeuronPopulation([(0, GLIF3(n_neuron=1))])
