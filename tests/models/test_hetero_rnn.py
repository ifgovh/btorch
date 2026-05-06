import torch

from btorch.models import environ
from btorch.models.functional import init_net_state, reset_net_state
from btorch.models.linear import DenseConn
from btorch.models.neurons import GLIF3, TwoCompartmentGLIF
from btorch.models.neurons.mixed import MixedNeuronPopulation
from btorch.models.rnn import HeterogeneousRecurrentNN
from btorch.models.synapse import AlphaPSC


DTYPE = torch.float32


def test_hetero_rnn_forward_shape():
    """HeterogeneousRecurrentNN with mixed neurons produces correct shape."""
    torch.manual_seed(0)
    T, batch_size, n_neuron = 8, 3, 10
    n_glif, n_tc = 6, 4

    glif = GLIF3(n_neuron=n_glif, step_mode="s")
    tc = TwoCompartmentGLIF(n_neuron=n_tc, step_mode="s")
    mixed = MixedNeuronPopulation([(n_glif, glif), (n_tc, tc)], step_mode="s")

    conn = DenseConn(n_neuron, n_neuron, bias=None)
    psc = AlphaPSC(n_neuron=n_neuron, tau_syn=5.0, linear=conn, step_mode="s")

    brain = HeterogeneousRecurrentNN(
        neuron=mixed,
        synapse=psc,
        step_mode="m",
        unroll=4,
    )

    init_net_state(brain, batch_size=batch_size, dtype=DTYPE)

    x = torch.randn(T, batch_size, n_neuron, dtype=DTYPE)
    with environ.context(dt=1.0):
        spikes, states = brain(x)

    assert spikes.shape == (T, batch_size, n_neuron)
    # States from sub-populations should be collected.
    assert any("group_0.v" in k for k in states)
    assert any("group_1.v" in k for k in states)


def test_hetero_rnn_apical_input():
    """Apical input changes the output of two-compartment neurons in the RNN."""
    torch.manual_seed(1)
    T, batch_size, n_neuron = 6, 2, 10
    n_glif, n_tc = 5, 5

    glif = GLIF3(n_neuron=n_glif, step_mode="s")
    tc = TwoCompartmentGLIF(n_neuron=n_tc, step_mode="s")
    mixed = MixedNeuronPopulation([(n_glif, glif), (n_tc, tc)], step_mode="s")

    conn = DenseConn(n_neuron, n_neuron, bias=None)
    psc = AlphaPSC(n_neuron=n_neuron, tau_syn=5.0, linear=conn, step_mode="s")

    brain = HeterogeneousRecurrentNN(
        neuron=mixed,
        synapse=psc,
        step_mode="m",
        unroll=2,
    )

    init_net_state(brain, batch_size=batch_size, dtype=DTYPE)

    x = torch.randn(T, batch_size, n_neuron, dtype=DTYPE)
    x_apical = torch.randn(T, batch_size, n_neuron, dtype=DTYPE)

    # NOTE: x_apical must be passed positionally so that RecurrentNNAbstract
    # slices it along the time dimension.  x_syn is passed as None.
    with environ.context(dt=1.0):
        spikes_with_apical, _ = brain(x, None, x_apical)
    i_a_with = tc.i_a.clone()

    reset_net_state(brain, batch_size=batch_size)

    with environ.context(dt=1.0):
        spikes_without_apical, _ = brain(x)
    i_a_without = tc.i_a.clone()

    # Apical drive should change the internal apical current of the
    # two-compartment neurons even if they do not spike in this random draw.
    assert not torch.allclose(i_a_with, i_a_without)


def test_hetero_rnn_gradient_flow():
    """Gradients flow through HeterogeneousRecurrentNN with mixed neurons."""
    torch.manual_seed(2)
    T, batch_size, n_neuron = 4, 2, 8
    n_glif, n_tc = 4, 4

    glif = GLIF3(n_neuron=n_glif, step_mode="s", trainable_param={"tau"})
    tc = TwoCompartmentGLIF(n_neuron=n_tc, step_mode="s", trainable_param={"tau_s"})
    mixed = MixedNeuronPopulation([(n_glif, glif), (n_tc, tc)], step_mode="s")

    conn = DenseConn(n_neuron, n_neuron, bias=None)
    psc = AlphaPSC(n_neuron=n_neuron, tau_syn=5.0, linear=conn, step_mode="s")

    brain = HeterogeneousRecurrentNN(
        neuron=mixed,
        synapse=psc,
        step_mode="m",
        unroll=2,
    )

    init_net_state(brain, batch_size=batch_size, dtype=DTYPE)

    x = torch.randn(T, batch_size, n_neuron, dtype=DTYPE, requires_grad=True)
    x_apical = torch.randn(T, batch_size, n_neuron, dtype=DTYPE, requires_grad=True)

    with environ.context(dt=1.0):
        spikes, _ = brain(x, None, x_apical)
    loss = spikes.sum()
    loss.backward()

    assert x.grad is not None
    assert x_apical.grad is not None
    assert glif.tau.grad is not None
    assert tc.tau_s.grad is not None
