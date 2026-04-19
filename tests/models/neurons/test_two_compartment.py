import torch

from btorch.analysis.two_compartment_fit import (
    AllenSweepBatch,
    exponential_filter_spike_train,
    fit_two_compartment_model,
    mask_post_spike_voltage_samples,
    two_compartment_loss,
)
from btorch.models import environ
from btorch.models.functional import init_net_state, reset_net
from btorch.models.neurons.two_compartment import TwoCompartmentGLIF


def test_two_compartment_single_step_updates_voltage_and_apical_state():
    neuron = TwoCompartmentGLIF(
        n_neuron=2,
        w_Ca=2.0,
        theta_Ca=-0.5,
        w_sa=0.7,
    )
    init_net_state(neuron, batch_size=3, dtype=torch.float32)

    i_soma = torch.full((3, 2), 50.0)
    i_apical = torch.full((3, 2), 1.5)

    with environ.context(dt=1.0):
        spike, voltage, state = neuron.single_step_forward(
            i_soma,
            i_apical,
            return_state=True,
        )

    assert spike.shape == (3, 2)
    assert voltage.shape == (3, 2)
    assert state["i_a"].shape == (3, 2)
    assert torch.all(state["i_a"] > 0.0)
    assert torch.equal(neuron.i_a, state["i_a"])


def test_two_compartment_reset_restores_deterministic_rollout():
    torch.manual_seed(4)
    neuron = TwoCompartmentGLIF(n_neuron=1, step_mode="m")
    i_soma = torch.randn(12, 2, 1)
    i_apical = torch.randn(12, 2, 1)

    init_net_state(neuron, batch_size=2, dtype=i_soma.dtype)
    with environ.context(dt=1.0):
        first_spike, first_v = neuron.multi_step_forward(i_soma, i_apical)
        continued_spike, continued_v = neuron.multi_step_forward(i_soma, i_apical)

        reset_net(neuron, batch_size=2)
        repeated_spike, repeated_v = neuron.multi_step_forward(i_soma, i_apical)

    assert not torch.allclose(first_v, continued_v)
    torch.testing.assert_close(first_spike, repeated_spike, atol=1e-6, rtol=0.0)
    torch.testing.assert_close(first_v, repeated_v, atol=1e-6, rtol=0.0)


def test_two_compartment_loss_masks_post_spike_samples_and_regularizes_w_ca():
    v_true = torch.tensor([0.0, 1.0, 10.0, 0.8, 0.2]).view(5, 1, 1)
    v_pred = torch.tensor([0.0, 1.2, -10.0, 0.6, 0.1]).view(5, 1, 1)
    spike_true = torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0]).view(5, 1, 1)
    spike_pred = spike_true.clone()

    losses = two_compartment_loss(
        v_pred=v_pred,
        spike_pred=spike_pred,
        v_true=v_true,
        spike_true=spike_true,
        dt_ms=1.0,
        w_Ca=torch.tensor([2.0]),
        post_spike_mask_ms=2.0,
        sparsity_weight=0.5,
    )
    mask = mask_post_spike_voltage_samples(spike_true, refractory_bins=2)

    assert mask.squeeze(-1).squeeze(-1).tolist() == [True, False, False, False, True]
    expected_voltage = torch.mean((v_pred[mask] - v_true[mask]) ** 2)
    torch.testing.assert_close(losses["voltage"], expected_voltage)
    torch.testing.assert_close(losses["sparsity"], torch.tensor(2.0))


def test_exponential_filter_spike_train_is_causal_and_stable():
    spikes = torch.tensor([0.0, 1.0, 0.0, 0.0]).view(4, 1, 1)
    filtered = exponential_filter_spike_train(spikes, tau_ms=2.0, dt_ms=1.0)

    assert filtered.shape == spikes.shape
    assert filtered[0].item() == 0.0
    assert filtered[1].item() > 0.0
    assert filtered[2].item() < filtered[1].item()


def test_tbptt_fit_loop_runs_on_synthetic_sweep():
    torch.manual_seed(9)
    model = TwoCompartmentGLIF(
        n_neuron=1,
        trainable_param={"w_Ca", "w_as", "tau_s"},
    )
    sweep = AllenSweepBatch(
        specimen_id=1,
        sweep_number=1,
        dt_ms=1.0,
        i_soma=torch.randn(16, 1, 1),
        v_true=torch.zeros(16, 1, 1),
        spike_true=torch.zeros(16, 1, 1),
        i_apical=torch.zeros(16, 1, 1),
        metadata={"synthetic": True},
    )

    history = fit_two_compartment_model(
        model,
        [sweep],
        lr=1e-3,
        epochs=1,
        chunk_size=5,
    )

    assert len(history) == 4
    assert all("total_loss" in row for row in history)
