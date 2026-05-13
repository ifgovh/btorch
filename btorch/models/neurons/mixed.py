"""Heterogeneous neuron population that mixes multiple neuron models.

This module allows a single recurrent layer to contain different neuron
types (e.g. ``GLIF3`` and ``TwoCompartmentGLIF``) by selecting each group's
logical neuron indices, dispatching to sub-populations, and scattering spikes
back into the original network order.
"""

from collections.abc import Mapping, Sequence

import torch
from torch import Tensor, nn

from .two_compartment import TwoCompartmentGLIF


GroupSpec = tuple[int, nn.Module] | tuple[int, nn.Module, Sequence[int] | Tensor]


class MixedNeuronPopulation(nn.Module):
    """Mix multiple neuron populations into a single logical layer.

    The input current tensor is sliced along the last (neuron) dimension and
    dispatched to each sub-population.  Spikes from all groups are
    concatenated back together so the layer behaves like a single neuron
    module from the point of view of :class:`~btorch.models.rnn.RecurrentNN`.

    Args:
        groups: Either a sequence of ``(count, neuron_module)`` tuples, a
            sequence of ``(count, neuron_module, indices)`` tuples, or a
            mapping of ``name -> tuple`` with the same tuple layouts.  When
            ``indices`` is omitted, the group occupies the next contiguous
            slice, preserving the historical concatenation behavior.  When
            ``indices`` is provided, it names the group's positions in the
            full logical neuron order; all groups together must cover
            ``0..n_neuron-1`` exactly once.
        step_mode: ``"s"`` for single-step or ``"m"`` for multi-step
            dispatch.  Default: ``"s"``.

    Raises:
        ValueError: If ``groups`` is empty or contains a non-positive count.

    Examples:
        >>> from btorch.models.neurons import GLIF3, TwoCompartmentGLIF
        >>> glif = GLIF3(n_neuron=60)
        >>> tc = TwoCompartmentGLIF(n_neuron=40)
        >>> mixed = MixedNeuronPopulation([(60, glif), (40, tc)])
    """

    def __init__(
        self,
        groups: Sequence[GroupSpec] | Mapping[str, GroupSpec],
        step_mode: str = "s",
    ):
        super().__init__()
        if not groups:
            raise ValueError("groups must contain at least one population.")

        # Normalise to a list of (name, count, neuron, optional_indices).
        if isinstance(groups, Mapping):
            items = []
            for name, spec in groups.items():
                if len(spec) == 2:
                    count, neuron = spec
                    indices = None
                elif len(spec) == 3:
                    count, neuron, indices = spec
                else:
                    raise ValueError(
                        f"Group {name!r} must be (count, neuron) or "
                        "(count, neuron, indices)."
                    )
                items.append((name, count, neuron, indices))
        else:
            items = []
            for i, spec in enumerate(groups):
                if len(spec) == 2:
                    count, neuron = spec
                    indices = None
                elif len(spec) == 3:
                    count, neuron, indices = spec
                else:
                    raise ValueError(
                        "Each group must be (count, neuron) or "
                        "(count, neuron, indices)."
                    )
                items.append((f"group_{i}", count, neuron, indices))

        counts: list[int] = []
        total = 0
        group_names: list[str] = []
        group_index_names: list[str] = []
        next_contiguous_start = 0
        all_indices: list[Tensor] = []
        for name, count, neuron, indices in items:
            if count <= 0:
                raise ValueError(
                    f"Group {name!r} count must be positive, got {count}."
                )
            self.add_module(name, neuron)
            group_names.append(name)
            counts.append(count)
            group_count = int(neuron.size) if hasattr(neuron, "size") else int(count)
            if group_count != int(count):
                raise ValueError(
                    f"Group {name!r} count={count} does not match neuron size "
                    f"{group_count}."
                )
            if indices is None:
                idx = torch.arange(
                    next_contiguous_start,
                    next_contiguous_start + group_count,
                    dtype=torch.long,
                )
            else:
                idx = torch.as_tensor(indices, dtype=torch.long).flatten()
                if int(idx.numel()) != group_count:
                    raise ValueError(
                        f"Group {name!r} indices length {idx.numel()} does not "
                        f"match count {group_count}."
                    )
            next_contiguous_start += group_count
            total += group_count
            buffer_name = f"_indices_{name}"
            self.register_buffer(buffer_name, idx)
            group_index_names.append(buffer_name)
            all_indices.append(idx)

        self.counts = counts
        self._group_names = group_names
        self._group_index_names = group_index_names
        self._cumsum = [0] + torch.cumsum(
            torch.tensor(counts), dim=0
        ).tolist()
        self.n_neuron = (total,)
        self.size = total
        self.step_mode = step_mode
        if all_indices:
            flat_indices = torch.cat(all_indices)
            if flat_indices.numel() != total:
                raise ValueError("Group indices do not cover the full population.")
            sorted_indices = torch.sort(flat_indices).values
            expected = torch.arange(total, dtype=torch.long)
            if not torch.equal(sorted_indices, expected):
                raise ValueError(
                    "Group indices must cover every logical neuron index exactly "
                    "once from 0 to n_neuron - 1."
                )
        self._uses_indexed_groups = any(
            not torch.equal(
                getattr(self, index_name).cpu(),
                torch.arange(self._cumsum[i], self._cumsum[i + 1], dtype=torch.long),
            )
            for i, index_name in enumerate(self._group_index_names)
        )

    def _indices(self, idx: int) -> Tensor:
        """Return logical neuron indices for group *idx*.

        Args:
            idx: Integer group index in construction order.

        Returns:
            Long tensor stored as a module buffer so it follows device moves.
        """
        return getattr(self, self._group_index_names[idx])

    def _concat_attr(self, attr: str) -> Tensor:
        """Assemble ``attr`` from sub-populations in logical neuron order.

        Args:
            attr: Name of the per-neuron tensor attribute, such as
                ``"v_threshold"`` or ``"v_reset"``.

        Returns:
            Tensor with the same leading dimensions as each group attribute and
            a final dimension ordered by the full logical neuron indices.
        """
        parts: list[tuple[Tensor, Tensor]] = []
        for idx, name in enumerate(self._group_names):
            neuron = getattr(self, name)
            val = getattr(neuron, attr, None)
            if val is None:
                raise AttributeError(
                    f"{neuron.__class__.__name__} has no attribute {attr!r}"
            )
            if isinstance(val, nn.Parameter):
                val = val.data
            parts.append((self._indices(idx).to(device=val.device), val))
        if not self._uses_indexed_groups:
            return torch.cat([val for _, val in parts], dim=-1)
        template = parts[0][1]
        out = template.new_empty(*template.shape[:-1], int(self.size))
        for indices, val in parts:
            out.index_copy_(-1, indices, val)
        return out

    @property
    def v_threshold(self) -> Tensor:
        return self._concat_attr("v_threshold")

    @property
    def v_reset(self) -> Tensor:
        return self._concat_attr("v_reset")

    def _slice(self, x: Tensor, idx: int) -> Tensor:
        """Select ``x`` along the last dimension for group *idx*.

        Args:
            x: Tensor whose final dimension is the full logical neuron order.
            idx: Integer group index in construction order.

        Returns:
            Tensor restricted to the group's neuron indices, preserving the
            within-group order used to construct the sub-population module.
        """
        if not self._uses_indexed_groups:
            start, end = self._cumsum[idx], self._cumsum[idx + 1]
            return x[..., start:end]
        return x.index_select(-1, self._indices(idx).to(device=x.device))

    def _combine_group_outputs(self, group_outputs: list[Tensor]) -> Tensor:
        """Combine per-group tensors into full logical neuron order.

        Args:
            group_outputs: Per-group tensors with identical leading dimensions
                and final dimensions matching each group count.

        Returns:
            Full tensor whose final dimension is ``self.size``.  Contiguous
            groups use concatenation for backward-compatible behavior; indexed
            groups scatter into the original logical order.
        """
        if not self._uses_indexed_groups:
            return torch.cat(group_outputs, dim=-1)
        out = group_outputs[0].new_zeros(*group_outputs[0].shape[:-1], int(self.size))
        for idx, group_value in enumerate(group_outputs):
            indices = self._indices(idx).to(device=group_value.device)
            out.index_copy_(-1, indices, group_value)
        return out

    def single_step_forward(
        self,
        x: Tensor,
        i_apical: Tensor | None = None,
    ) -> Tensor:
        """Advance all sub-populations by one timestep.

        Args:
            x: Input current of shape ``(*batch, n_neuron)``.
            i_apical: Optional apical current of the same shape.  Only slices
                corresponding to :class:`TwoCompartmentGLIF` groups are used.

        Returns:
            Spike tensor of shape ``(*batch, n_neuron)``.
        """
        spikes: list[Tensor] = []
        for idx, name in enumerate(self._group_names):
            neuron = getattr(self, name)
            soma = self._slice(x, idx)
            out: Tensor | tuple[Tensor, ...]

            if isinstance(neuron, TwoCompartmentGLIF):
                apical = (
                    self._slice(i_apical, idx)
                    if i_apical is not None
                    else None
                )
                out = neuron(soma, apical)
            else:
                out = neuron(soma)

            # Normalise return value to spikes only.
            spikes.append(out[0] if isinstance(out, tuple) else out)

        return self._combine_group_outputs(spikes)

    def multi_step_forward(
        self,
        x: Tensor,
        i_apical: Tensor | None = None,
    ) -> Tensor:
        """Advance all sub-populations over a full time sequence.

        Args:
            x: Input sequence of shape ``(T, *batch, n_neuron)``.
            i_apical: Optional apical sequence of the same shape.

        Returns:
            Spike sequence of shape ``(T, *batch, n_neuron)``.
        """
        spike_seq: list[Tensor] = []
        T = x.shape[0]
        for t in range(T):
            step_apical = None if i_apical is None else i_apical[t]
            spike_seq.append(self.single_step_forward(x[t], step_apical))
        return torch.stack(spike_seq, dim=0)

    def forward(
        self,
        x: Tensor,
        i_apical: Tensor | None = None,
    ) -> Tensor:
        """Dispatch to single-step or multi-step forward."""
        if self.step_mode == "m":
            return self.multi_step_forward(x, i_apical)
        return self.single_step_forward(x, i_apical)

    def extra_repr(self) -> str:
        parts = [f"n_neuron={self.n_neuron}"]
        for i, (name, neuron) in enumerate(self.named_children()):
            index_desc = ""
            if self._uses_indexed_groups:
                index_desc = ", indexed=True"
            parts.append(
                f"{name}={neuron.__class__.__name__}(n={self.counts[i]}{index_desc})"
            )
        return ", ".join(parts)
