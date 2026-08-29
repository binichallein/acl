"""Abstract interface between surface tools and canonical semantics."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping

from toolshift.types import ExecutionTrace, JSONValue, SchemaVariant, SemanticAction


class SemanticAdapter(ABC):
    """Translate calls and observations for one immutable schema variant."""

    __slots__ = ("_variant",)

    def __init__(self, variant: SchemaVariant) -> None:
        if not isinstance(variant, SchemaVariant):
            raise ValueError("variant must be a SchemaVariant")
        self._variant = variant

    @property
    def variant(self) -> SchemaVariant:
        """The surface schema handled by this adapter."""

        return self._variant

    @abstractmethod
    def surface_to_semantic(
        self,
        surface_call: Mapping[str, JSONValue],
    ) -> tuple[SemanticAction, ...]:
        """Parse one surface call into zero or more semantic actions.

        Implementations parse only; they must not execute tools. Unknown tools,
        malformed calls, and invalid arguments must raise ``ValueError``.
        """

        raise NotImplementedError

    @abstractmethod
    def semantic_to_base_calls(
        self,
        action: SemanticAction,
    ) -> tuple[Mapping[str, JSONValue], ...]:
        """Map one semantic action to one or more executable base calls.

        Implementations must never silently return an empty tuple. Unknown or
        invalid actions must raise ``ValueError``.
        """

        raise NotImplementedError

    @abstractmethod
    def base_observation_to_surface(
        self,
        surface_call: Mapping[str, JSONValue],
        actions: tuple[SemanticAction, ...],
        base_observation_groups: tuple[tuple[JSONValue, ...], ...],
    ) -> JSONValue:
        """Wrap grouped base observations for one original surface call.

        ``actions`` and ``base_observation_groups`` must have equal length. Each
        action has a non-empty group containing the observations from its one or
        more base calls. Zero actions is valid only with zero groups, for a
        surface protocol step that performs no base execution. Unknown actions,
        malformed groups, and invalid values must raise ``ValueError``.
        """

        raise NotImplementedError

    @abstractmethod
    def canonicalize_trace(self, trace: ExecutionTrace) -> tuple[SemanticAction, ...]:
        """Return the ordered canonical semantic actions represented by a trace.

        Malformed traces and unknown actions must raise ``ValueError``.
        """

        raise NotImplementedError
