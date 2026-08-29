"""Behavioral-equivalence contracts for schema variants."""

from toolshift.contracts.denotation import DenotationCase, check_denotation_contract
from toolshift.contracts.schema import (
    ContractDiagnostic,
    ContractSuiteResult,
    LayerContractResult,
    SchemaProbe,
    check_schema_contract,
    contract_suite_fingerprint,
    evaluate_contract_suite,
    require_dataset_admission,
    schema_fingerprint,
)
from toolshift.contracts.state import (
    StateCase,
    StateEvidence,
    StateEvidenceProvider,
    check_state_contract,
)
from toolshift.contracts.trace import (
    PhysicalCallEffect,
    TraceCase,
    TraceEvidence,
    TraceEvidenceProvider,
    base_call_fingerprint,
    check_trace_contract,
)

__all__ = [
    "ContractDiagnostic",
    "ContractSuiteResult",
    "DenotationCase",
    "LayerContractResult",
    "PhysicalCallEffect",
    "SchemaProbe",
    "StateCase",
    "StateEvidence",
    "StateEvidenceProvider",
    "TraceCase",
    "TraceEvidence",
    "TraceEvidenceProvider",
    "base_call_fingerprint",
    "check_denotation_contract",
    "check_schema_contract",
    "check_state_contract",
    "check_trace_contract",
    "contract_suite_fingerprint",
    "evaluate_contract_suite",
    "require_dataset_admission",
    "schema_fingerprint",
]
