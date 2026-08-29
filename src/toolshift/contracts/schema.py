"""Public compatibility facade for behavioral-equivalence contracts.

Implementation modules use a directed internal dependency graph and never import
this facade. The aliases below preserve the original public import locations.
"""

from toolshift.contracts._common import (
    ContractDiagnostic,
    LayerContractResult,
)
from toolshift.contracts._schema import (
    SchemaProbe,
    check_schema_contract,
    schema_fingerprint,
)
from toolshift.contracts.suite import (
    ContractSuiteResult,
    contract_suite_fingerprint,
    evaluate_contract_suite,
    require_dataset_admission,
)

__all__ = [
    "ContractDiagnostic",
    "ContractSuiteResult",
    "LayerContractResult",
    "SchemaProbe",
    "check_schema_contract",
    "contract_suite_fingerprint",
    "evaluate_contract_suite",
    "require_dataset_admission",
    "schema_fingerprint",
]
