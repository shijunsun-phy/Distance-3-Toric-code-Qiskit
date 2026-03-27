"""Small toric-code QEC demo package built from the companion notebook."""

from .core import (
    ToricCode,
    build_toric_pcm,
    binary_rank,
    build_logicals,
    gf2_row_basis,
    build_logical_statevector,
)
from .circuits import (
    build_syndrome_circuit,
    build_syndrome_circuit_clifford,
    build_logical_state_prep_clifford,
    sample_pauli_error,
)
from .simulate import (
    counts_to_syndrome_arrays,
    syndrome_from_error,
    build_matchings,
    run_one_qiskit_trial,
    estimate_logical_failure_qiskit,
)

__all__ = [
    "ToricCode",
    "build_toric_pcm",
    "binary_rank",
    "build_logicals",
    "gf2_row_basis",
    "build_logical_statevector",
    "build_syndrome_circuit",
    "build_syndrome_circuit_clifford",
    "build_logical_state_prep_clifford",
    "sample_pauli_error",
    "counts_to_syndrome_arrays",
    "syndrome_from_error",
    "build_matchings",
    "run_one_qiskit_trial",
    "estimate_logical_failure_qiskit",
]
