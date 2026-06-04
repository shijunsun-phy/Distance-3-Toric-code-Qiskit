"""Unit tests for circuit-construction helper functions."""

from __future__ import annotations

import numpy as np
import pytest

from toric_qec.circuits import (
    binary_symplectic_row_to_pauli_label,
    build_logical_state_prep_clifford,
    build_syndrome_circuit_clifford,
    sample_pauli_error,
)
from toric_qec.core import build_logicals, build_toric_pcm


def test_sample_pauli_error_zero_probability_returns_no_error() -> None:
    rng = np.random.default_rng(123)

    x_err, z_err = sample_pauli_error(n=18, p=0.0, rng=rng)

    assert x_err.dtype == np.uint8
    assert z_err.dtype == np.uint8
    assert np.all(x_err == 0)
    assert np.all(z_err == 0)


def test_sample_pauli_error_unit_probability_errors_every_qubit() -> None:
    rng = np.random.default_rng(123)

    x_err, z_err = sample_pauli_error(n=18, p=1.0, rng=rng)

    # In the depolarizing sampler, every qubit should receive X, Y, or Z when p=1.
    assert np.all((x_err | z_err) == 1)


def test_binary_symplectic_row_to_pauli_label_uses_qiskit_little_endian_label_order() -> None:
    x_row = np.array([1, 0, 1, 0], dtype=np.uint8)
    z_row = np.array([0, 1, 1, 0], dtype=np.uint8)

    label = binary_symplectic_row_to_pauli_label(x_row, z_row)

    # Per-qubit labels before Qiskit's string reversal are X, Z, Y, I.
    assert label == "+IYZX"


def test_binary_symplectic_row_to_pauli_label_rejects_mismatched_shapes() -> None:
    x_row = np.array([1, 0, 1], dtype=np.uint8)
    z_row = np.array([0, 1], dtype=np.uint8)

    with pytest.raises(ValueError, match="same shape"):
        binary_symplectic_row_to_pauli_label(x_row, z_row)


def test_logical_state_prep_rejects_wrong_logical_shape() -> None:
    code, Hx, Hz = build_toric_pcm(3)
    LX, LZ = build_logicals(code)

    with pytest.raises(ValueError, match="length-2"):
        build_logical_state_prep_clifford(
            Hx=Hx,
            Hz=Hz,
            LX=LX,
            LZ=LZ,
            x_logical=np.array([0, 0, 0], dtype=np.uint8),
        )


def test_clifford_syndrome_circuit_has_expected_register_sizes() -> None:
    code, Hx, Hz = build_toric_pcm(3)
    LX, LZ = build_logicals(code)

    x_err = np.zeros(code.n, dtype=np.uint8)
    z_err = np.zeros(code.n, dtype=np.uint8)
    x_logical = np.array([0, 0], dtype=np.uint8)

    qc = build_syndrome_circuit_clifford(
        code=code,
        Hx=Hx,
        Hz=Hz,
        LX=LX,
        LZ=LZ,
        x_logical=x_logical,
        x_err=x_err,
        z_err=z_err,
    )

    assert qc.num_qubits == code.n + code.n_plaquette + code.n_star
    assert qc.num_clbits == code.n_plaquette + code.n_star
