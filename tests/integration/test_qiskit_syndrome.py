"""Integration tests comparing Qiskit-measured syndromes with classical syndromes."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("qiskit_aer")

from qiskit import transpile
from qiskit_aer import AerSimulator

from toric_qec.circuits import build_syndrome_circuit_clifford
from toric_qec.core import build_logicals, build_toric_pcm
from toric_qec.simulate import counts_to_syndrome_arrays, syndrome_from_error


def _measure_syndrome_with_qiskit(
    *,
    x_err: np.ndarray,
    z_err: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    code, Hx, Hz = build_toric_pcm(3)
    LX, LZ = build_logicals(code)

    qc = build_syndrome_circuit_clifford(
        code=code,
        Hx=Hx,
        Hz=Hz,
        LX=LX,
        LZ=LZ,
        x_logical=np.array([0, 0], dtype=np.uint8),
        x_err=x_err,
        z_err=z_err,
    )

    simulator = AerSimulator(method="stabilizer")
    transpiled = transpile(qc, simulator)
    result = simulator.run(transpiled, shots=1).result()
    counts = result.get_counts()

    return counts_to_syndrome_arrays(
        counts=counts,
        n_plaq=code.n_plaquette,
        n_star=code.n_star,
    )


def test_qiskit_syndrome_matches_classical_syndrome_for_zero_error() -> None:
    code, Hx, Hz = build_toric_pcm(3)

    x_err = np.zeros(code.n, dtype=np.uint8)
    z_err = np.zeros(code.n, dtype=np.uint8)

    syn_plaq_meas, syn_star_meas = _measure_syndrome_with_qiskit(
        x_err=x_err,
        z_err=z_err,
    )

    assert np.array_equal(syn_plaq_meas, syndrome_from_error(Hz, x_err))
    assert np.array_equal(syn_star_meas, syndrome_from_error(Hx, z_err))


def test_qiskit_syndrome_matches_classical_syndrome_for_known_x_error() -> None:
    code, Hx, Hz = build_toric_pcm(3)

    x_err = np.zeros(code.n, dtype=np.uint8)
    z_err = np.zeros(code.n, dtype=np.uint8)
    x_err[code.h(0, 0)] = 1

    syn_plaq_meas, syn_star_meas = _measure_syndrome_with_qiskit(
        x_err=x_err,
        z_err=z_err,
    )

    assert np.array_equal(syn_plaq_meas, syndrome_from_error(Hz, x_err))
    assert np.array_equal(syn_star_meas, syndrome_from_error(Hx, z_err))


def test_qiskit_syndrome_matches_classical_syndrome_for_known_y_error() -> None:
    code, Hx, Hz = build_toric_pcm(3)

    x_err = np.zeros(code.n, dtype=np.uint8)
    z_err = np.zeros(code.n, dtype=np.uint8)

    # A Y error has both X and Z binary components.
    q = code.h(0, 0)
    x_err[q] = 1
    z_err[q] = 1

    syn_plaq_meas, syn_star_meas = _measure_syndrome_with_qiskit(
        x_err=x_err,
        z_err=z_err,
    )

    assert np.array_equal(syn_plaq_meas, syndrome_from_error(Hz, x_err))
    assert np.array_equal(syn_star_meas, syndrome_from_error(Hx, z_err))
