"""Integration tests for the PyMatching decoding path."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pymatching")
pytest.importorskip("qiskit_aer")

from toric_qec.core import build_logicals, build_toric_pcm
from toric_qec.simulate import build_matchings, run_one_qiskit_trial, syndrome_from_error


def test_single_x_error_decodes_to_no_logical_failure() -> None:
    code, Hx, Hz = build_toric_pcm(3)
    _, LZ = build_logicals(code)
    matching_X, _ = build_matchings(Hx, Hz)

    x_err = np.zeros(code.n, dtype=np.uint8)
    x_err[code.h(0, 0)] = 1

    syn_plaq = syndrome_from_error(Hz, x_err)
    x_corr = matching_X.decode(syn_plaq).astype(np.uint8)
    x_residual = x_err ^ x_corr

    assert np.all((LZ @ x_residual) % 2 == 0)


def test_single_z_error_decodes_to_no_logical_failure() -> None:
    code, Hx, Hz = build_toric_pcm(3)
    LX, _ = build_logicals(code)
    _, matching_Z = build_matchings(Hx, Hz)

    z_err = np.zeros(code.n, dtype=np.uint8)
    z_err[code.v(0, 0)] = 1

    syn_star = syndrome_from_error(Hx, z_err)
    z_corr = matching_Z.decode(syn_star).astype(np.uint8)
    z_residual = z_err ^ z_corr

    assert np.all((LX @ z_residual) % 2 == 0)


def test_noiseless_qiskit_trial_has_no_logical_failure() -> None:
    code, Hx, Hz = build_toric_pcm(3)
    LX, LZ = build_logicals(code)
    matching_X, matching_Z = build_matchings(Hx, Hz)

    result = run_one_qiskit_trial(
        code=code,
        Hx=Hx,
        Hz=Hz,
        LX=LX,
        LZ=LZ,
        matching_X=matching_X,
        matching_Z=matching_Z,
        p=0.0,
        seed=123,
        use_clifford_prep=True,
        x_logical=np.array([0, 0], dtype=np.uint8),
    )

    assert result["fail"] is False
    assert np.all(result["x_err"] == 0)
    assert np.all(result["z_err"] == 0)
    assert np.all(result["syn_plaq"] == 0)
    assert np.all(result["syn_star"] == 0)
