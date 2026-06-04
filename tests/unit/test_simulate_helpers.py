"""Unit tests for syndrome and counts-parsing helpers."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("qiskit_aer")
pytest.importorskip("pymatching")

from toric_qec.core import build_toric_pcm
from toric_qec.simulate import counts_to_syndrome_arrays, syndrome_from_error


def test_zero_error_has_zero_syndrome() -> None:
    code, Hx, Hz = build_toric_pcm(3)

    x_err = np.zeros(code.n, dtype=np.uint8)
    z_err = np.zeros(code.n, dtype=np.uint8)

    assert np.all(syndrome_from_error(Hz, x_err) == 0)
    assert np.all(syndrome_from_error(Hx, z_err) == 0)


def test_single_x_error_flips_two_plaquette_checks() -> None:
    code, _, Hz = build_toric_pcm(3)

    x_err = np.zeros(code.n, dtype=np.uint8)
    x_err[code.h(0, 0)] = 1

    syndrome = syndrome_from_error(Hz, x_err)

    assert syndrome.dtype == np.uint8
    assert int(syndrome.sum()) == 2


def test_single_z_error_flips_two_star_checks() -> None:
    code, Hx, _ = build_toric_pcm(3)

    z_err = np.zeros(code.n, dtype=np.uint8)
    z_err[code.h(0, 0)] = 1

    syndrome = syndrome_from_error(Hx, z_err)

    assert syndrome.dtype == np.uint8
    assert int(syndrome.sum()) == 2


def test_counts_to_syndrome_arrays_parses_qiskit_register_order() -> None:
    counts = {"101 011": 1}

    syn_plaq, syn_star = counts_to_syndrome_arrays(
        counts=counts,
        n_plaq=3,
        n_star=3,
    )

    # The helper reverses each register string to convert from displayed bit order
    # to the code's integer-index order.
    assert np.array_equal(syn_star, np.array([1, 0, 1], dtype=np.uint8))
    assert np.array_equal(syn_plaq, np.array([1, 1, 0], dtype=np.uint8))


def test_counts_to_syndrome_arrays_rejects_unexpected_key_format() -> None:
    with pytest.raises(ValueError, match="Unexpected counts key format"):
        counts_to_syndrome_arrays({"010101": 1}, n_plaq=3, n_star=3)


def test_counts_to_syndrome_arrays_rejects_wrong_lengths() -> None:
    with pytest.raises(ValueError, match="Parsed syndrome lengths"):
        counts_to_syndrome_arrays({"10 01": 1}, n_plaq=3, n_star=3)
