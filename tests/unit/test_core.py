"""Unit tests for toric-code algebraic construction helpers."""

from __future__ import annotations

import numpy as np

from toric_qec.core import (
    binary_rank,
    build_logicals,
    build_toric_pcm,
    gf2_row_basis,
)


def test_toric_code_basic_dimensions_for_multiple_distances() -> None:
    """The L x L toric code should have 2L^2 data qubits and L^2 checks of each type."""
    for distance in (3, 4, 5):
        code, Hx, Hz = build_toric_pcm(distance)

        assert code.L == distance
        assert code.n == 2 * distance * distance
        assert code.n_star == distance * distance
        assert code.n_plaquette == distance * distance
        assert Hx.shape == (distance * distance, 2 * distance * distance)
        assert Hz.shape == (distance * distance, 2 * distance * distance)


def test_each_stabilizer_check_has_weight_four() -> None:
    """Each star and plaquette check in the square-lattice toric code has weight 4."""
    _, Hx, Hz = build_toric_pcm(3)

    assert np.all(Hx.sum(axis=1) == 4)
    assert np.all(Hz.sum(axis=1) == 4)


def test_css_stabilizer_checks_commute() -> None:
    """CSS X and Z checks must commute: Hx Hz^T = 0 over GF(2)."""
    _, Hx, Hz = build_toric_pcm(3)

    commutator_matrix = (Hx @ Hz.T) % 2

    assert np.array_equal(commutator_matrix, np.zeros_like(commutator_matrix))


def test_distance_three_encodes_two_logical_qubits() -> None:
    """For the toric code on a torus, k = n - rank(Hx) - rank(Hz) should be 2."""
    code, Hx, Hz = build_toric_pcm(3)

    encoded_qubits = code.n - binary_rank(Hx) - binary_rank(Hz)

    assert encoded_qubits == 2


def test_logical_operators_commute_with_stabilizers_and_pair_nontrivially() -> None:
    """Logical strings should commute with stabilizers and have nontrivial X/Z pairing."""
    code, Hx, Hz = build_toric_pcm(3)
    LX, LZ = build_logicals(code)

    assert LX.shape == (2, code.n)
    assert LZ.shape == (2, code.n)

    # Logical X strings commute with Z checks; logical Z strings commute with X checks.
    assert np.all((Hz @ LX.T) % 2 == 0)
    assert np.all((Hx @ LZ.T) % 2 == 0)

    # With the repo's current logical-string convention, the L=3 pairing is off-diagonal:
    # X_1 pairs with Z_2 and X_2 pairs with Z_1.
    expected_pairing = np.eye(2, dtype=np.uint8)
    assert np.array_equal((LX @ LZ.T) % 2, expected_pairing)


def test_gf2_row_basis_preserves_rank_and_is_independent() -> None:
    """The GF(2) row-basis routine should return an independent basis of the row span."""
    _, Hx, _ = build_toric_pcm(3)

    basis = gf2_row_basis(Hx)

    assert basis.ndim == 2
    assert binary_rank(basis) == basis.shape[0]
    assert binary_rank(basis) == binary_rank(Hx)
