from __future__ import annotations

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.synthesis import synth_circuit_from_stabilizers

from .core import gf2_row_basis


def append_plaquette_measurements(qc: QuantumCircuit, code, data_reg, plaq_reg, c_plaq) -> None:
    for x in range(code.L):
        for y in range(code.L):
            p = code.plaquette_index(x, y)
            for q in code.plaquette_edges(x, y):
                qc.cx(data_reg[q], plaq_reg[p])
            qc.measure(plaq_reg[p], c_plaq[p])



def append_star_measurements(qc: QuantumCircuit, code, data_reg, star_reg, c_star) -> None:
    for x in range(code.L):
        for y in range(code.L):
            s = code.star_index(x, y)
            qc.h(star_reg[s])
            for q in code.star_edges(x, y):
                qc.cx(star_reg[s], data_reg[q])
            qc.h(star_reg[s])
            qc.measure(star_reg[s], c_star[s])



def sample_pauli_error(n: int, p: float, rng=None):
    """IID depolarizing sample represented as binary X/Z supports."""
    if rng is None:
        rng = np.random.default_rng()

    x_err = np.zeros(n, dtype=np.uint8)
    z_err = np.zeros(n, dtype=np.uint8)

    for q in range(n):
        if rng.random() >= p:
            continue
        pauli = rng.integers(1, 4)  # 1=X, 2=Y, 3=Z
        if pauli == 1:
            x_err[q] = 1
        elif pauli == 2:
            x_err[q] = 1
            z_err[q] = 1
        else:
            z_err[q] = 1

    return x_err, z_err



def apply_pauli_error_gates(qc: QuantumCircuit, data_reg, x_err: np.ndarray, z_err: np.ndarray) -> None:
    for q in range(len(x_err)):
        if x_err[q] and z_err[q]:
            qc.y(data_reg[q])
        elif x_err[q]:
            qc.x(data_reg[q])
        elif z_err[q]:
            qc.z(data_reg[q])



def build_syndrome_circuit(code, psi00: np.ndarray, x_err: np.ndarray, z_err: np.ndarray) -> QuantumCircuit:
    """Notebook-original version using initialize(statevector)."""
    data = QuantumRegister(code.n, "data")
    plaq = QuantumRegister(code.n_plaquette, "plaq")
    star = QuantumRegister(code.n_star, "star")
    c_plaq = ClassicalRegister(code.n_plaquette, "c_plaq")
    c_star = ClassicalRegister(code.n_star, "c_star")

    qc = QuantumCircuit(data, plaq, star, c_plaq, c_star)
    qc.initialize(psi00, data)
    qc.barrier()
    apply_pauli_error_gates(qc, data, x_err, z_err)
    qc.barrier()
    append_plaquette_measurements(qc, code, data, plaq, c_plaq)
    qc.barrier()
    append_star_measurements(qc, code, data, star, c_star)
    return qc



def binary_symplectic_row_to_pauli_label(x_row: np.ndarray, z_row: np.ndarray) -> str:
    """Convert binary X/Z supports into a signed Qiskit Pauli label."""
    x_row = np.asarray(x_row, dtype=np.uint8)
    z_row = np.asarray(z_row, dtype=np.uint8)
    if x_row.shape != z_row.shape:
        raise ValueError("x_row and z_row must have the same shape.")

    chars = []
    for x, z in zip(x_row, z_row):
        if x and z:
            chars.append("Y")
        elif x:
            chars.append("X")
        elif z:
            chars.append("Z")
        else:
            chars.append("I")

    return "+" + "".join(chars[::-1])



def build_logical_zero_zero_stabilizer_labels(Hx: np.ndarray, Hz: np.ndarray, LZ: np.ndarray):
    """Independent stabilizer set that fixes the logical |00> sector."""
    n = Hx.shape[1]
    x_basis = gf2_row_basis(Hx)
    z_basis = gf2_row_basis(Hz)

    labels = []
    for row in x_basis:
        labels.append(binary_symplectic_row_to_pauli_label(row, np.zeros(n, dtype=np.uint8)))
    for row in z_basis:
        labels.append(binary_symplectic_row_to_pauli_label(np.zeros(n, dtype=np.uint8), row))
    for row in LZ:
        labels.append(binary_symplectic_row_to_pauli_label(np.zeros(n, dtype=np.uint8), row))

    if len(labels) != n:
        raise ValueError(f"Need exactly n={n} independent stabilizers, but got {len(labels)}.")
    return labels



def build_logical_state_prep_clifford(
    Hx: np.ndarray,
    Hz: np.ndarray,
    LX: np.ndarray,
    LZ: np.ndarray,
    x_logical: np.ndarray,
) -> QuantumCircuit:
    """Clifford-only preparation of the logical toric-code basis state."""
    x_logical = np.asarray(x_logical, dtype=np.uint8)
    if x_logical.shape != (2,):
        raise ValueError("x_logical must be a length-2 binary vector.")

    stabilizer_labels = build_logical_zero_zero_stabilizer_labels(Hx, Hz, LZ)
    prep = synth_circuit_from_stabilizers(stabilizer_labels)
    for i in range(2):
        if x_logical[i]:
            for q in np.flatnonzero(LX[i]):
                prep.x(int(q))
    return prep



def build_syndrome_circuit_clifford(
    code,
    Hx: np.ndarray,
    Hz: np.ndarray,
    LX: np.ndarray,
    LZ: np.ndarray,
    x_logical: np.ndarray,
    x_err: np.ndarray,
    z_err: np.ndarray,
) -> QuantumCircuit:
    """Efficient-simulation version using Clifford state preparation."""
    data = QuantumRegister(code.n, "data")
    plaq = QuantumRegister(code.n_plaquette, "plaq")
    star = QuantumRegister(code.n_star, "star")
    c_plaq = ClassicalRegister(code.n_plaquette, "c_plaq")
    c_star = ClassicalRegister(code.n_star, "c_star")

    qc = QuantumCircuit(data, plaq, star, c_plaq, c_star)
    prep = build_logical_state_prep_clifford(Hx, Hz, LX, LZ, x_logical)
    qc.compose(prep, qubits=data, inplace=True)
    qc.barrier()
    apply_pauli_error_gates(qc, data, x_err, z_err)
    qc.barrier()
    append_plaquette_measurements(qc, code, data, plaq, c_plaq)
    qc.barrier()
    append_star_measurements(qc, code, data, star, c_star)
    return qc
