from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pymatching import Matching
from qiskit import transpile
from qiskit_aer import AerSimulator

from .circuits import build_syndrome_circuit, build_syndrome_circuit_clifford, sample_pauli_error



def counts_to_syndrome_arrays(counts: dict, n_plaq: int, n_star: int):
    key = list(counts.keys())[0]
    parts = key.split()
    if len(parts) != 2:
        raise ValueError(f"Unexpected counts key format: {key}")

    star_bits_str, plaq_bits_str = parts[0], parts[1]
    syn_star = np.array([int(b) for b in star_bits_str[::-1]], dtype=np.uint8)
    syn_plaq = np.array([int(b) for b in plaq_bits_str[::-1]], dtype=np.uint8)

    if len(syn_star) != n_star or len(syn_plaq) != n_plaq:
        raise ValueError("Parsed syndrome lengths do not match expectations.")
    return syn_plaq, syn_star



def syndrome_from_error(H: np.ndarray, e: np.ndarray) -> np.ndarray:
    return ((H @ e) % 2).astype(np.uint8)



def build_matchings(Hx: np.ndarray, Hz: np.ndarray):
    return Matching(Hz), Matching(Hx)



def run_one_qiskit_trial(
    code,
    Hx: np.ndarray,
    Hz: np.ndarray,
    LX: np.ndarray,
    LZ: np.ndarray,
    matching_X: Matching,
    matching_Z: Matching,
    p: float,
    simulator=None,
    seed: int | None = None,
    use_clifford_prep: bool = True,
    x_logical: np.ndarray | None = None,
    psi00: np.ndarray | None = None,
):
    rng = np.random.default_rng(seed)
    x_err, z_err = sample_pauli_error(code.n, p, rng=rng)
    if simulator is None:
        method = "stabilizer" if use_clifford_prep else "automatic"
        simulator = AerSimulator(method=method)
    if x_logical is None:
        x_logical = np.array([0, 0], dtype=np.uint8)

    if use_clifford_prep:
        qc = build_syndrome_circuit_clifford(code, Hx, Hz, LX, LZ, x_logical, x_err, z_err)
    else:
        if psi00 is None:
            raise ValueError("psi00 must be supplied when use_clifford_prep is False.")
        qc = build_syndrome_circuit(code, psi00, x_err, z_err)

    tqc = transpile(qc, simulator)
    result = simulator.run(tqc, shots=1).result()
    counts = result.get_counts()
    syn_plaq, syn_star = counts_to_syndrome_arrays(counts, code.n_plaquette, code.n_star)

    x_corr = matching_X.decode(syn_plaq).astype(np.uint8)
    z_corr = matching_Z.decode(syn_star).astype(np.uint8)

    x_residual = x_err ^ x_corr
    z_residual = z_err ^ z_corr

    fail_x = bool(np.any((LZ @ x_residual) % 2))
    fail_z = bool(np.any((LX @ z_residual) % 2))

    return {
        "x_err": x_err,
        "z_err": z_err,
        "syn_plaq": syn_plaq,
        "syn_star": syn_star,
        "counts": counts,
        "fail": bool(fail_x or fail_z),
    }



def estimate_logical_failure_qiskit(
    code,
    Hx: np.ndarray,
    Hz: np.ndarray,
    LX: np.ndarray,
    LZ: np.ndarray,
    matching_X: Matching,
    matching_Z: Matching,
    p: float,
    shots: int = 30,
    use_clifford_prep: bool = True,
    x_logical: np.ndarray | None = None,
    psi00: np.ndarray | None = None,
):
    method = "stabilizer" if use_clifford_prep else "automatic"
    sim = AerSimulator(method=method)
    fails = 0
    for t in range(shots):
        out = run_one_qiskit_trial(
            code=code,
            Hx=Hx,
            Hz=Hz,
            LX=LX,
            LZ=LZ,
            matching_X=matching_X,
            matching_Z=matching_Z,
            p=p,
            simulator=sim,
            seed=1000 + t,
            use_clifford_prep=use_clifford_prep,
            x_logical=x_logical,
            psi00=psi00,
        )
        fails += int(out["fail"])
    return fails / shots



def edge_midpoint(code, q: int):
    L = code.L
    if q < L * L:
        y, x = divmod(q, L)
        return x + 0.5, y
    q2 = q - L * L
    y, x = divmod(q2, L)
    return x, y + 0.5



def draw_toric_state(
    code,
    x_err=None,
    z_err=None,
    syn_star=None,
    syn_plaquette=None,
    title: str = "Toric code sample",
    savepath: str | Path | None = None,
):
    L = code.L
    fig, ax = plt.subplots(figsize=(5, 5))

    for x in range(L + 1):
        ax.plot([x, x], [0, L], color="lightgray", lw=1)
    for y in range(L + 1):
        ax.plot([0, L], [y, y], color="lightgray", lw=1)

    for x in range(L):
        for y in range(L):
            ax.scatter([x], [y], s=25)

    if x_err is not None:
        for q in np.where(x_err)[0]:
            xm, ym = edge_midpoint(code, q)
            ax.scatter([xm], [ym], s=120, marker="s")

    if z_err is not None:
        for q in np.where(z_err)[0]:
            xm, ym = edge_midpoint(code, q)
            ax.scatter([xm], [ym], s=120, marker="x")

    if syn_star is not None:
        for s in np.where(syn_star)[0]:
            y, x = divmod(s, L)
            ax.scatter([x], [y], s=180, marker="o", facecolors="none", edgecolors="purple")

    if syn_plaquette is not None:
        for p in np.where(syn_plaquette)[0]:
            y, x = divmod(p, L)
            ax.scatter([x + 0.5], [y + 0.5], s=180, marker="^", facecolors="none", edgecolors="orange")

    ax.set_xlim(-0.2, L + 0.2)
    ax.set_ylim(-0.2, L + 0.2)
    ax.set_aspect("equal")
    ax.set_xticks(range(L + 1))
    ax.set_yticks(range(L + 1))
    ax.set_title(title)

    if savepath is not None:
        savepath = Path(savepath)
        savepath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savepath, bbox_inches="tight", dpi=160)
    return fig, ax
