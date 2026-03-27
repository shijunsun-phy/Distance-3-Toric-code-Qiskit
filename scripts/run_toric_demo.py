from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from qiskit import transpile
from qiskit_aer import AerSimulator

from toric_qec import build_logicals, build_toric_pcm
from toric_qec.circuits import build_syndrome_circuit_clifford, sample_pauli_error
from toric_qec.simulate import (
    build_matchings,
    counts_to_syndrome_arrays,
    draw_toric_state,
    estimate_logical_failure_qiskit,
    syndrome_from_error,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the toric-code syndrome-extraction demo.")
    parser.add_argument("--distance", type=int, default=3, help="Linear system size L. Default: 3")
    parser.add_argument("--p", type=float, default=0.05, help="Physical depolarizing error rate.")
    parser.add_argument("--shots", type=int, default=50, help="Shots used in logical-failure estimation.")
    parser.add_argument("--seed", type=int, default=7, help="Seed for the single displayed sample.")
    parser.add_argument(
        "--outdir",
        type=Path,
        default=REPO_ROOT / "outputs",
        help="Directory for saved figures and circuit diagram.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    code, Hx, Hz = build_toric_pcm(args.distance)
    LX, LZ = build_logicals(code)
    matching_X, matching_Z = build_matchings(Hx, Hz)

    rng = np.random.default_rng(args.seed)
    x_err, z_err = sample_pauli_error(code.n, args.p, rng=rng)
    x_logical = np.array([0, 0], dtype=np.uint8)

    qc = build_syndrome_circuit_clifford(code, Hx, Hz, LX, LZ, x_logical, x_err, z_err)
    sim = AerSimulator(method="stabilizer")
    tqc = transpile(qc, sim)
    result = sim.run(tqc, shots=1).result()
    counts = result.get_counts()
    syn_plaq_meas, syn_star_meas = counts_to_syndrome_arrays(counts, code.n_plaquette, code.n_star)

    syn_plaq_expected = syndrome_from_error(Hz, x_err)
    syn_star_expected = syndrome_from_error(Hx, z_err)

    fail_rate = estimate_logical_failure_qiskit(
        code=code,
        Hx=Hx,
        Hz=Hz,
        LX=LX,
        LZ=LZ,
        matching_X=matching_X,
        matching_Z=matching_Z,
        p=args.p,
        shots=args.shots,
        use_clifford_prep=True,
        x_logical=x_logical,
    )

    try:
        fig = qc.draw("mpl")
        fig.savefig(args.outdir / "toric_code_clifford_circuit.png", bbox_inches="tight", dpi=160)
    except Exception as exc:  # pragma: no cover - visualization fallback
        print(f"[warning] Could not save circuit diagram with matplotlib drawer: {exc}")

    draw_toric_state(
        code,
        x_err=x_err,
        z_err=z_err,
        syn_star=syn_star_expected,
        syn_plaquette=syn_plaq_expected,
        title=f"Toric code sample (L={args.distance}, p={args.p})",
        savepath=args.outdir / "toric_code_sample.png",
    )

    print("=" * 72)
    print("Toric-code syndrome extraction demo")
    print("=" * 72)
    print(f"Distance / linear size L      : {code.L}")
    print(f"Physical data qubits          : {code.n}")
    print(f"Star checks / plaquette checks: {code.n_star} / {code.n_plaquette}")
    print(f"Single-shot physical error rate p: {args.p:.3f}")
    print()
    print("Single sample")
    print(f"  X-error weight              : {int(x_err.sum())}")
    print(f"  Z-error weight              : {int(z_err.sum())}")
    print(f"  Expected plaquette syndrome : {syn_plaq_expected.tolist()}")
    print(f"  Measured plaquette syndrome : {syn_plaq_meas.tolist()}")
    print(f"  Expected star syndrome      : {syn_star_expected.tolist()}")
    print(f"  Measured star syndrome      : {syn_star_meas.tolist()}")
    print()
    print(f"Estimated logical failure rate over {args.shots} shots: {fail_rate:.3f}")
    print(f"Saved outputs to: {args.outdir}")
    print("Primary artifact: Clifford-only preparation + stabilizer simulation.")


if __name__ == "__main__":
    main()
