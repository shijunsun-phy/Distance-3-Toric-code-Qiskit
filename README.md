# Toric code syndrome extraction + MWPM decoding in Qiskit

**Shijun Sun**

This repository is a compact demo of building and simulating a toric-code quantum error-correction workflow in Qiskit.

It starts from the lattice geometry, constructs the parity-check matrices, prepares a logical toric-code state, injects Pauli noise, measures stabilizer syndromes with ancillas, and decodes with minimum-weight perfect matching (MWPM). The key engineering upgrade over the original notebook is a **Clifford-only logical-state preparation path**, so the full circuit can be run with **efficient stabilizer simulation** rather than a generic statevector initializer.

---

## Physics intuition

The toric code is one of the cleanest models of topological quantum memory.

- Physical qubits live on the **edges** of a square lattice with periodic boundary conditions.
- The code space is defined by commuting local constraints:
  - **star operators** enforce an X-type parity around each vertex,
  - **plaquette operators** enforce a Z-type parity around each face.
- Local Pauli errors create pairs of syndrome defects, while noncontractible loops correspond to **logical operators**.

---

## What this repo includes

This project is built around the notebook workflow, but reorganized into a small codebase that is easy to run and review.

### 1. Geometry → parity-check matrices
The code explicitly maps the periodic square lattice into binary check matrices `Hx` and `Hz`.

### 2. Logical operators and encoded states
The repository constructs the two independent noncontractible logical strings on the torus and uses them to label logical basis states.

### 3. Syndrome-extraction circuits in Qiskit
Ancilla-assisted circuits measure plaquette and star stabilizers in a hardware-style way, rather than computing syndromes only classically.

### 4. Physical Pauli noise injection
A simple depolarizing sampling routine inserts X, Y, and Z faults directly into the quantum circuit.

### 5. MWPM decoding
Measured syndromes are passed to `pymatching`, which returns candidate corrections for the X and Z sectors.

### 6. Clifford state preparation for efficient simulation
To be compatible with the stabilizer simulation pathway, I use a Clifford-only preparation circuit for the toric-code logical state, so the end-to-end demo can run efficiently using `AerSimulator(method="stabilizer")`.

---

## Quickstart

Create an environment and install dependencies:

```bash
pip install -r requirements.txt
```

Run the main demo in one line:

```bash
python scripts/run_toric_demo.py
```

That command will:

1. build the distance-3 toric-code circuit,
2. prepare the logical state with a Clifford circuit from `|0...0⟩`,
3. inject Pauli noise,
4. measure the stabilizer syndromes,
5. decode with MWPM,
6. estimate a logical failure rate,
7. save a sample circuit diagram and lattice plot into `outputs/`.

You can also vary the distance and noise rate:

```bash
python scripts/run_toric_demo.py --distance 3 --p 0.08 --shots 100
```

---

## Repository layout

```text
.
├── README.md
├── requirements.txt
├── notebooks/
│   └── toric_code_qiskit_mwpm_clifford.ipynb
├── scripts/
│   └── run_toric_demo.py
├── tests/
│   └── test_toric_helpers.py
└── toric_qec/
    ├── __init__.py
    ├── core.py
    ├── circuits.py
    └── simulate.py
```

- `notebooks/` keeps the original exploratory workflow and presentation-style narrative.
- `toric_qec/` turns the notebook logic into reusable Python modules.
- `scripts/run_toric_demo.py` is the primary recruiter-facing entrypoint.
- `tests/` keeps one lightweight sanity check without bloating the repo.


## Notes

- The demo is intentionally small and self-contained.
- The included notebook remains the best place to see the derivation and exploratory reasoning; the script and package are the fastest way to run the demo.
