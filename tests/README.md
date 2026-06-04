# Test suite layout

This test suite is organized into two layers.

## Unit tests

Located in `tests/unit/`.

These tests check individual algebraic and helper functions:

- toric-code dimensions,
- stabilizer weights,
- CSS commutation,
- encoded-qubit count,
- logical-string commutation/pairing,
- GF(2) row-basis rank preservation,
- Pauli-error sampling,
- Pauli-label conversion,
- syndrome calculation,
- count-string parsing.

Run only unit tests with:

```bash
pytest tests/unit -v
```

## Integration tests

Located in `tests/integration/`.

These tests check that independently written components agree:

- the Qiskit stabilizer-simulator syndrome agrees with the classical parity-check syndrome,
- PyMatching corrects simple single-qubit X/Z errors without logical failure,
- the noiseless Qiskit trial path has zero syndrome and no logical failure.

Run only integration tests with:

```bash
pytest tests/integration -v
```

Run everything with:

```bash
pytest -v
```
