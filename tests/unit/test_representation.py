import numpy as np
import pytest

from mhrqi.core import representation as circuit


def test_circuit_init():
    """Test that MHRQI initialization creates a circuit with correct registers."""
    depth = 3
    bit_depth = 8

    model = circuit.MHRQI(depth, bit_depth)
    qc = model.circuit
    pos_regs = model.pos_regs
    intensity_reg = model.intensity_reg
    outcome = model.outcome_reg

    # Check number of position qubits (2 * depth)
    assert len(pos_regs) == 2 * depth

    # Check intensity register size
    assert len(intensity_reg) == bit_depth

    # Check outcome qubit
    assert outcome is not None

    # Check total qubits: (2*depth) + bit_depth + 1 (outcome) + 2 (work)
    expected_qubits = (2 * depth) + bit_depth + 1 + 2
    assert qc.num_qubits == expected_qubits


def test_prepare_controls():
    """Test the internal control preparation helper."""
    from qiskit import QuantumCircuit, QuantumRegister

    qr = QuantumRegister(2)
    qc = QuantumCircuit(qr)

    controls = [qr[0], qr[1]]
    ctrl_states = [1, 0]

    circuit._prepare_controls_on_states(qc, controls, ctrl_states)

    # Line 0: no gate (QR[0] is 1, so no X)
    # Line 1: X gate on QR[1] (since its state is 0)
    assert qc.data[0].operation.name == "x"
    assert qc.data[0].qubits[0] == qr[1]


def test_lazy_upload():
    """Test the fast statevector-based upload."""
    depth = 2  # 4x4 image
    bit_depth = 1
    model = circuit.MHRQI(depth, bit_depth)

    # 4x4 coordinate matrix (d=2)
    from mhrqi.utils import general as utils

    coords = utils.generate_hierarchical_coord_matrix(4, 2)

    # Simple checkerboard
    img = np.array([[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]])

    model.lazy_upload(coords, img)

    # Verify the circuit has a SetStatevector instruction
    found_sv = False
    for instr in model.circuit.data:
        if instr.operation.name == "set_statevector":
            found_sv = True
            break
    assert found_sv
