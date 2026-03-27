"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  MHRQI - Multi-scale Hierarchical Representation of Quantum Images             ║
║  Core Representation and Encoding                                            ║
║                                                                              ║
║  Author: Keno S. Jose                                                        ║
║  License: Apache 2.0                                                         ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import itertools
import warnings
from collections import defaultdict

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit.library import MCXGate
from qiskit_aer import Aer, AerSimulator
from qiskit_aer.library import SetStatevector

from mhrqi.core.denoising import apply_denoising
from mhrqi.core.results import MHRQIResult
from mhrqi.utils import general as utils


# -------------------------
# Helper primitives
# -------------------------
def _prepare_controls_on_states(qc: QuantumCircuit, controls, ctrl_states):
    """Flip controls where ctrl_state == 0 so that all required controls are 1."""
    for q, s in zip(controls, ctrl_states):
        if int(s) == 0:
            qc.x(q)


def _restore_controls(qc: QuantumCircuit, controls, ctrl_states):
    for q, s in zip(controls, ctrl_states):
        if int(s) == 0:
            qc.x(q)


def apply_multi_controlled_ry(qc, controls, ctrl_states, target, ancilla_for_and, angle):
    _prepare_controls_on_states(qc, controls, ctrl_states)

    try:
        if len(controls) == 0:
            qc.ry(angle, target)
            return

        if len(controls) == 1:
            # Single control: use controlled RY directly
            qc.cry(angle, controls[0], target)
            return

        qc.reset(ancilla_for_and)

        qc.mcx(controls, ancilla_for_and)
        qc.cry(angle, ancilla_for_and, target)
        qc.mcx(controls, ancilla_for_and)  # Uncompute

        qc.reset(ancilla_for_and)

    finally:
        _restore_controls(qc, controls, ctrl_states)


# -------------------------
# Circuit construction
# -------------------------


class MHRQI:
    """
    Main class for Multi-scale Hierarchical Representation of Quantum Images.
    Encapsulates the quantum circuit, registers, and operations.
    """

    def __init__(self, depth, bit_depth=8):
        """
        Initialize an MHRQI instance.

        Args:
            depth: Hierarchy depth (log2 of image side)
            bit_depth: Bits for intensity encoding (default 8)
        """
        self.depth = depth
        self.bit_depth = bit_depth
        self.hierarchical_coord_matrix = None
        self.denoise_enabled = False

        self.pos_regs = []
        # Position qubits (2 per level: y and x)
        for k in range(depth):
            self.pos_regs.append(QuantumRegister(1, f"q_y_{k}"))
            self.pos_regs.append(QuantumRegister(1, f"q_x_{k}"))

        self.intensity_reg = QuantumRegister(bit_depth, "intensity")
        self.outcome_reg = QuantumRegister(1, "outcome")
        self.work_reg = QuantumRegister(2, "work")

        self.circuit = QuantumCircuit(
            *self.pos_regs, self.intensity_reg, self.outcome_reg, self.work_reg
        )

        # Place position qubits in uniform superposition
        for reg in self.pos_regs:
            self.circuit.h(reg[0])

    @property
    def qubits_to_measure(self):
        """Returns a list of qubits that should be measured for reconstruction."""
        qubits = []
        for reg in self.pos_regs:
            qubits.append(reg[0])
        qubits.extend(list(self.intensity_reg))
        qubits.extend(list(self.outcome_reg))
        return qubits

    def upload(self, hierarchical_coord_matrix, img):
        """
        Upload intensity values using MHRQI(basis) encoding.

        Args:
            hierarchical_coord_matrix: matrix of position states
            img: normalized image (0-1 range)
        """
        self.hierarchical_coord_matrix = hierarchical_coord_matrix
        controls = [reg[0] for reg in self.pos_regs]
        intensity_qubits = list(self.intensity_reg)
        and_ancilla = self.work_reg[0]

        for vec in hierarchical_coord_matrix:
            ctrl_states = list(vec)
            r, c = utils.compose_rc(vec, 2)  # d=2 for images
            pixel_value = float(img[r, c])
            intensity_int = int(pixel_value * (2**self.bit_depth - 1))
            intensity_bits = format(intensity_int, f"0{self.bit_depth}b")[::-1]

            _prepare_controls_on_states(self.circuit, controls, ctrl_states)

            if len(controls) > 0:
                self.circuit.mcx(controls, and_ancilla)
                for bit_idx, bit_val in enumerate(intensity_bits):
                    if bit_val == "1":
                        self.circuit.cx(and_ancilla, intensity_qubits[bit_idx])
                self.circuit.mcx(controls, and_ancilla)
            else:
                for bit_idx, bit_val in enumerate(intensity_bits):
                    if bit_val == "1":
                        self.circuit.x(intensity_qubits[bit_idx])

            _restore_controls(self.circuit, controls, ctrl_states)
        return self.circuit

    def lazy_upload(self, hierarchical_coord_matrix, image):
        """
        Fast basis upload using direct statevector initialization.

        Args:
            hierarchical_coord_matrix (list): List of hierarchical_coord_vectors for all pixels.
            image (np.ndarray): Normalized image array [0, 1].

        Returns:
            QuantumCircuit: The initialized circuit.
        """
        self.hierarchical_coord_matrix = hierarchical_coord_matrix
        qubit_to_idx = {q: i for i, q in enumerate(self.circuit.qubits)}
        pos_indices = [qubit_to_idx[reg[0]] for reg in self.pos_regs]
        intensity_indices = [qubit_to_idx[q] for q in self.intensity_reg]

        num_qubits = self.circuit.num_qubits
        dim = 2**num_qubits
        state = np.zeros(dim, dtype=complex)

        all_indices = pos_indices + intensity_indices
        is_sequential = all(all_indices[i] == i for i in range(len(all_indices)))

        if is_sequential:
            for hierarchical_coord_vector in hierarchical_coord_matrix:
                p = 0
                for i, val in enumerate(hierarchical_coord_vector):
                    if val:
                        p |= 1 << i

                r, c = utils.compose_rc(hierarchical_coord_vector, 2)
                pixel_value = float(image[r, c])
                intensity_int = int(pixel_value * (2**self.bit_depth - 1))

                base_idx = p
                intensity_offset = 0
                for bit_idx in range(self.bit_depth):
                    if (intensity_int >> bit_idx) & 1:
                        intensity_offset |= 1 << (len(pos_indices) + bit_idx)
                state[base_idx + intensity_offset] = 1.0
        else:
            warnings.warn("Qubits not sequential, falling back to gate-based upload.", stacklevel=2)
            return self.upload(hierarchical_coord_matrix, image)

        state_norm = np.linalg.norm(state)
        if state_norm > 0:
            state = state / state_norm

        self.circuit.append(SetStatevector(state), self.circuit.qubits)
        return self.circuit

    def simulate(self, shots=None, use_gpu=False):
        """
        Simulate the MHRQI circuit.

        Args:
            shots (int, optional): Number of shots. Returns statevector if None.
            use_gpu (bool): Whether to use GPU acceleration (if available).

        Returns:
            MHRQIResult: Simulation results object.
        """
        if shots is None:
            backend = Aer.get_backend("statevector_simulator", device="GPU" if use_gpu else "CPU")
            transpiled = transpile(self.circuit, backend)
            raw = backend.run(transpiled).result().get_statevector()
        else:
            qc_measure = self.circuit.copy()
            creg = ClassicalRegister(len(self.qubits_to_measure), "c")
            qc_measure.add_register(creg)
            qc_measure.measure(self.qubits_to_measure, creg)

            if use_gpu:
                try:
                    backend = AerSimulator(
                        method="statevector", device="GPU", cuStateVec_enable=True
                    )
                except Exception:
                    backend = Aer.get_backend("qasm_simulator")
            else:
                backend = Aer.get_backend("qasm_simulator")

            transpiled = transpile(qc_measure, backend)
            raw = backend.run(transpiled, shots=shots).result().get_counts()

        return MHRQIResult(
            raw, self.hierarchical_coord_matrix, self.bit_depth, self.denoise_enabled
        )

    def apply_denoising(self):
        """
        Apply hierarchical consistency denoising to the circuit.
        """
        _, denoise_qc = apply_denoising(
            self.circuit, self.pos_regs, self.intensity_reg, self.outcome_reg
        )
        self.denoise_enabled = True
        return denoise_qc

    def decode(self, results):
        """
        Decode simulation results into image bins.

        Args:
            results (dict or Statevector): Simulation results.

        Returns:
            dict: Reconstructed image bins.
        """
        # The actual decoding logic is now handled within MHRQIResult
        # This method might be removed or refactored depending on usage
        # For now, it can serve as a placeholder or direct call to MHRQIResult's decode
        mhrqi_result = MHRQIResult(
            results, self.hierarchical_coord_matrix, self.bit_depth, self.denoise_enabled
        )
        return mhrqi_result.decode()


# -------------------------
# Lazy Upload (Faster)
# -------------------------

# End of MHRQI class
# -------------------------
# Private Binning Helpers
# -------------------------


# Position extraction logic moved to utils/result.py
