"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  MHRQI - Multi-scale Hierarchical Representation of Quantum Images             ║
║  Denoising Application and Extensions                                        ║
║                                                                              ║
║  Author: Keno S. Jose                                                        ║
║  License: Apache 2.0                                                         ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import logging

import numpy as np
from qiskit import QuantumCircuit


def apply_denoising(qc: QuantumCircuit, pos_regs, intensity_reg, outcome=None):
    """
    Apply the hierarchical consistency denoiser to an MHRQI circuit.

    Args:
        qc: QuantumCircuit (initialized and uploaded with data)
        pos_regs: position registers
        intensity_reg: intensity register
        outcome: outcome register (must be provided if denoising is applied)

    Returns:
        tuple: (modified QuantumCircuit, denoise-only QuantumCircuit)
    """
    denoise_qc = QuantumCircuit(*qc.qregs)
    num_levels = len(pos_regs) // 2

    # ========================================
    # Ancilla allocation
    # ========================================

    work_qubits = []
    for r in denoise_qc.qregs:
        if r.name == "work":
            work_qubits = list(r)
            break

    if len(work_qubits) < 2 or outcome is None:
        logging.warning("Insufficient ancillas for denoising")
        return qc, denoise_qc

    parent_avg_ancilla = work_qubits[0]
    consistency_ancilla = work_qubits[1]
    outcome_qubit = outcome[0]
    intensity_qubits = list(intensity_reg)

    # ==========================================
    # CHECK FINEST LEVEL VS PARENT
    # ==========================================

    finest_level = num_levels - 1

    if finest_level == 0:
        denoise_qc.x(outcome_qubit)
        qc.compose(denoise_qc, inplace=True)
        return qc, denoise_qc

    qy_fine = pos_regs[2 * finest_level][0]
    qx_fine = pos_regs[2 * finest_level + 1][0]

    # === Sibling superposition ===
    denoise_qc.h(qy_fine)
    denoise_qc.h(qx_fine)

    # === Parent average encoding ===

    intensity_msb = intensity_qubits[-1]
    intensity_msb_1 = intensity_qubits[-2] if len(intensity_qubits) > 1 else intensity_msb
    intensity_msb_2 = intensity_qubits[-3] if len(intensity_qubits) > 2 else intensity_msb_1
    intensity_msb_3 = intensity_qubits[-4] if len(intensity_qubits) > 3 else intensity_msb_2

    # Flip MSB bits before rotation
    denoise_qc.x(intensity_msb)
    denoise_qc.x(intensity_msb_1)
    denoise_qc.x(intensity_msb_2)
    denoise_qc.x(intensity_msb_3)

    # Rotate ancilla proportionally to intensity bits
    denoise_qc.cry(np.pi / 16, intensity_msb, parent_avg_ancilla)
    denoise_qc.cry(np.pi / 8, intensity_msb_1, parent_avg_ancilla)
    denoise_qc.cry(np.pi / 4, intensity_msb_2, parent_avg_ancilla)
    denoise_qc.cry(np.pi / 2, intensity_msb_3, parent_avg_ancilla)

    # Unflip MSB bits
    denoise_qc.x(intensity_msb)
    denoise_qc.x(intensity_msb_1)
    denoise_qc.x(intensity_msb_2)
    denoise_qc.x(intensity_msb_3)

    # === Uncompute sibling superposition ===
    denoise_qc.h(qx_fine)
    denoise_qc.h(qy_fine)

    # === Compare pixel to parent average ===
    # XNOR logic: consistent if MSB matches parent average
    denoise_qc.x(parent_avg_ancilla)
    denoise_qc.ccx(intensity_msb, parent_avg_ancilla, consistency_ancilla)
    denoise_qc.x(parent_avg_ancilla)

    denoise_qc.x(intensity_msb)
    denoise_qc.ccx(intensity_msb, parent_avg_ancilla, consistency_ancilla)
    denoise_qc.x(parent_avg_ancilla)
    denoise_qc.x(intensity_msb)

    # === Set outcome ===
    denoise_qc.x(consistency_ancilla)
    denoise_qc.cx(consistency_ancilla, outcome_qubit)

    # === Uncompute ===
    denoise_qc.x(intensity_msb)
    denoise_qc.x(parent_avg_ancilla)
    denoise_qc.ccx(intensity_msb, parent_avg_ancilla, consistency_ancilla)
    denoise_qc.x(parent_avg_ancilla)
    denoise_qc.x(intensity_msb)

    denoise_qc.ccx(intensity_msb, parent_avg_ancilla, consistency_ancilla)

    # Uncompute parent average encoding
    denoise_qc.h(qy_fine)
    denoise_qc.h(qx_fine)

    denoise_qc.x(intensity_msb)
    denoise_qc.x(intensity_msb_1)
    denoise_qc.x(intensity_msb_2)
    denoise_qc.x(intensity_msb_3)

    denoise_qc.cry(-np.pi / 16, intensity_msb, parent_avg_ancilla)
    denoise_qc.cry(-np.pi / 8, intensity_msb_1, parent_avg_ancilla)
    denoise_qc.cry(-np.pi / 4, intensity_msb_2, parent_avg_ancilla)
    denoise_qc.cry(-np.pi / 2, intensity_msb_3, parent_avg_ancilla)

    denoise_qc.x(intensity_msb)
    denoise_qc.x(intensity_msb_1)
    denoise_qc.x(intensity_msb_2)
    denoise_qc.x(intensity_msb_3)

    denoise_qc.h(qx_fine)
    denoise_qc.h(qy_fine)

    qc.compose(denoise_qc, inplace=True)

    return qc, denoise_qc
