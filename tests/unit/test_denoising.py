import numpy as np
import pytest
from qiskit import QuantumCircuit

from mhrqi.core import denoising, representation


def test_denoiser_application():
    """Test that the denoiser can be applied to an initialized circuit."""
    depth = 2
    bit_depth = 4

    # Initialize core representation
    model = representation.MHRQI(depth=depth, bit_depth=bit_depth)
    qc = model.circuit
    pos_regs = model.pos_regs
    intensity_reg = model.intensity_reg
    outcome = model.outcome_reg

    # Apply denoising application
    qc_mod, denoise_qc = denoising.apply_denoising(qc, pos_regs, intensity_reg, outcome)

    assert isinstance(qc_mod, QuantumCircuit)
    assert isinstance(denoise_qc, QuantumCircuit)

    # Check that outcome qubit was involved (X gate or CX)
    # The actual implementation of DENOISER is complex, but we check if it composed something
    assert len(denoise_qc.data) > 0
