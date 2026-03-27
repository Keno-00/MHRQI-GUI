import numpy as np
import pytest

from mhrqi.core.results import MHRQIResult


def test_result_reconstruction_statevector():
    # Mock data for a 2x2 image
    # Pos: 2 qubits (y, x), Bit-depth: 1
    # Total bits: 3 (pos_y, pos_x, intensity)
    hierarchical_coord_matrix = [(0, 0), (0, 1), (1, 0), (1, 1)]
    # Statevector with 2^3 = 8 components
    # Indices:
    # 000: pos=(0,0), int=0
    # 001: pos=(0,1), int=0
    # 010: pos=(1,0), int=0
    # 011: pos=(1,1), int=0
    # 100: pos=(0,0), int=1
    # 101: pos=(0,1), int=1
    # 110: pos=(1,0), int=1
    # 111: pos=(1,1), int=1
    sv = np.zeros(8, dtype=complex)
    sv[4] = 1.0 / np.sqrt(2)  # (0,0) -> 1
    sv[1] = 1.0 / np.sqrt(2)  # (0,1) -> 0

    result = MHRQIResult(sv, hierarchical_coord_matrix, bit_depth=1)
    recon = result.reconstruct()

    assert recon.shape == (2, 2)
    assert recon[0, 0] == 1.0
    assert recon[0, 1] == 0.0


def test_result_reconstruction_counts():
    hierarchical_coord_matrix = [(0, 0), (0, 1), (1, 0), (1, 1)]
    # Counts for 3 bits: (int, pos_x, pos_y) reversed in Qiskit output
    # but MHRQIResult.bins handles string manipulation correctly.
    # Actually, MHRQIResult._make_bins_counts uses b = bitstring[::-1]
    # So if b = "001", pos=(0,0), int=1
    counts = {
        "100": 10,  # b="001" -> pos=(0,0), int=1
        "010": 10,  # b="010" -> pos=(0,1), int=0
    }

    result = MHRQIResult(counts, hierarchical_coord_matrix, bit_depth=1)
    recon = result.reconstruct()

    assert recon.shape == (2, 2)
    assert recon[0, 0] == 1.0
    assert recon[0, 1] == 0.0
