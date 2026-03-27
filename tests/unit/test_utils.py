import numpy as np
import pytest

from mhrqi.utils import general as utils


def test_angle_map():
    """Test intensity to angle mapping."""
    img = np.array([[0, 255]], dtype=np.uint8)
    angles = utils.angle_map(img, bit_depth=8)

    assert angles[0, 0] == 0.0  # 0 intensity -> 0 angle
    assert np.isclose(angles[0, 1], np.pi)  # 255 intensity -> pi angle


def test_get_max_depth():
    """Test hierarchy level calculation."""
    assert utils.get_max_depth(64, 2) == 6
    assert utils.get_max_depth(16, 2) == 4


def test_generate_hierarchical_coord_matrix():
    """Test generating coordinate matrix."""
    matrix = utils.generate_hierarchical_coord_matrix(4, 2)
    assert len(matrix) == 16
    assert len(matrix[0]) == 4  # 2*log2(4) = 4


def test_compose_rc():
    """Test hierarchical coordinate to (r, c) conversion."""
    # level 0: (qy0, qx0)
    # image 2x2, d=2
    # vec [0, 0] -> (0, 0)
    # vec [0, 1] -> (0, 1)
    # vec [1, 0] -> (1, 0)
    # vec [1, 1] -> (1, 1)

    assert utils.compose_rc([0, 0], d=2) == (0, 0)
    assert utils.compose_rc([0, 1], d=2) == (0, 1)
    assert utils.compose_rc([1, 0], d=2) == (1, 0)
    assert utils.compose_rc([1, 1], d=2) == (1, 1)

    # 4x4, d=2, 2 levels
    # vec [qy0, qx0, qy1, qx1]
    # vec [0, 0, 1, 1] -> r=0*2+1=1, c=0*2+1=1 -> (1, 1)
    assert utils.compose_rc([0, 0, 1, 1], d=2) == (1, 1)
