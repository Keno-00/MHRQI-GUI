"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  MHRQI - Multi-scale Hierarchical Representation of Quantum Images            ║
║  Utility Functions: Encoding, Reconstruction, Sibling Smoothing             ║
║                                                                              ║
║  Author: Keno S. Jose                                                        ║
║  License: Apache 2.0                                                         ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import math

import numpy as np


def angle_map(img, bit_depth=8):
    """
    Map pixel intensities to quantum rotation angles via arcsin encoding.

    Args:
        img: Grayscale image as integer array.
        bit_depth: Bit depth of the image (default 8).

    Returns:
        Array of angles in [0, π].
    """
    max_val = (1 << bit_depth) - 1
    u = np.clip(img.astype(np.float64) / max_val, 0.0, 1.0)
    theta = 2.0 * np.arcsin(np.sqrt(u))
    return theta


def get_max_depth(N, d):
    """
    Compute the maximum hierarchy depth for image size N and qudit dimension d.

    Args:
        N: Image side length.
        d: Qudit dimension.

    Returns:
        max_depth = floor(log_d(N)).
    """
    max_depth = math.floor(math.log(N, d))
    return max_depth


def get_subdiv_size(k, N, d):
    """
    Compute the subdivision size at hierarchy level k.

    Args:
        k: Hierarchy level.
        N: Image side length.
        d: Qudit dimension.

    Returns:
        Side length of subregions at level k.
    """
    s = N / (d**k)
    return s


def compute_register(r, c, d, sk_prev):
    """
    Compute the qudit register values (qy, qx) for pixel (r, c) at a given scale.

    Args:
        r: Row index.
        c: Column index.
        d: Qudit dimension.
        sk_prev: Subdivision size at the previous level.

    Returns:
        Tuple (qy, qx).
    """
    qx = min(math.floor((c % sk_prev) * (d / sk_prev)), d - 1)
    qy = min(math.floor((r % sk_prev) * (d / sk_prev)), d - 1)
    return qy, qx


def compose_rc(hierarchical_coord_vector, d=2):
    """
    Convert a hierarchical coordinate vector to (row, col) pixel coordinates.

    Args:
        hierarchical_coord_vector: Sequence of qudit values (qy0, qx0, qy1, qx1, ...). Length must be even.
        d: Qudit dimension.

    Returns:
        Tuple (r, c).

    Raises:
        ValueError: If hierarchical_coord_vector length is odd or any digit is out of range.
    """
    if len(hierarchical_coord_vector) % 2 != 0:
        raise ValueError("hierarchical_coord_vector length must be even (pairs of qy,qx).")

    qy_digits = hierarchical_coord_vector[0::2]
    qx_digits = hierarchical_coord_vector[1::2]

    r = 0
    c = 0
    for digit in qy_digits:
        if not (0 <= digit < d):
            raise ValueError("qy digit out of range for given d.")
        r = r * d + int(digit)

    for digit in qx_digits:
        if not (0 <= digit < d):
            raise ValueError("qx digit out of range for given d.")
        c = c * d + int(digit)

    return r, c


def mhrqi_bins_to_image(
    bins, hierarchical_coord_matrix, d, image_shape, bias_stats=None, original_img=None
):
    """
    Reconstruct an image from measurement bins with optional confidence-weighted smoothing.

    When bias_stats is provided, each pixel is blended with its 8-neighborhood
    weighted by its denoiser confidence. Neighbors with confidence below
    CONFIDENCE_THRESHOLD are used as context; high-confidence pixels are
    trusted as-is.

    Args:
        bins: Measurement bins dict mapping position tuples to intensity stats.
        hierarchical_coord_matrix: List of hierarchical coordinate vectors.
        d: Qudit dimension.
        image_shape: Output image shape as (H, W).
        bias_stats: Optional dict mapping position tuples to hit/miss counts.
        original_img: Optional pre-computed baseline image to use as source.

    Returns:
        Reconstructed image as a float array of shape image_shape.
    """
    img = np.zeros(image_shape)
    N = image_shape[0]

    reconstructed_baseline = np.zeros(image_shape)
    for hierarchical_coord_vector in hierarchical_coord_matrix:
        key = tuple(hierarchical_coord_vector)
        if key in bins and bins[key].get("count", 0) > 0:
            avg_intensity = bins[key]["intensity_sum"] / bins[key]["count"]
            r, c = compose_rc(hierarchical_coord_vector, d)
            reconstructed_baseline[r, c] = avg_intensity

    source_img = original_img if original_img is not None else reconstructed_baseline

    if bias_stats is None:
        return source_img

    CONFIDENCE_THRESHOLD = 0.7

    confidence_map = np.ones(image_shape) * 0.5
    for hierarchical_coord_vector in hierarchical_coord_matrix:
        key = tuple(hierarchical_coord_vector)
        r, c = compose_rc(hierarchical_coord_vector, d)
        if key in bias_stats:
            hit = bias_stats[key].get("hit", 0)
            miss = bias_stats[key].get("miss", 0)
            total = hit + miss
            confidence_map[r, c] = hit / total if total > 0 else 0.5

    for hierarchical_coord_vector in hierarchical_coord_matrix:
        key = tuple(hierarchical_coord_vector)
        r, c = compose_rc(hierarchical_coord_vector, d)
        confidence = confidence_map[r, c]

        trusted_neighbor_vals = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < N and 0 <= nc < N and confidence_map[nr, nc] <= CONFIDENCE_THRESHOLD:
                    trusted_neighbor_vals.append(source_img[nr, nc])

        if len(trusted_neighbor_vals) == 0:
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < N and 0 <= nc < N:
                        trusted_neighbor_vals.append(source_img[nr, nc])

        context_avg = (
            np.median(trusted_neighbor_vals) if trusted_neighbor_vals else source_img[r, c]
        )

        img[r, c] = (confidence * source_img[r, c]) + ((1 - confidence) * context_avg)

    return img


def generate_hierarchical_coord_matrix(N, d=2):
    """
    Generate the hierarchical coordinate matrix for an image of size N x N.

    Args:
        N: Image side length.
        d: Qudit dimension.

    Returns:
        List of hierarchical coordinate vectors.
    """
    max_depth = get_max_depth(N, d)
    subdiv_sizes = []
    for level in range(0, max_depth):
        subdiv_sizes.append(N if level == 0 else get_subdiv_size(level, N, d))

    hierarchical_coord_matrix = []
    for r, c in np.ndindex(N, N):
        hierarchical_coord_vector = []
        for size in subdiv_sizes:
            sub_hcv = compute_register(r, c, d, size)
            hierarchical_coord_vector.extend(sub_hcv)
        hierarchical_coord_matrix.append(hierarchical_coord_vector)
    return hierarchical_coord_matrix
