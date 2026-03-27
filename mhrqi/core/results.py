from collections import defaultdict

import numpy as np

from mhrqi.utils import general as utils


class MHRQIResult:
    """
    Encapsulates the result of an MHRQI simulation, providing methods for
    decoding, reconstruction, and metric calculation.
    """

    def __init__(self, raw_results, hierarchical_coord_matrix, bit_depth=8, denoise=False):
        """
        Initialize an MHRQIResult instance.

        Args:
            raw_results (dict or Statevector): The raw data from simulation.
            hierarchical_coord_matrix (list): The matrix used for current upload.
            bit_depth (int): Bit depth of intensities.
            denoise (bool): Whether denoising was enabled.
        """
        self.raw_results = raw_results
        self.hierarchical_coord_matrix = hierarchical_coord_matrix
        self.bit_depth = bit_depth
        self.denoise = denoise
        self._image = None
        self._bins = None
        self._bias_stats = None

    @property
    def bins(self):
        """Lazy-loaded decoded bins and bias stats."""
        if self._bins is None:
            if isinstance(self.raw_results, dict):
                decoded = _make_bins_counts(
                    self.raw_results, self.hierarchical_coord_matrix, self.bit_depth, self.denoise
                )
            else:
                decoded = _make_bins_sv(
                    self.raw_results, self.hierarchical_coord_matrix, self.bit_depth, self.denoise
                )

            if self.denoise:
                self._bins, self._bias_stats = decoded
            else:
                self._bins = decoded
        return self._bins

    @property
    def bias_stats(self):
        """Returns bias stats if denoising was enabled."""
        if self._bins is None:
            _ = self.bins
        return self._bias_stats

    def reconstruct(self, use_denoising_bias=True):
        """
        Reconstruct the image from simulation results.

        Args:
            use_denoising_bias (bool): If True and denoising was on, apply sibling smoothing.

        Returns:
            np.ndarray: Reconstructed image.
        """
        if self._image is not None:
            return self._image

        # Infer image size from hierarchy matrix
        pos_len = len(self.hierarchical_coord_matrix[0])
        # N = 2^(pos_len/2) assuming d=2
        image_side = int(2 ** (pos_len // 2))
        shape = (image_side, image_side)

        self._image = utils.mhrqi_bins_to_image(
            self.bins,
            self.hierarchical_coord_matrix,
            d=2,
            image_shape=shape,
            bias_stats=self.bias_stats if use_denoising_bias else None,
        )
        return self._image

    def compute_metrics(self, reference_image=None):
        """
        Compute quality metrics (MSE, SSIM, NIQE, etc.).

        Args:
            reference_image (np.ndarray, optional): Ground truth for full-ref metrics.

        Returns:
            dict: Dictionary of metrics.
        """
        from mhrqi.utils import visualization as plots

        recon = self.reconstruct()
        # Scale back to 0-255 for standard metrics if needed,
        # but our compute functions handle float 0-1 too.
        metrics = {"NIQE": plots.compute_niqe(recon), "ENL": plots.compute_enl(recon)}
        if reference_image is not None:
            metrics["MSE"] = plots.compute_mse(reference_image, recon)
            metrics["SSIM"] = plots.compute_ssim(reference_image, recon)

        return metrics

    def plot(self, title="MHRQI Reconstruction", cmap="gray"):
        """Show the reconstructed image in a plot."""
        import matplotlib.pyplot as plt

        recon = self.reconstruct()
        plt.figure(figsize=(6, 6))
        plt.imshow(recon, cmap=cmap)
        plt.title(title)
        plt.axis("off")
        plt.show()


def _make_bins_counts(counts, hierarchical_coord_matrix, bit_depth=8, denoise=False):
    """Internal helper to bin counts."""
    bins = defaultdict(lambda: {"intensity_sum": 0, "count": 0, "intensity_squared_sum": 0})
    outcome_stats = (
        defaultdict(lambda: {"hit": 0, "miss": 0, "intensity_hit": 0, "intensity_miss": 0})
        if denoise
        else None
    )
    pos_len = len(hierarchical_coord_matrix[0])
    for bitstring, count in counts.items():
        b = bitstring[::-1]
        expected_len = pos_len + bit_depth + (1 if denoise else 0)
        if len(b) < expected_len:
            continue
        pos_bits = tuple(int(b[i]) for i in range(pos_len))
        intensity_bits = [int(b[pos_len + i]) for i in range(bit_depth)]
        intensity_value = sum(bit * (2**idx) for idx, bit in enumerate(intensity_bits))
        intensity_normalized = intensity_value / (2**bit_depth - 1)
        bins[pos_bits]["intensity_sum"] += intensity_normalized * count
        bins[pos_bits]["intensity_squared_sum"] += (intensity_normalized**2) * count
        bins[pos_bits]["count"] += count
        if denoise:
            outcome_bit = int(b[pos_len + bit_depth])
            if outcome_bit == 1:
                outcome_stats[pos_bits]["hit"] += count
                outcome_stats[pos_bits]["intensity_hit"] += intensity_normalized * count
            else:
                outcome_stats[pos_bits]["miss"] += count
                outcome_stats[pos_bits]["intensity_miss"] += intensity_normalized * count
    if denoise:
        return bins, outcome_stats
    return bins


def _make_bins_sv(state_vector, hierarchical_coord_matrix, bit_depth=8, denoise=False):
    """Internal helper to bin statevector results."""
    bins = defaultdict(lambda: {"intensity_sum": 0, "count": 0, "intensity_squared_sum": 0})
    outcome_stats = (
        defaultdict(lambda: {"hit": 0, "miss": 0, "intensity_hit": 0, "intensity_miss": 0})
        if denoise
        else None
    )
    pos_len = len(hierarchical_coord_matrix[0])
    sv = np.array(state_vector).flatten()
    outcome_idx = pos_len + bit_depth if denoise else None
    probs = np.abs(sv) ** 2
    nonzero_mask = probs > 1e-10
    nonzero_indices = np.where(nonzero_mask)[0]
    nonzero_probs = probs[nonzero_mask]
    for idx, prb in zip(nonzero_indices, nonzero_probs):
        pos_bits_list = []
        for i in range(pos_len):
            pos_bits_list.append((idx >> i) & 1)
        pos_bits = tuple(pos_bits_list)
        intensity_value = 0
        for i in range(bit_depth):
            if (idx >> (pos_len + i)) & 1:
                intensity_value |= 1 << i
        intensity_normalized = intensity_value / (2**bit_depth - 1)
        bins[pos_bits]["intensity_sum"] += intensity_normalized * prb
        bins[pos_bits]["intensity_squared_sum"] += (intensity_normalized**2) * prb
        bins[pos_bits]["count"] += prb
        if denoise and outcome_idx is not None:
            outcome_bit = (idx >> outcome_idx) & 1
            if outcome_bit == 1:
                outcome_stats[pos_bits]["hit"] += prb
                outcome_stats[pos_bits]["intensity_hit"] += intensity_normalized * prb
            else:
                outcome_stats[pos_bits]["miss"] += prb
                outcome_stats[pos_bits]["intensity_miss"] += intensity_normalized * prb
    if denoise:
        return bins, outcome_stats
    return bins
