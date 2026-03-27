"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  MHRQI - Multi-scale Hierarchical Representation of Quantum Images            ║
║  Classical Denoiser Comparison: BM3D, NL-Means, SRAD                        ║
║                                                                              ║
║  Author: Keno S. Jose                                                        ║
║  License: Apache 2.0                                                         ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import datetime
import os

import cv2
import numpy as np
import srad
from bm3d import BM3DProfile, BM3DStages, bm3d

from mhrqi.utils import visualization as plots

# -----------------------------
# Image domain utilities
# -----------------------------


def to_float01(img):
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    else:
        img = img.astype(np.float32)
        if img.max() > 1.0:
            img /= 255.0
    return np.clip(img, 0.0, 1.0)


def to_uint8(img):
    return (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)


def extract_roi(img, roi):
    y, x, h, w = roi
    H, W = img.shape
    if y < 0 or x < 0 or y + h > H or x + w > W:
        raise ValueError(f"ROI {roi} out of bounds for image {img.shape}")
    return img[y : y + h, x : x + w]


# -----------------------------
# Denoisers (float in, float out)
# -----------------------------


def denoise_bm3d(img, sigma=0.05, stage=BM3DStages.ALL_STAGES):
    out = bm3d(img, sigma_psd=sigma, stage_arg=stage, profile=BM3DProfile())
    return np.clip(out, 0.0, 1.0)


def denoise_nlmeans(img, h=10, template=7, search=21):
    u8 = to_uint8(img)
    out = cv2.fastNlMeansDenoising(u8, None, h, template, search)
    return out.astype(np.float32) / 255.0


def denoise_srad(img, iters=400, dt=0.65, decay=0.8):
    srad_in = (img * 255.0).astype(np.float32) + 1e-5
    out = srad.SRAD(srad_in, iters, dt, decay)
    return np.clip(out / 255.0, 0.0, 1.0)


# -----------------------------
# Automated ROI selection
# -----------------------------


def auto_homogeneous_roi(img, win=20, stride=10):
    """
    Find the most homogeneous window in the image.

    Selects the window with lowest coefficient of variation (CoV),
    with a center-bias penalty to prefer central regions.

    Args:
        img: Grayscale float image in [0, 1].
        win: Window size in pixels.
        stride: Step size for the sliding window search.

    Returns:
        Tuple (y, x, h, w) of the selected ROI.

    Raises:
        RuntimeError: If no valid homogeneous region is found.
    """
    H, W = img.shape
    best_cov = np.inf
    best_roi = None
    eps = 1e-6

    for y in range(0, H - win, stride):
        for x in range(0, W - win, stride):
            patch = img[y : y + win, x : x + win]
            mu = patch.mean()
            if mu < eps or mu > 0.95:
                continue
            sigma = patch.std()
            cov = sigma / mu
            dist_to_center = np.sqrt((y + win / 2 - H / 2) ** 2 + (x + win / 2 - W / 2) ** 2)
            dist_norm = dist_to_center / np.sqrt((H / 2) ** 2 + (W / 2) ** 2)
            cost = cov + 0.1 * dist_norm
            if cost < best_cov:
                best_cov = cost
                best_roi = (y, x, win, win)

    if best_roi is None:
        raise RuntimeError("Failed to find homogeneous ROI")

    return best_roi


# -----------------------------
# Main comparison
# -----------------------------


class BenchmarkSuite:
    """
    Class-based interface for running classical denoiser benchmarks.
    """

    def __init__(self, noisy_image, reference_image=None, save_dir=None):
        self.noisy_image = to_float01(noisy_image)
        self.reference_image = to_float01(reference_image) if reference_image is not None else None
        self.save_dir = save_dir
        self.results = []

        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)

        try:
            self.roi = auto_homogeneous_roi(self.noisy_image)
        except RuntimeError:
            self.roi = (0, 0, self.noisy_image.shape[0], self.noisy_image.shape[1])

    def run(self, methods="all", proposed_image=None):
        """
        Run benchmarks and return results.
        """
        denoisers = {"bm3d": denoise_bm3d, "nlmeans": denoise_nlmeans, "srad": denoise_srad}

        if methods == "all":
            methods_list = list(denoisers.keys())
        elif isinstance(methods, str):
            methods_list = [methods]
        else:
            methods_list = methods

        # Build list of (name, function)
        to_run = [("Original", None)]
        for name in methods_list:
            if name in denoisers:
                to_run.append((name, denoisers[name]))

        if proposed_image is not None:
            to_run.append(("Proposed", None))
            proposed_float = to_float01(proposed_image)
        else:
            proposed_float = None

        self.results = []
        for name, func in to_run:
            if name == "Original":
                res_img = self.noisy_image
            elif name == "Proposed":
                res_img = proposed_float
            else:
                res_img = func(self.noisy_image)

            metrics = self._compute_metrics(res_img)
            self.results.append({"name": name, "metrics": metrics, "image": to_uint8(res_img)})
        return self.results

    def _compute_metrics(self, res_img):
        m = {}
        m["NIQE"] = plots.compute_niqe(res_img)
        m["SSI"] = plots.compute_ssi(self.noisy_image, res_img, self.roi)
        m["SMPI"] = plots.compute_smpi(self.noisy_image, res_img)
        m["ENL"] = plots.compute_enl(res_img, self.roi)
        cnr_result = plots.compute_cnr(res_img)
        m["CNR"] = cnr_result[0]
        omqdi_result = OMQDI(self.noisy_image, res_img)
        m["OMQDI"] = omqdi_result[0]
        m["EPF"] = omqdi_result[1]
        m["NSF"] = omqdi_result[2]
        m["EPI"] = plots.compute_epi(self.noisy_image, res_img)

        if self.reference_image is not None:
            m["FSIM"] = plots.compute_fsim(self.reference_image, res_img)
            m["SSIM"] = plots.compute_ssim(self.reference_image, res_img)
        else:
            m["FSIM"] = np.nan
            m["SSIM"] = np.nan
        return m

    def save_reports(self, prefix="report"):
        """Save comparison reports and images."""
        if not self.save_dir:
            raise ValueError("save_dir must be set to save reports")

        for res in self.results:
            cv2.imwrite(os.path.join(self.save_dir, f"{prefix}_{res['name']}.png"), res["image"])

        # Split results for report generation
        display_results = [r for r in self.results if r["name"] != "Original"]

        if self.reference_image is not None:
            plots.MetricsPlotter.save_summary_report(
                to_uint8(self.reference_image),
                display_results,
                ["FSIM", "SSIM"],
                "Full Reference Metrics",
                f"{prefix}_full_ref",
                self.save_dir,
            )

        plots.MetricsPlotter.save_summary_report(
            to_uint8(self.noisy_image),
            display_results,
            ["SSI", "SMPI", "NSF", "ENL", "CNR"],
            "Speckle Reduction Metrics",
            f"{prefix}_speckle",
            self.save_dir,
        )

        plots.MetricsPlotter.save_summary_report(
            to_uint8(self.noisy_image),
            display_results,
            ["EPF", "EPI", "OMQDI"],
            "Structural Similarity Metrics",
            f"{prefix}_structural",
            self.save_dir,
        )


def compare_to(image_input, **kwargs):
    """Legacy wrapper for BenchmarkSuite."""
    suite = BenchmarkSuite(
        image_input, reference_image=kwargs.get("reference_image"), save_dir=kwargs.get("save_dir")
    )
    results = suite.run(
        methods=kwargs.get("methods", "all"), proposed_image=kwargs.get("proposed_img")
    )
    if kwargs.get("save", True):
        suite.save_reports(prefix=kwargs.get("save_prefix", "denoised"))
    return results


# -----------------------------
# OMQDI
# -----------------------------

import pywt
from scipy.ndimage import convolve


def getCDF97(weight=1):
    """
    Return the CDF 9/7 biorthogonal wavelet used by OMQDI.

    Args:
        weight: Optional scalar weight applied to all filter coefficients.

    Returns:
        pywt.Wavelet object configured with CDF 9/7 filter banks.
    """
    analysis_LP = np.array(
        [
            0,
            0.026748757411,
            -0.016864118443,
            -0.078223266529,
            0.266864118443,
            0.602949018236,
            0.266864118443,
            -0.078223266529,
            -0.016864118443,
            0.026748757411,
        ]
    )
    analysis_LP *= weight

    analysis_HP = np.array(
        [
            0,
            0.091271763114,
            -0.057543526229,
            -0.591271763114,
            1.11508705,
            -0.591271763114,
            -0.057543526229,
            0.091271763114,
            0,
            0,
        ]
    )
    analysis_HP *= weight

    synthesis_LP = np.array(
        [
            0,
            -0.091271763114,
            -0.057543526229,
            0.591271763114,
            1.11508705,
            0.591271763114,
            -0.057543526229,
            -0.091271763114,
            0,
            0,
        ]
    )
    synthesis_LP *= weight

    synthesis_HP = np.array(
        [
            0,
            0.026748757411,
            0.016864118443,
            -0.078223266529,
            -0.266864118443,
            0.602949018236,
            -0.266864118443,
            -0.078223266529,
            0.016864118443,
            0.026748757411,
        ]
    )
    synthesis_HP *= weight

    return pywt.Wavelet("CDF97", [analysis_LP, analysis_HP, synthesis_LP, synthesis_HP])


def sbEn(coeffs: np.array) -> float:
    """
    Compute the energy of a wavelet sub-band.

    Args:
        coeffs: 2D array of wavelet coefficients for the sub-band.

    Returns:
        Sub-band energy as a scalar.
    """
    rows, cols = coeffs.shape
    return np.log(1 + np.sum(np.square(coeffs)) / (rows * cols))


def En(lvlCoeffs: tuple, alpha=0.8) -> float:
    """
    Compute the weighted energy for one level of wavelet decomposition.

    Args:
        lvlCoeffs: Tuple of (LHn, HLn, HHn) coefficient arrays.
        alpha: Weight for the HH sub-band relative to LH and HL.
               Recommended value 0.8 per ISBN 0139353224.

    Returns:
        Weighted energy scalar.
    """
    LHn, HLn, HHn = lvlCoeffs
    return (1 - alpha) * (sbEn(LHn) + sbEn(HLn)) / 2 + alpha * sbEn(HHn)


def S(decompCoeffs: list, alpha=0.8) -> float:
    """
    Compute the cumulative multi-level wavelet energy.

    Args:
        decompCoeffs: List of (LHn, HLn, HHn) tuples for each decomposition level,
                      ordered from coarsest to finest.
        alpha: Weight for the HH sub-band. See En().

    Returns:
        Cumulative energy scalar.
    """
    energy = 0
    for i, lvlCoeffs in enumerate(decompCoeffs):
        n = i + 1
        energy += 2 ** (3 - n) * En(lvlCoeffs, alpha)
    return energy


def local_mean(img: np.array, window=3, pad_mode="reflect") -> np.array:
    """
    Compute the local mean of pixel intensities using a uniform kernel.

    Args:
        img: Input image of shape (M, N).
        window: Kernel size.
        pad_mode: Padding mode for convolution.

    Returns:
        Local mean array of shape (M, N).
    """
    return convolve(img, np.full((window, window), 1 / window**2), mode=pad_mode)


def local_variance(img: np.array) -> np.array:
    """
    Compute the local variance of an image.

    Args:
        img: Input image of shape (M, N).

    Returns:
        Local variance array of shape (M, N).
    """
    mu_sq = np.square(local_mean(img))
    return local_mean(np.square(img)) - mu_sq


def noise_power(img: np.array) -> float:
    """
    Estimate the noise power (σ̂) of an image as the mean local variance.

    Args:
        img: Input image of shape (M, N).

    Returns:
        Estimated noise power scalar.
    """
    return np.mean(local_variance(img))


def OMQDI(X: np.array, Y: np.array, C=1e-10) -> tuple:
    """
    Compute the Objective Measure of Quality of Denoised Images.

    Reference: DOI 10.1016/j.bspc.2021.102962

    Args:
        X: Noisy input image of shape (M, N).
        Y: Denoised output image of shape (M, N).
        C: Small constant to avoid division by zero.

    Returns:
        Tuple (OMQDI, Q1, Q2):
            - OMQDI: Combined metric Q1 + Q2, ideal value 2, range [1, 2].
            - Q1: Edge-Preservation Factor, ideal value 1, range [0, 1].
            - Q2: Noise-Suppression Factor, ideal value 1, range [0, 1].
    """
    CDF97 = getCDF97()

    coeffX = pywt.wavedec2(X, CDF97, level=3)
    coeffY = pywt.wavedec2(Y, CDF97, level=3)

    SX = S(coeffX[1:])
    SY = S(coeffY[1:])

    npX = noise_power(X)
    npY = noise_power(Y)

    Q1 = (2 * SX * SY + C) / (SX**2 + SY**2 + C)
    Q2 = (npX - npY) ** 2 / (npX**2 + npY**2 + C)
    return (Q1 + Q2, Q1, Q2)


# -----------------------------
# Entry point
# -----------------------------

if __name__ == "__main__":
    n = 256
    img_path = os.path.join("resources", "drusen1.jpeg")
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (n, n))

    proposed = cv2.imread("testme.png", cv2.IMREAD_GRAYSCALE)
    if proposed is not None:
        proposed = cv2.resize(proposed, (n, n))

    compare_to(
        img, proposed_img=proposed, methods="all", plot=True, save=True, save_prefix="denoised"
    )
