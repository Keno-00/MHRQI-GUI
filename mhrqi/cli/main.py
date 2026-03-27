"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  MHRQI - Multi-scale Hierarchical Representation of Quantum Images            ║
║  Main Pipeline: Encoding, Denoising, Benchmarking                           ║
║                                                                              ║
║  Author: Keno S. Jose                                                        ║
║  License: Apache 2.0                                                         ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import argparse
import csv
import math
import os
import time
from datetime import datetime
from pathlib import Path

# Use a non-interactive backend for matplotlib to allow drawing in workers
os.environ.setdefault("MPLBACKEND", "Agg")
import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import base64

from mhrqi.benchmarks import BenchmarkSuite
from mhrqi.core.representation import MHRQI
from mhrqi.utils import general as utils
from mhrqi.utils import visualization as plots

CSV_PATH = Path("mhrqi_runs.csv")


def save_rows_to_csv(rows, csv_path=CSV_PATH):
    fieldnames = ["timestamp", "n", "bins", "shots", "shots_per_bin", "mse"]
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerows(rows)


def main(
    shots=1000,
    n=4,
    d=2,
    denoise=False,
    use_shots=True,
    fast=False,
    verbose_plots=False,
    use_gpu=False,
    img_path=None,
    run_comparison=True,
    bit_depth=8,
    return_diagnostics=False,
):
    """
    Main MHRQI simulation pipeline.

    Args:
        shots: number of measurement shots (if use_shots=True)
        n: image dimension (will be resized to n x n)
        d: qudit dimension (2=qubit)
        denoise: whether to apply denoising circuit
        use_shots: if True, use shot-based simulation; if False, use statevector
        fast: if True, use lazy (statevector-based) upload for speed
        verbose_plots: if True, show additional debug plots
        use_gpu: if True, prefer GPU-enabled simulation backends
        img_path: path to input image (defaults to resources/drusen1.jpeg)
        run_comparison: if True, run comparison benchmarks against BM3D/NL-Means/SRAD
        bit_depth: bits used for intensity encoding

    Returns:
        tuple: (original_image, reconstructed_image, run_directory_path)
        when return_diagnostics=True, returns
        (original_image, reconstructed_image, run_directory_path, diagnostics_dict)
    """

    # Use default image if not specified
    if img_path is None:
        # Look for resources folder in the project root (3 levels up from mhrqi/cli/main.py)
        root_dir = Path(__file__).resolve().parent.parent.parent
        img_path = str(root_dir / "resources" / "drusen1.jpeg")

    myimg = cv2.imread(img_path)
    myimg = cv2.resize(myimg, (n, n))

    myimg = cv2.cvtColor(myimg, cv2.COLOR_RGB2GRAY)
    N = myimg.shape[1]
    angle_norm = utils.angle_map(myimg)
    normalized_img = np.clip(myimg.astype(np.float64) / 255.0, 0.0, 1.0)

    H, W = angle_norm.shape
    max_depth = utils.get_max_depth(N, d)
    hierarchical_coord_matrix = utils.generate_hierarchical_coord_matrix(N, d)

    # -------------------------
    # Circuit Construction & Upload
    # -------------------------
    model = MHRQI(max_depth, bit_depth=bit_depth)
    if fast:
        model.lazy_upload(hierarchical_coord_matrix, normalized_img)
    else:
        model.upload(hierarchical_coord_matrix, normalized_img)

    # -------------------------
    # Denoising (Extension)
    # -------------------------
    if denoise:
        model.apply_denoising()

    # Simulate
    start_time = time.perf_counter()
    result = model.simulate(shots=shots if use_shots else None, use_gpu=use_gpu)
    end_time = time.perf_counter()

    # -------------------------
    # Reconstruction
    # -------------------------
    newimg = result.reconstruct(use_denoising_bias=denoise)
    newimg = (np.clip(newimg, 0.0, 1.0) * 255).astype(np.uint8)

    # -------------------------
    # Create run directory
    # -------------------------
    run_dir = plots.get_run_dir()

    # Save bias_stats for plotting if needed
    bias_stats = result.bias_stats
    bias_map = None

    # -------------------------
    # Verbose Plots (Diagnostics)
    # -------------------------
    if verbose_plots:
        plots.plot_mse_map(myimg, newimg, run_dir=run_dir)
        if denoise:
            bias_map = plots.plot_bias_map(bias_stats, normalized_img, N, d, run_dir=run_dir)

    # Save settings
    settings = {
        "Image": os.path.basename(img_path) if img_path else "drusen1.jpeg",
        "Size": f"{n}x{n}",
        "Backend": "MHRQI (Qiskit)",
        "Fast Mode": fast,
        "Denoise": denoise,
        "Use Shots": use_shots,
        "Shots": shots if use_shots else "N/A (statevector)",
        "d (qudit dim)": d,
        "Bit Depth": bit_depth,
        "Use GPU": use_gpu,
        "Simulation Time": f"{end_time - start_time:.2f}s",
    }
    plots.save_settings_plot(settings, run_dir)

    # Get a clean image name from path
    img_name = os.path.splitext(os.path.basename(img_path or "drusen1.jpeg"))[0]
    plots.show_image_comparison(myimg, newimg, run_dir=run_dir, img_name=img_name)

    # Try to save a visual representation of the quantum circuit (matplotlib)
    circuit_path = None
    try:
        try:
            out = model.circuit.draw(output="mpl", fold=100)
        except TypeError:
            out = model.circuit.draw(output="mpl")

        # out may be a matplotlib.figure.Figure or an Axes; normalize to Figure
        fig = None
        try:
            # If it's already a Figure
            import matplotlib.figure as _mf

            if isinstance(out, _mf.Figure):
                fig = out
            elif hasattr(out, "figure"):
                fig = out.figure
            elif hasattr(out, "get_figure"):
                fig = out.get_figure()
        except Exception:
            fig = None

        if fig is not None:
            circuit_path = os.path.abspath(os.path.join(run_dir, "circuit.png"))
            fig.savefig(circuit_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
        else:
            # Fallback: try a text rendering of the circuit into a matplotlib figure
            try:
                text_repr = model.circuit.draw(output="text")
                lines = text_repr.splitlines()
                n_lines = max(1, len(lines))
                max_cols = max((len(l) for l in lines), default=80)
                # Estimate figure size: width in inches proportional to cols, height to lines
                dpi = 150
                fig_w = max(6, max_cols / 12)
                fig_h = max(2, n_lines / 6)
                fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
                fig.text(0.0, 0.99, text_repr, family="monospace", fontsize=8, va="top")
                plt.axis("off")
                circuit_path = os.path.abspath(os.path.join(run_dir, "circuit.png"))
                fig.savefig(circuit_path, dpi=dpi, bbox_inches="tight")
                plt.close(fig)
            except Exception:
                circuit_path = None
    except Exception:
        circuit_path = None

    diagnostics = {
        "mse_map": (myimg.astype(np.float32) - newimg.astype(np.float32)) ** 2,
        "bias_map": bias_map,
        "mpl_paths": {
            "comparison": os.path.abspath(os.path.join(run_dir, f"{img_name}_comparison.png")),
            "mse": os.path.abspath(os.path.join(run_dir, "mse_map.png")),
            "bias": os.path.abspath(os.path.join(run_dir, "bias_map.png")),
            **({"circuit": circuit_path} if circuit_path else {}),
        },
    }

    # If we saved a circuit image, also include it as base64 so UIs can load it directly
    if circuit_path and os.path.isfile(circuit_path):
        try:
            with open(circuit_path, "rb") as f:
                b = f.read()
            diagnostics["circuit_b64"] = base64.b64encode(b).decode("ascii")
        except Exception:
            pass

    # -------------------------
    # Run comparison benchmarks
    # -------------------------
    if run_comparison:
        evals_dir = os.path.join(run_dir, "evals")
        print(f"Running benchmarks... saving to {evals_dir}")

        suite = BenchmarkSuite(myimg, save_dir=evals_dir)
        suite.run(methods="all", proposed_image=newimg)
        suite.save_reports(prefix="comp")
        # Expose saved benchmark MPL plots to the caller via diagnostics paths.
        try:
            for fname in os.listdir(evals_dir):
                if fname.lower().endswith(".png") and fname.startswith("comp_"):
                    key = os.path.splitext(fname)[0]  # e.g., comp_speckle
                    diagnostics["mpl_paths"][key] = os.path.abspath(os.path.join(evals_dir, fname))
        except Exception:
            # If anything goes wrong (e.g., directory not present), skip bench plotting exposures.
            pass
        # Also expose the numeric benchmark results for GUI plotting (DPG)
        try:
            diagnostics["bench_results"] = []
            for r in suite.results:
                metrics_clean = {}
                for k, v in r["metrics"].items():
                    try:
                        metrics_clean[k] = float(v)
                    except Exception:
                        metrics_clean[k] = float("nan")
                diagnostics["bench_results"].append({"name": r["name"], "metrics": metrics_clean})
        except Exception:
            pass

    if return_diagnostics:
        return myimg, newimg, run_dir, diagnostics

    return myimg, newimg, run_dir


def main_cli():
    parser = argparse.ArgumentParser(
        description="MHRQI - Multi-scale Hierarchical Representation of Quantum Images"
    )
    parser.add_argument("--shots", type=int, default=1000, help="Number of measurement shots")
    parser.add_argument("-n", "--size", type=int, default=4, help="Image size (n x n)")
    parser.add_argument(
        "-d", "--dimension", type=int, default=2, help="Qudit dimension (default 2 for qubits)"
    )
    parser.add_argument(
        "--bit-depth", type=int, default=8, help="Intensity encoding bit depth"
    )
    parser.add_argument("--denoise", action="store_true", help="Apply denoising circuit")
    parser.add_argument(
        "--statevector", action="store_true", help="Use statevector simulation instead of shots"
    )
    parser.add_argument("--fast", action="store_true", help="Use lazy upload for speed")
    parser.add_argument("--verbose", action="store_true", help="Show additional debug plots")
    parser.add_argument("--cpu", action="store_true", help="Force CPU simulation backend")
    parser.add_argument("--img", type=str, help="Path to input image")
    parser.add_argument(
        "--no-comparison",
        action="store_false",
        dest="comparison",
        help="Skip comparison benchmarks",
    )
    parser.set_defaults(comparison=True)

    args = parser.parse_args()

    main(
        shots=args.shots,
        n=args.size,
        d=args.dimension,
        bit_depth=args.bit_depth,
        denoise=args.denoise,
        use_shots=not args.statevector,
        fast=args.fast,
        verbose_plots=args.verbose,
        use_gpu=not args.cpu,
        img_path=args.img,
        run_comparison=args.comparison,
    )


if __name__ == "__main__":
    main_cli()
