import os
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional

# Force non-interactive backend before importing pipeline module,
# so any matplotlib usage in processing does not create Tk windows.
os.environ.setdefault("MPLBACKEND", "Agg")

import dearpygui.dearpygui as dpg
from matplotlib import colormaps
import numpy as np
import cv2

from mhrqi.cli.main import main as run_pipeline


TEXTURE_REGISTRY_TAG = "texture_registry"
ORIG_TEXTURE_TAG = "orig_texture"
RECON_TEXTURE_TAG = "recon_texture"
MSE_TEXTURE_TAG = "mse_texture"
BIAS_TEXTURE_TAG = "bias_texture"
CIRCUIT_TEXTURE_TAG = "circuit_texture"
CIRCUIT_DRAWLIST_TAG = "circuit_drawlist"
CIRCUIT_DRAW_IMAGE_TAG = "circuit_draw_image"
CIRCUIT_VIEWPORT_RECT_TAG = "circuit_viewport_rect"
CIRCUIT_HANDLER_TAG = "circuit_handler"
PREVIEW_CHILD_TAG = "preview_child"
BENCH_FULL_REF_TEXTURE_TAG = "bench_full_ref_texture"
BENCH_SPECKLE_TEXTURE_TAG = "bench_speckle_texture"
BENCH_STRUCTURAL_TEXTURE_TAG = "bench_structural_texture"
IMAGE_FILE_TYPES = (
    ("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
    ("PNG", "*.png"),
    ("JPEG", "*.jpg *.jpeg"),
    ("Bitmap", "*.bmp"),
    ("TIFF", "*.tif *.tiff"),
    ("All files", "*.*"),
)


@dataclass
class GUIConfig:
    img_path: str
    size: int
    dimension: int
    bit_depth: int
    use_shots: bool
    shots: int
    denoise: bool
    fast: bool
    verbose_plots: bool
    run_comparison: bool
    use_gpu: bool


class GUIState:
    def __init__(self):
        self.executor: Optional[ProcessPoolExecutor] = None
        self.pending_future: Optional[Future[dict[str, Any]]] = None
        self.run_directory: Optional[str] = None
        self.is_running = False
        self.texture_counter = 0
        self.current_orig_texture: Optional[str] = None
        self.current_recon_texture: Optional[str] = None
        self.current_mse_texture: Optional[str] = None
        self.current_bias_texture: Optional[str] = None
        self.current_bench_full_ref_texture: Optional[str] = None
        self.current_bench_speckle_texture: Optional[str] = None
        self.current_bench_structural_texture: Optional[str] = None
        self.mpl_plot_paths: dict[str, str] = {}


STATE = GUIState()


def _status(message):
    dpg.set_value("status_text", message)


def _is_power_of_two(value):
    return value > 0 and (value & (value - 1)) == 0


def _get_config_from_ui():
    return GUIConfig(
        img_path=dpg.get_value("img_path").strip(),
        size=max(2, int(dpg.get_value("size"))),
        dimension=max(2, int(dpg.get_value("dimension"))),
        bit_depth=max(1, int(dpg.get_value("bit_depth"))),
        use_shots=bool(dpg.get_value("use_shots")),
        shots=max(1, int(dpg.get_value("shots"))),
        denoise=bool(dpg.get_value("denoise")),
        fast=bool(dpg.get_value("fast")),
        verbose_plots=bool(dpg.get_value("verbose_plots")),
        run_comparison=bool(dpg.get_value("run_comparison")),
        use_gpu=bool(dpg.get_value("use_gpu")),
    )


def _grayscale_to_rgba_texture(image, target_w=None, target_h=None):
    array = np.asarray(image)
    if array.ndim == 3:
        # Convert to grayscale by averaging channels.
        array = array.mean(axis=2)

    if array.dtype == np.uint8:
        array = array.astype(np.float32) / 255.0
    else:
        array = np.clip(array.astype(np.float32), 0.0, 1.0)

    height, width = array.shape
    if target_w is not None and target_h is not None and (width != target_w or height != target_h):
        array = _nearest_neighbor_resize(array, target_w, target_h)
        height, width = array.shape

    rgba = np.empty((height, width, 4), dtype=np.float32)
    rgba[..., 0] = array
    rgba[..., 1] = array
    rgba[..., 2] = array
    rgba[..., 3] = 1.0
    return width, height, rgba.flatten().tolist()


def _nearest_neighbor_resize(arr, target_w, target_h):
    """Nearest-neighbor resize for 2D arrays (or 3D where channels are last)."""
    arr = np.asarray(arr)
    src_h, src_w = arr.shape[:2]
    if src_h == target_h and src_w == target_w:
        return arr
    row_idx = (np.arange(target_h) * src_h // target_h).astype(int)
    col_idx = (np.arange(target_w) * src_w // target_w).astype(int)
    # Use np.ix_ to index first two dims and preserve channels if present.
    return arr[np.ix_(row_idx, col_idx)]


def _scalar_to_colormap_rgba_texture(image, cmap_name, target_w=None, target_h=None):
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr.mean(axis=2)

    if arr.size == 0:
        arr = np.zeros((1, 1), dtype=np.float32)

    finite = np.isfinite(arr)
    if not np.any(finite):
        arr = np.zeros_like(arr)
        vmin, vmax = 0.0, 1.0
    else:
        vmin = float(np.nanmin(arr))
        vmax = float(np.nanmax(arr))
        if vmax <= vmin:
            vmax = vmin + 1.0

    normalized = np.clip(np.nan_to_num((arr - vmin) / (vmax - vmin)), 0.0, 1.0)

    if target_w is not None and target_h is not None:
        normalized = _nearest_neighbor_resize(normalized, target_w, target_h)

    cmap = colormaps[cmap_name]
    rgba = cmap(normalized).astype(np.float32)

    height, width = normalized.shape
    return width, height, rgba.flatten().tolist(), vmin, vmax


def _set_texture(texture_tag, image, target_w=None, target_h=None):
    width, height, data = _grayscale_to_rgba_texture(image, target_w, target_h)
    dpg.add_static_texture(width, height, data, tag=texture_tag, parent=TEXTURE_REGISTRY_TAG)


def _set_colormap_texture(texture_tag, image, cmap_name, target_w=None, target_h=None):
    width, height, data, vmin, vmax = _scalar_to_colormap_rgba_texture(image, cmap_name, target_w, target_h)
    dpg.add_static_texture(width, height, data, tag=texture_tag, parent=TEXTURE_REGISTRY_TAG)
    return vmin, vmax


def _set_color_texture(texture_tag, image, target_w=None, target_h=None):
    """Create an RGBA texture from a color (RGB/BGR) or grayscale image and register it."""
    arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[2] == 3:
        # cv2.imread -> BGR; convert to RGB
        arr = arr[..., ::-1]

    if arr.dtype == np.uint8:
        arr = arr.astype(np.float32) / 255.0
    else:
        arr = np.clip(arr.astype(np.float32), 0.0, 1.0)

    height, width = arr.shape[:2]
    if target_w is not None and target_h is not None and (width != target_w or height != target_h):
        arr = _nearest_neighbor_resize(arr, target_w, target_h)
        height, width = arr.shape[:2]

    rgba = np.empty((height, width, 4), dtype=np.float32)
    if arr.ndim == 2:
        rgba[..., 0] = arr
        rgba[..., 1] = arr
        rgba[..., 2] = arr
    else:
        rgba[..., 0:3] = arr[..., 0:3]
    rgba[..., 3] = 1.0

    dpg.add_static_texture(width, height, rgba.flatten().tolist(), tag=texture_tag, parent=TEXTURE_REGISTRY_TAG)


def _load_and_set_color_texture_from_path(texture_prefix, state_attr_name, image_tag, file_path, target_w=720, target_h=300):
    """Load an image from disk and create/update a texture for a dpg image item."""
    if not file_path or not os.path.isfile(file_path):
        return False
    img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return False

    new_tag = _next_texture_tag(texture_prefix)
    _set_color_texture(new_tag, img, target_w=target_w, target_h=target_h)
    dpg.configure_item(image_tag, texture_tag=new_tag, width=target_w, height=target_h)

    current_tag = getattr(STATE, state_attr_name)
    if current_tag and dpg.does_item_exist(current_tag):
        dpg.delete_item(current_tag)
    setattr(STATE, state_attr_name, new_tag)
    return True


def _next_texture_tag(prefix):
    STATE.texture_counter += 1
    return f"{prefix}_{STATE.texture_counter}"


def _set_preview_images(original, reconstructed):
    new_orig_tag = _next_texture_tag("orig_texture")
    new_recon_tag = _next_texture_tag("recon_texture")

    _set_texture(new_orig_tag, original, target_w=360, target_h=360)
    _set_texture(new_recon_tag, reconstructed, target_w=360, target_h=360)

    dpg.configure_item("orig_image", texture_tag=new_orig_tag, width=360, height=360)
    dpg.configure_item("recon_image", texture_tag=new_recon_tag, width=360, height=360)

    if STATE.current_orig_texture and dpg.does_item_exist(STATE.current_orig_texture):
        dpg.delete_item(STATE.current_orig_texture)
    if STATE.current_recon_texture and dpg.does_item_exist(STATE.current_recon_texture):
        dpg.delete_item(STATE.current_recon_texture)

    STATE.current_orig_texture = new_orig_tag
    STATE.current_recon_texture = new_recon_tag


def _set_diagnostic_map(image, prefix, image_tag, state_attr_name, cmap_name):
    if image is None:
        return None

    new_tag = _next_texture_tag(prefix)
    vmin, vmax = _set_colormap_texture(new_tag, image, cmap_name, target_w=360, target_h=360)
    dpg.configure_item(image_tag, texture_tag=new_tag, width=360, height=360)

    current_tag = getattr(STATE, state_attr_name)
    if current_tag and dpg.does_item_exist(current_tag):
        dpg.delete_item(current_tag)

    setattr(STATE, state_attr_name, new_tag)
    return vmin, vmax


def _update_diagnostic_plots(original, reconstructed, diagnostics, verbose_enabled, denoise_enabled):
    # MSE map mirrors mhrqi.utils.visualization.plot_mse_map concept.
    mse_map = diagnostics.get("mse_map") if diagnostics else None
    if mse_map is None:
        diff = original.astype(np.float32) - reconstructed.astype(np.float32)
        mse_map = diff * diff

    mse_range = _set_diagnostic_map(
        mse_map,
        prefix="mse_texture",
        image_tag="mse_image",
        state_attr_name="current_mse_texture",
        cmap_name="RdYlGn_r",
    )
    if mse_range is None:
        dpg.set_value("mse_plot_stats", "No MSE map available.")
    else:
        dpg.set_value(
            "mse_plot_stats",
            f"Squared error range: {mse_range[0]:.4f} to {mse_range[1]:.4f}",
        )

    show_diag_panel = bool(verbose_enabled)
    dpg.configure_item("diag_panel", show=show_diag_panel)

    bias_map = diagnostics.get("bias_map") if diagnostics else None
    has_bias = show_diag_panel and denoise_enabled and bias_map is not None
    dpg.configure_item("bias_plot_group", show=has_bias)
    if has_bias:
        bias_range = _set_diagnostic_map(
            bias_map,
            prefix="bias_texture",
            image_tag="bias_image",
            state_attr_name="current_bias_texture",
            cmap_name="viridis",
        )
        if bias_range is not None:
            dpg.set_value(
                "bias_plot_stats",
                f"Hit-ratio range: {bias_range[0]:.4f} to {bias_range[1]:.4f}",
            )
    else:
        dpg.set_value("bias_plot_stats", "Bias map available for denoise + verbose runs.")

    dpg.configure_item("open_mpl_mse_button", enabled=show_diag_panel)
    dpg.configure_item("open_mpl_bias_button", enabled=has_bias)


def _display_label(name: str) -> str:
    n = str(name).lower()
    if n == "bm3d":
        return "BM3D"
    if "nl" in n:
        return "NL-Means"
    if "srad" in n:
        return "SRAD"
    if "proposed" in n or "mhrqi" in n:
        return "MHRQI (Ours)"
    return str(name)


def _clear_children(item_tag: str):
    try:
        children = dpg.get_item_children(item_tag, 1) or []
        for c in children:
            try:
                dpg.delete_item(c)
            except Exception:
                pass
    except Exception:
        pass


def _format_metrics_table_text(methods, metric_keys, results_map):
    # Build a simple monospaced text table
    col_widths = [max(6, len(m)) for m in ["Method"] + metric_keys]
    for i, k in enumerate(metric_keys, start=1):
        col_widths[i] = max(col_widths[i], len(k))

    lines = []
    header = "".join(f"{h:<{w}}  " for h, w in zip(["Method"] + metric_keys, col_widths))
    lines.append(header)
    lines.append("-" * len(header))
    for name in methods:
        row = [ _display_label(name) ]
        for k in metric_keys:
            v = results_map.get(name, {}).get(k, float("nan"))
            if v is None or (isinstance(v, float) and np.isnan(v)):
                row.append("N/A")
            else:
                try:
                    row.append(f"{float(v):.4f}")
                except Exception:
                    row.append(str(v))
        lines.append("".join(f"{c:<{w}}  " for c, w in zip(row, col_widths)))
    return "\n".join(lines)


def _update_benchmark_dpg(bench_results: list):
    if not bench_results:
        dpg.configure_item("bench_panel", show=False)
        return

    # Exclude Original from the displayed competitors
    display_results = [r for r in bench_results if r.get("name") != "Original"]
    if not display_results:
        dpg.configure_item("bench_panel", show=False)
        return

    methods = [r["name"] for r in display_results]
    method_labels = [_display_label(m) for m in methods]

    # Build a quick lookup map for metrics
    results_map = {r["name"]: r.get("metrics", {}) for r in display_results}

    # We display one DPG-native plot per metric (each plot compares methods
    # for that single metric). This avoids combining multiple metrics into
    # a single confusing plot.
    groups = [
        ("bench_speckle", "bench_speckle_table_text", ["SSI", "SMPI", "NSF", "ENL", "CNR"]),
        ("bench_structural", "bench_structural_table_text", ["EPF", "EPI", "OMQDI"]),
    ]

    for group_prefix, text_tag, metric_keys in groups:
        # For each metric create/populate a dedicated plot (axes were
        # pre-created in the UI build step with predictable tags).
        for metric in metric_keys:
            xaxis_tag = f"{group_prefix}_xaxis_{metric}"
            yaxis_tag = f"{group_prefix}_yaxis_{metric}"

            # Clear existing series under this y-axis
            try:
                existing = dpg.get_item_children(yaxis_tag, 1) or []
                for child in existing:
                    try:
                        dpg.delete_item(child)
                    except Exception:
                        pass
            except Exception:
                pass


            # Build one bar series per method so each method gets its own
            # colored bar and legend entry. Also collect numeric values for
            # compact display and for determining the top-ranked method.
            values_text_parts = []
            vals = []
            for i, name in enumerate(methods):
                v = results_map.get(name, {}).get(metric, float("nan"))
                try:
                    if v is None or (isinstance(v, float) and np.isnan(v)):
                        val = 0.0
                    else:
                        val = float(v)
                except Exception:
                    val = 0.0

                vals.append(val)
                try:
                    dpg.add_bar_series([i], [val], label=_display_label(name), parent=yaxis_tag)
                except Exception:
                    pass

                values_text_parts.append(f"{_display_label(name)}: {val:.3f}")

            # Update axis ticks for this metric plot
            try:
                ticks = [(i, label) for i, label in enumerate(method_labels)]
                dpg.set_axis_ticks(xaxis_tag, ticks)
            except Exception:
                pass

            # Update the compact values text below the plot for quick reading
            try:
                values_tag = f"{group_prefix}_values_{metric}"
                if dpg.does_item_exist(values_tag):
                    dpg.set_value(values_tag, " | ".join(values_text_parts))
            except Exception:
                pass

            # Compute and display the top-ranked method for this metric
            try:
                if vals:
                    max_idx = int(np.argmax(vals))
                    max_val = vals[max_idx]
                    top_tag = f"{group_prefix}_top_{metric}"
                    if dpg.does_item_exist(top_tag):
                        dpg.set_value(top_tag, f"Top: {method_labels[max_idx]} ({max_val:.4f})")
            except Exception:
                pass

        # Recreate DPG table for this group (one table per metric group)
        try:
            if text_tag == "bench_full_ref_table_text":
                parent = "bench_full_ref_group"
                before_img = "bench_full_ref_image"
                table_tag = "bench_full_ref_table"
            elif text_tag == "bench_speckle_table_text":
                parent = "bench_speckle_group"
                before_img = "bench_speckle_image"
                table_tag = "bench_speckle_table"
            else:
                parent = "bench_structural_group"
                before_img = "bench_structural_image"
                table_tag = "bench_structural_table"

            cols = ["Method"] + metric_keys
            rows = []
            for name in methods:
                row = {"Method": name}
                for k in metric_keys:
                    row[k] = results_map.get(name, {}).get(k, float("nan"))
                rows.append(row)

            try:
                if dpg.does_item_exist(table_tag):
                    dpg.delete_item(table_tag)
            except Exception:
                pass

            try:
                tbl = dpg.add_table(header_row=True, tag=table_tag, parent=parent, before=before_img)
                for c in cols:
                    dpg.add_table_column(label=c, parent=tbl)
                for r in rows:
                    row_id = dpg.add_table_row(parent=tbl)
                    for c in cols:
                        val = r.get(c, "")
                        if c == "Method":
                            val = _display_label(val)
                        dpg.add_text(str(val), parent=row_id)
            except Exception:
                pass
        except Exception:
            pass

    dpg.configure_item("bench_panel", show=True)


def _add_static_texture_from_array(texture_tag, image, target_w=None, target_h=None):
    """Add a static texture from a numpy image array and return (width, height)."""
    if image is None:
        return 0, 0
    arr = np.asarray(image)
    h, w = arr.shape[:2]
    final_w, final_h = (target_w or w, target_h or h)
    _set_color_texture(texture_tag, arr, target_w=final_w, target_h=final_h)
    return final_w, final_h


def _on_ui_config_change(sender, app_data, user_data=None):
    """Callback attached to UI controls. Circuit preview has been disabled."""
    # No-op: circuit preview removed per user request.
    return


# Circuit preview build functions removed per user request.
# Previously worker functions for building and scheduling circuit preview image
# were implemented here; they have been intentionally removed to revert the
# application to a state without the circuit preview feature.



def _set_controls_enabled(enabled):
    dpg.configure_item("run_button", enabled=enabled)
    dpg.configure_item("open_output_button", enabled=enabled and bool(STATE.run_directory))


def _pick_image_file_native(current_path):
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as exc:
        raise RuntimeError("Native file dialog is unavailable because tkinter is not installed.") from exc

    initialdir = None
    initialfile = None
    current = Path(current_path).expanduser() if current_path else None
    if current:
        if current.is_file():
            initialdir = str(current.parent)
            initialfile = current.name
        elif current.is_dir():
            initialdir = str(current)

    root = tk.Tk()
    root.withdraw()
    try:
        root.attributes("-topmost", True)
    except Exception:
        pass

    try:
        return filedialog.askopenfilename(
            title="Select Input Image",
            filetypes=IMAGE_FILE_TYPES,
            initialdir=initialdir,
            initialfile=initialfile,
        )
    finally:
        root.destroy()


def _show_file_dialog(sender=None, app_data=None, user_data=None):
    current_path = dpg.get_value("img_path")
    try:
        file_path = _pick_image_file_native(current_path)
    except RuntimeError as exc:
        _status(str(exc))
        return

    if file_path:
        dpg.set_value("img_path", file_path)
        _status(f"Selected image: {Path(file_path).name}")
        # Circuit preview support removed; no further action required on file select.


def _on_toggle_shots(sender, app_data):
    dpg.configure_item("shots", enabled=bool(app_data))


def _open_output_dir():
    if STATE.run_directory and os.path.isdir(STATE.run_directory):
        os.startfile(STATE.run_directory)
    else:
        _status("No output directory available yet.")


def _open_saved_plot(sender=None, app_data=None, user_data=None):
    key = str(user_data or "")
    path = STATE.mpl_plot_paths.get(key)
    if path and os.path.isfile(path):
        os.startfile(path)
        return
    _status(f"Saved MPL plot not found for '{key}'.")


def _on_toggle_verbose_plots(sender, app_data):
    dpg.configure_item("diag_panel", show=bool(app_data))


def _run_pipeline_job(config_dict):
    config = GUIConfig(**config_dict)
    try:
        result_tuple = run_pipeline(
            shots=config.shots,
            n=config.size,
            d=config.dimension,
            bit_depth=config.bit_depth,
            denoise=config.denoise,
            use_shots=config.use_shots,
            fast=config.fast,
            verbose_plots=config.verbose_plots,
            use_gpu=config.use_gpu,
            img_path=config.img_path,
            run_comparison=config.run_comparison,
            return_diagnostics=True,
        )

        if len(result_tuple) == 4:
            original, reconstructed, run_dir, diagnostics = result_tuple
        else:
            original, reconstructed, run_dir = result_tuple
            diagnostics = {
                "mse_map": (original.astype(np.float32) - reconstructed.astype(np.float32)) ** 2,
                "bias_map": None,
                "mpl_paths": {},
            }

        mse = float(np.mean((original.astype(np.float32) - reconstructed.astype(np.float32)) ** 2))
        return {
            "ok": True,
            "original": original,
            "reconstructed": reconstructed,
            "run_dir": run_dir,
            "mse": mse,
            "diagnostics": diagnostics,
            "config": asdict(config),
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def _on_run():
    if STATE.is_running:
        _status("A run is already in progress.")
        return

    config = _get_config_from_ui()

    if not config.img_path:
        _status("Please select an input image first.")
        return

    if not os.path.isfile(config.img_path):
        _status("Image path does not exist.")
        return

    if not _is_power_of_two(config.size):
        _status("Image size must be a power of two (e.g., 4, 8, 16, 32).")
        return

    if config.dimension < 2:
        _status("Dimension must be >= 2.")
        return

    if config.bit_depth < 1:
        _status("Bit depth must be >= 1.")
        return

    STATE.is_running = True
    _set_controls_enabled(False)
    _status("Running MHRQI pipeline... this can take a while for large shots/comparisons.")
    if STATE.executor is None:
        STATE.executor = ProcessPoolExecutor(max_workers=1)
    STATE.pending_future = STATE.executor.submit(_run_pipeline_job, asdict(config))


def _process_async_results():
    # Handle pipeline job results (circuit preview removed)
    if STATE.pending_future is None:
        return

    if not STATE.pending_future.done():
        return

    try:
        result = STATE.pending_future.result()
    except Exception as exc:
        result = {"ok": False, "error": f"Worker failed: {exc}"}
    STATE.pending_future = None

    STATE.is_running = False

    if result.get("ok"):
        STATE.run_directory = result["run_dir"]
        _set_preview_images(result["original"], result["reconstructed"])
        diagnostics = result.get("diagnostics") or {}
        STATE.mpl_plot_paths = diagnostics.get("mpl_paths") or {}
        _update_diagnostic_plots(
            result["original"],
            result["reconstructed"],
            diagnostics=diagnostics,
            verbose_enabled=result["config"].get("verbose_plots", False),
            denoise_enabled=result["config"].get("denoise", False),
        )

        dpg.configure_item(
            "open_mpl_comparison_button",
            enabled=bool(STATE.mpl_plot_paths.get("comparison")),
        )
        dpg.configure_item("open_mpl_mse_button", enabled=bool(STATE.mpl_plot_paths.get("mse")))
        dpg.configure_item("open_mpl_bias_button", enabled=bool(STATE.mpl_plot_paths.get("bias")))
        # Benchmark plots (if run_comparison was enabled)
        if result["config"].get("run_comparison", False):
            has_full_ref = bool(STATE.mpl_plot_paths.get("comp_full_ref"))
            has_speckle = bool(STATE.mpl_plot_paths.get("comp_speckle"))
            has_structural = bool(STATE.mpl_plot_paths.get("comp_structural"))

            dpg.configure_item("bench_panel", show=(has_full_ref or has_speckle or has_structural))

            # Do NOT load MPL-rendered benchmark images into the UI. The
            # saved Matplotlib plots remain available via the "Open Saved MPL"
            # buttons only; this keeps the UI using DPG-native plots exclusively.
            # (Textures are intentionally not created/assigned here.)

            if dpg.does_item_exist("open_mpl_bench_full_ref"):
                dpg.configure_item("open_mpl_bench_full_ref", enabled=has_full_ref)
            dpg.configure_item("open_mpl_bench_speckle", enabled=has_speckle)
            dpg.configure_item("open_mpl_bench_structural", enabled=has_structural)
            # Update DPG-native plots/tables from numeric benchmark results if available
            bench_results = diagnostics.get("bench_results") if diagnostics else None
            if bench_results:
                try:
                    _update_benchmark_dpg(bench_results)
                except Exception:
                    pass
            else:
                dpg.configure_item("bench_panel", show=(has_full_ref or has_speckle or has_structural))
        _status("Pipeline completed successfully.")
        metrics_text = (
            f"MSE: {result['mse']:.4f}\n"
            f"Output directory: {result['run_dir']}\n"
            f"Denoise: {result['config']['denoise']} | "
            f"Use shots: {result['config']['use_shots']} | "
            f"Shots: {result['config']['shots'] if result['config']['use_shots'] else 'N/A'}"
        )
        dpg.set_value("metrics_text", metrics_text)
    else:
        _status(f"Run failed: {result.get('error', 'Unknown error')}")



    _set_controls_enabled(True)


def _build_ui():
    dpg.create_context()

    with dpg.texture_registry(show=False, tag=TEXTURE_REGISTRY_TAG):
        dpg.add_static_texture(1, 1, [0.2, 0.2, 0.2, 1.0], tag=ORIG_TEXTURE_TAG)
        dpg.add_static_texture(1, 1, [0.2, 0.2, 0.2, 1.0], tag=RECON_TEXTURE_TAG)
        dpg.add_static_texture(1, 1, [0.2, 0.2, 0.2, 1.0], tag=MSE_TEXTURE_TAG)
        dpg.add_static_texture(1, 1, [0.2, 0.2, 0.2, 1.0], tag=BIAS_TEXTURE_TAG)
        # Circuit preview texture removed per user request.
        dpg.add_static_texture(1, 1, [0.2, 0.2, 0.2, 1.0], tag=BENCH_FULL_REF_TEXTURE_TAG)
        dpg.add_static_texture(1, 1, [0.2, 0.2, 0.2, 1.0], tag=BENCH_SPECKLE_TEXTURE_TAG)
        dpg.add_static_texture(1, 1, [0.2, 0.2, 0.2, 1.0], tag=BENCH_STRUCTURAL_TEXTURE_TAG)
        STATE.current_orig_texture = ORIG_TEXTURE_TAG
        STATE.current_recon_texture = RECON_TEXTURE_TAG
        STATE.current_mse_texture = MSE_TEXTURE_TAG
        STATE.current_bias_texture = BIAS_TEXTURE_TAG
        # STATE.current_circuit_texture intentionally omitted (preview removed)
        STATE.current_bench_full_ref_texture = BENCH_FULL_REF_TEXTURE_TAG
        STATE.current_bench_speckle_texture = BENCH_SPECKLE_TEXTURE_TAG
        STATE.current_bench_structural_texture = BENCH_STRUCTURAL_TEXTURE_TAG

    with dpg.window(tag="main_window", label="MHRQI Denoising Workbench", width=1200, height=760):
        with dpg.menu_bar():
            with dpg.menu(label="File"):
                dpg.add_menu_item(label="Load Image", callback=_show_file_dialog)
                dpg.add_menu_item(label="Open Output Directory", callback=_open_output_dir)
                dpg.add_menu_item(label="Exit", callback=lambda: dpg.stop_dearpygui())
            with dpg.menu(label="Run"):
                dpg.add_menu_item(label="Execute Pipeline", callback=_on_run)

        with dpg.group(horizontal=True):
            with dpg.child_window(width=420, autosize_y=True, border=True):
                dpg.add_text("Input")
                with dpg.group(horizontal=True):
                    dpg.add_input_text(tag="img_path", hint="Path to image file", width=280, callback=_on_ui_config_change)
                    dpg.add_button(label="Browse", callback=_show_file_dialog)

                dpg.add_separator()
                dpg.add_text("Pipeline Configuration")
                dpg.add_input_int(tag="size", label="Image Size (n x n)", default_value=16, min_value=2, callback=_on_ui_config_change)
                dpg.add_input_int(tag="dimension", label="Qudit Dimension (d)", default_value=2, min_value=2, callback=_on_ui_config_change)
                dpg.add_input_int(tag="bit_depth", label="Bit Depth", default_value=8, min_value=1, callback=_on_ui_config_change)
                dpg.add_checkbox(
                    tag="use_shots",
                    label="Use Shot-Based Simulation",
                    default_value=True,
                    callback=_on_toggle_shots,
                )
                dpg.add_input_int(tag="shots", label="Shots", default_value=1000, min_value=1)
                dpg.add_checkbox(tag="denoise", label="Apply Denoising", default_value=True, callback=_on_ui_config_change)
                dpg.add_checkbox(tag="fast", label="Fast Upload (lazy)", default_value=False, callback=_on_ui_config_change)
                dpg.add_checkbox(
                    tag="verbose_plots",
                    label="Verbose Plots",
                    default_value=False,
                    callback=_on_toggle_verbose_plots,
                )
                dpg.add_checkbox(tag="run_comparison", label="Run Classical Benchmarks", default_value=False)
                dpg.add_checkbox(tag="use_gpu", label="Use GPU Backend (if available)", default_value=False)

                dpg.add_separator()
                with dpg.group(horizontal=True):
                    dpg.add_button(tag="run_button", label="Run Denoising", callback=_on_run, width=180)
                    dpg.add_button(
                        tag="open_output_button",
                        label="Open Output Folder",
                        callback=_open_output_dir,
                        width=180,
                        enabled=False,
                    )

                dpg.add_separator()
                dpg.add_text("Status")
                dpg.add_text("Idle", tag="status_text", wrap=380)
                dpg.add_spacer(height=8)
                dpg.add_text("Quick Metrics")
                dpg.add_text("No runs yet.", tag="metrics_text", wrap=380)

            with dpg.child_window(autosize_x=True, autosize_y=True, border=True, tag=PREVIEW_CHILD_TAG):
                dpg.add_text("Preview")
                with dpg.collapsing_header(label="Image Comparison Preview", default_open=True):
                    with dpg.group(horizontal=True):
                        with dpg.group():
                            dpg.add_text("Original")
                            dpg.add_image(ORIG_TEXTURE_TAG, tag="orig_image", width=360, height=360)
                        with dpg.group():
                            dpg.add_text("Reconstructed")
                            dpg.add_image(RECON_TEXTURE_TAG, tag="recon_image", width=360, height=360)
                    dpg.add_button(
                        tag="open_mpl_comparison_button",
                        label="Open Saved MPL Comparison Plot",
                        callback=_open_saved_plot,
                        user_data="comparison",
                        enabled=False,
                    )

                dpg.add_spacer(height=8)
                # Quantum circuit preview removed per user request.
                with dpg.collapsing_header(
                    label="Diagnostic Plots (DPG)",
                    default_open=False,
                    tag="diag_panel",
                    show=False,
                ):
                    with dpg.group(tag="mse_plot_group"):
                        dpg.add_text("Per-pixel Squared Error (MSE Map)")
                        dpg.add_image(MSE_TEXTURE_TAG, tag="mse_image", width=360, height=360)
                        dpg.add_text("No MSE map available.", tag="mse_plot_stats", wrap=720)
                        dpg.add_button(
                            tag="open_mpl_mse_button",
                            label="Open Saved MPL MSE Plot",
                            callback=_open_saved_plot,
                            user_data="mse",
                            enabled=False,
                        )

                    dpg.add_spacer(height=8)
                    with dpg.group(tag="bias_plot_group", show=False):
                        dpg.add_text("Bias Confidence (Hit Ratio)")
                        dpg.add_image(BIAS_TEXTURE_TAG, tag="bias_image", width=360, height=360)
                        dpg.add_text(
                            "Bias map available for denoise + verbose runs.",
                            tag="bias_plot_stats",
                            wrap=720,
                        )
                        dpg.add_button(
                            tag="open_mpl_bias_button",
                            label="Open Saved MPL Bias Plot",
                            callback=_open_saved_plot,
                            user_data="bias",
                            enabled=False,
                        )

                dpg.add_spacer(height=8)
                with dpg.collapsing_header(
                    label="Benchmark Plots (DPG)",
                    default_open=False,
                    tag="bench_panel",
                    show=False,
                ):
                    # Full-reference metrics are not displayed in the DPG UI.

                    dpg.add_spacer(height=8)
                    with dpg.group(tag="bench_speckle_group"):
                        dpg.add_text("Speckle Reduction Metrics")
                        # Create individual DPG plots for each speckle metric
                        for _metric in ["SSI", "SMPI", "NSF", "ENL", "CNR"]:
                            with dpg.plot(
                                tag=f"bench_speckle_plot_{_metric}",
                                height=240,
                                width=720,
                                no_menus=True,
                                no_box_select=True,
                                no_mouse_pos=True,
                                no_inputs=True,
                            ):
                                dpg.add_plot_axis(dpg.mvXAxis, tag=f"bench_speckle_xaxis_{_metric}", label="Method")
                                dpg.add_plot_axis(dpg.mvYAxis, tag=f"bench_speckle_yaxis_{_metric}", label="Score")
                                dpg.add_plot_legend()
                            dpg.add_spacer(height=4)
                            dpg.add_text("", tag=f"bench_speckle_values_{_metric}")
                            dpg.add_text("", tag=f"bench_speckle_top_{_metric}")
                        # Table (will be populated dynamically)
                        with dpg.table(header_row=True, tag="bench_speckle_table"):
                            dpg.add_table_column(label="Method")
                            dpg.add_table_column(label="SSI")
                            dpg.add_table_column(label="SMPI")
                            dpg.add_table_column(label="NSF")
                            dpg.add_table_column(label="ENL")
                            dpg.add_table_column(label="CNR")
                        dpg.add_image(BENCH_SPECKLE_TEXTURE_TAG, tag="bench_speckle_image", width=720, height=120, show=False)
                        dpg.add_button(
                            tag="open_mpl_bench_speckle",
                            label="Open Saved MPL Speckle Plot",
                            callback=_open_saved_plot,
                            user_data="comp_speckle",
                            enabled=False,
                        )

                    dpg.add_spacer(height=8)
                    with dpg.group(tag="bench_structural_group"):
                        dpg.add_text("Structural Similarity Metrics")
                        # Create individual DPG plots for each structural metric
                        for _metric in ["EPF", "EPI", "OMQDI"]:
                            with dpg.plot(
                                tag=f"bench_structural_plot_{_metric}",
                                height=240,
                                width=720,
                                no_menus=True,
                                no_box_select=True,
                                no_mouse_pos=True,
                                no_inputs=True,
                            ):
                                dpg.add_plot_axis(dpg.mvXAxis, tag=f"bench_structural_xaxis_{_metric}", label="Method")
                                dpg.add_plot_axis(dpg.mvYAxis, tag=f"bench_structural_yaxis_{_metric}", label="Score")
                                dpg.add_plot_legend()
                            dpg.add_spacer(height=4)
                            dpg.add_text("", tag=f"bench_structural_values_{_metric}")
                            dpg.add_text("", tag=f"bench_structural_top_{_metric}")
                        # Table (will be populated dynamically)
                        with dpg.table(header_row=True, tag="bench_structural_table"):
                            dpg.add_table_column(label="Method")
                            dpg.add_table_column(label="EPF")
                            dpg.add_table_column(label="EPI")
                            dpg.add_table_column(label="OMQDI")
                        dpg.add_image(BENCH_STRUCTURAL_TEXTURE_TAG, tag="bench_structural_image", width=720, height=120, show=False)
                        dpg.add_button(
                            tag="open_mpl_bench_structural",
                            label="Open Saved MPL Structural Plot",
                            callback=_open_saved_plot,
                            user_data="comp_structural",
                            enabled=False,
                        )

    dpg.create_viewport(title="MHRQI GUI", width=1220, height=800)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("main_window", True)

    # Circuit preview support removed; no drawlist sizing or initial build scheduled.


def launch_gui():
    if STATE.executor is None:
        STATE.executor = ProcessPoolExecutor(max_workers=1)
    _build_ui()
    while dpg.is_dearpygui_running():
        _process_async_results()
        dpg.render_dearpygui_frame()
    if STATE.executor is not None:
        STATE.executor.shutdown(wait=False, cancel_futures=True)
        STATE.executor = None
    dpg.destroy_context()


if __name__ == "__main__":
    launch_gui()
