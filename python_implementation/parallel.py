import argparse
import time
from multiprocessing import get_context
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from paper_kan_core import PaperKANParams, forward_paper_kan, load_params_json


def load_csv(path: Path) -> np.ndarray:
    return np.loadtxt(path, delimiter=",").astype(np.float32, copy=False)


def save_csv(matrix: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, matrix, delimiter=",")


def init_par_viz_dir(
    base_dir: Path,
    input_shape: tuple,
    kernel: int,
    stride: int,
    padding: int,
    out_channels: int,
    workers: int,
    tiles_per_dim: int,
) -> Path:
    directory = base_dir / f"par_visualization_np{workers}_i{input_shape[0]}_k{kernel}_s{stride}_p{padding}_oc{out_channels}_t{tiles_per_dim}"
    (directory / "input").mkdir(parents=True, exist_ok=True)
    (directory / "tiles_input").mkdir(parents=True, exist_ok=True)
    (directory / "tiles_output").mkdir(parents=True, exist_ok=True)
    (directory / "output").mkdir(parents=True, exist_ok=True)
    return directory


# Worker globals (initialized once per process)
_G_INPUT_PADDED: Optional[np.ndarray] = None
_G_PARAMS: Optional[PaperKANParams] = None
_G_KERNEL: int = 0
_G_STRIDE: int = 0
_G_W_OUT: int = 0
_G_H_OUT: int = 0


def _init_worker(
    input_padded: np.ndarray,
    params_dir: str,
    kernel: int,
    stride: int,
    h_out: int,
    w_out: int,
) -> None:
    global _G_INPUT_PADDED, _G_PARAMS, _G_KERNEL, _G_STRIDE, _G_H_OUT, _G_W_OUT
    _G_INPUT_PADDED = input_padded
    _G_PARAMS = load_params_json(Path(params_dir))
    _G_KERNEL = int(kernel)
    _G_STRIDE = int(stride)
    _G_H_OUT = int(h_out)
    _G_W_OUT = int(w_out)



def _compute_tile_halo(task: Tuple[int, int, int, int, int, int, int, int]) -> Tuple[int, int, np.ndarray]:
    """
    Compute a tile but explicitly using a local input subregion (tile + halo).

    Task fields:
      - oi0, oi1, oj0, oj1: output-space tile bounds
      - i0, i1, j0, j1: input-space bounds on the *padded* input for this tile
    Returns (oi0, oj0, chunk) where chunk has shape (oi1-oi0, oj1-oj0, out_channels).
    """
    oi0, oi1, oj0, oj1, i0, i1, j0, j1 = task
    assert _G_INPUT_PADDED is not None
    assert _G_PARAMS is not None

    kernel = _G_KERNEL
    stride = _G_STRIDE
    p = _G_PARAMS

    x_tile = _G_INPUT_PADDED[i0:i1, j0:j1]
    h = oi1 - oi0
    w = oj1 - oj0
    out_chunk = np.zeros((h, w, p.out_features), dtype=np.float32)

    # Offsets of this tile within the local input region
    base_i = oi0 * stride - i0
    base_j = oj0 * stride - j0

    for local_oi in range(h):
        ii = base_i + local_oi * stride
        for local_oj in range(w):
            jj = base_j + local_oj * stride
            patch = x_tile[ii : ii + kernel, jj : jj + kernel].reshape(-1)
            out_chunk[local_oi, local_oj, :] = forward_paper_kan(patch, p).astype(np.float32, copy=False)

    return oi0, oj0, out_chunk


def sliding_window_paper_kan_parallel(
    input_matrix: np.ndarray,
    params_dir: Path,
    kernel: int,
    stride: int,
    padding: int,
    workers: int,
    tiles_per_dim: int,
) -> np.ndarray:
    """Parallel ReLU-KAN sliding-window using spatial tiles + halo."""
    return sliding_window_paper_kan_parallel_tiles_halo(
        input_matrix,
        params_dir,
        kernel=kernel,
        stride=stride,
        padding=padding,
        workers=workers,
        tiles_per_dim=tiles_per_dim,
    )


def sliding_window_paper_kan_parallel_tiles_halo(
    input_matrix: np.ndarray,
    params_dir: Path,
    kernel: int,
    stride: int,
    padding: int,
    workers: int,
    tiles_per_dim: int,
) -> np.ndarray:
    """
    Like sliding_window_paper_kan_parallel_tiles, but each tile computes using an explicit
    local input subregion (tile + halo), i.e. the minimal region of the padded input
    needed to evaluate the tile without reading outside that subregion.
    """
    p = load_params_json(params_dir)
    if p.in_features != kernel * kernel:
        raise ValueError(f"Params in_features={p.in_features} != kernel^2={kernel*kernel}")

    x_padded = np.pad(input_matrix, ((padding, padding), (padding, padding)), mode="constant", constant_values=0.0)
    H, W = x_padded.shape
    H_out = ((H - kernel) // stride) + 1
    W_out = ((W - kernel) // stride) + 1

    out = np.zeros((H_out, W_out, p.out_features), dtype=np.float32)

    ti = max(1, int(tiles_per_dim))
    tile_h = (H_out + ti - 1) // ti
    tile_w = (W_out + ti - 1) // ti

    tasks = []
    for bi in range(ti):
        oi0 = bi * tile_h
        oi1 = min((bi + 1) * tile_h, H_out)
        if oi0 >= oi1:
            continue
        for bj in range(ti):
            oj0 = bj * tile_w
            oj1 = min((bj + 1) * tile_w, W_out)
            if oj0 >= oj1:
                continue

            # Minimal input region (on padded input) required for this tile.
            i0 = oi0 * stride
            i1 = (oi1 - 1) * stride + kernel
            j0 = oj0 * stride
            j1 = (oj1 - 1) * stride + kernel

            tasks.append((oi0, oi1, oj0, oj1, i0, i1, j0, j1))

    try:
        ctx = get_context("fork")
    except ValueError:
        ctx = get_context()

    with ctx.Pool(
        processes=workers,
        initializer=_init_worker,
        initargs=(x_padded, str(params_dir), kernel, stride, H_out, W_out),
    ) as pool:
        for oi0, oj0, chunk in pool.imap_unordered(_compute_tile_halo, tasks, chunksize=1):
            out[oi0 : oi0 + chunk.shape[0], oj0 : oj0 + chunk.shape[1], :] = chunk

    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=Path("../data/input_matrix_128.csv"))
    ap.add_argument("--params-dir", type=Path, required=True)
    ap.add_argument("--kernel", type=int, default=3)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--padding", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--tiles-per-dim", type=int, default=2)
    ap.add_argument("--viz", action="store_true", help="Dump tile CSV u ./par_visualization_* (mali ulazi).")
    ap.add_argument(
        "--save-csv",
        type=Path,
        default=None,
        help="Opciono: jedan CSV (kanal 0) za poređenje / debug.",
    )
    args = ap.parse_args()

    x = load_csv(args.input)

    viz_dir = None
    if args.viz:
        p_tmp = load_params_json(args.params_dir)
        viz_dir = init_par_viz_dir(
            Path("."),
            x.shape,
            args.kernel,
            args.stride,
            args.padding,
            p_tmp.out_features,
            args.workers,
            args.tiles_per_dim,
        )
        save_csv(x, viz_dir / "input" / "input_matrix.csv")

    t0 = time.time()
    y = sliding_window_paper_kan_parallel(
        x,
        args.params_dir,
        kernel=args.kernel,
        stride=args.stride,
        padding=args.padding,
        workers=args.workers,
        tiles_per_dim=args.tiles_per_dim,
    )
    dt = time.time() - t0

    print(f"Done in {dt:.4f}s. Output shape: {y.shape} (workers={args.workers}, tiles_per_dim={args.tiles_per_dim})")
    print(
        "Stats: "
        + f"min={float(np.min(y)):.6g} "
        + f"max={float(np.max(y)):.6g} "
        + f"mean={float(np.mean(y)):.6g} "
        + f"std={float(np.std(y)):.6g}"
    )

    if viz_dir is not None:
        # Dump per-tile input region and per-tile output (channel only) by reusing same tiling logic.
        p = load_params_json(args.params_dir)
        kernel = args.kernel
        stride = args.stride
        padding = args.padding
        x_padded = np.pad(x, ((padding, padding), (padding, padding)), mode="constant", constant_values=0.0)
        H, W = x_padded.shape
        H_out = ((H - kernel) // stride) + 1
        W_out = ((W - kernel) // stride) + 1
        ti = max(1, int(args.tiles_per_dim))
        tile_h = (H_out + ti - 1) // ti
        tile_w = (W_out + ti - 1) // ti
        ch = 0

        tile_idx = 0
        for bi in range(ti):
            oi0 = bi * tile_h
            oi1 = min((bi + 1) * tile_h, H_out)
            if oi0 >= oi1:
                continue
            for bj in range(ti):
                oj0 = bj * tile_w
                oj1 = min((bj + 1) * tile_w, W_out)
                if oj0 >= oj1:
                    continue
                i0 = oi0 * stride
                i1 = (oi1 - 1) * stride + kernel
                j0 = oj0 * stride
                j1 = (oj1 - 1) * stride + kernel
                save_csv(x_padded[i0:i1, j0:j1], viz_dir / "tiles_input" / f"tile_{tile_idx}.csv")
                save_csv(y[oi0:oi1, oj0:oj1, ch], viz_dir / "tiles_output" / f"tile_{tile_idx}.csv")
                tile_idx += 1

        # Save merged output (single channel) for the visualization.
        save_csv(y[:, :, ch], viz_dir / "output" / "output_matrix.csv")

    if args.save_csv is not None:
        args.save_csv.parent.mkdir(parents=True, exist_ok=True)
        save_csv(y[:, :, 0], args.save_csv)
        print(f"Saved channel 0 to {args.save_csv}")


if __name__ == "__main__":
    main()

