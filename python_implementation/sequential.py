import argparse
import time
from pathlib import Path

import numpy as np

from paper_kan_core import forward_paper_kan, load_params_json


def load_csv(path: Path) -> np.ndarray:
    return np.loadtxt(path, delimiter=",").astype(np.float32, copy=False)


def save_csv(matrix: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, matrix, delimiter=",")


def init_seq_viz_dir(base_dir: Path, input_shape: tuple, kernel: int, stride: int, padding: int, out_channels: int) -> Path:
    directory = base_dir / f"seq_visualization_i{input_shape[0]}_k{kernel}_s{stride}_p{padding}_oc{out_channels}"
    (directory / "input").mkdir(parents=True, exist_ok=True)
    (directory / "kan").mkdir(parents=True, exist_ok=True)
    return directory


def sliding_window_paper_kan(
    input_matrix: np.ndarray,
    params_dir: Path,
    kernel: int,
    stride: int,
    padding: int,
    viz_dir: Path | None = None,
    viz_channel: int = 0,
) -> np.ndarray:
    """
    Applies paper-KAN core over all sliding windows of size (kernel x kernel).

    Returns:
      output: (H_out, W_out, out_channels)
    """
    p = load_params_json(params_dir)

    if p.in_features != kernel * kernel:
        raise ValueError(f"Params in_features={p.in_features} != kernel^2={kernel*kernel}")

    x_padded = np.pad(input_matrix, ((padding, padding), (padding, padding)), mode="constant", constant_values=0.0)
    H, W = x_padded.shape
    H_out = ((H - kernel) // stride) + 1
    W_out = ((W - kernel) // stride) + 1

    out = np.zeros((H_out, W_out, p.out_features), dtype=np.float32)

    if viz_dir is not None:
        save_csv(input_matrix, viz_dir / "input" / "input_matrix.csv")

    for oi, i in enumerate(range(0, H - kernel + 1, stride)):
        for oj, j in enumerate(range(0, W - kernel + 1, stride)):
            patch = x_padded[i : i + kernel, j : j + kernel].reshape(-1)
            out[oi, oj, :] = forward_paper_kan(patch, p).astype(np.float32, copy=False)
            if viz_dir is not None:
                ch = int(viz_channel)
                ch = max(0, min(ch, p.out_features - 1))
                save_csv(out[:, :, ch], viz_dir / "kan" / f"step_{oi*W_out+oj}.csv")

    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=Path("../data/input_matrix_128.csv"))
    ap.add_argument("--params-dir", type=Path, required=True)
    ap.add_argument("--kernel", type=int, default=3)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--padding", type=int, default=0)
    ap.add_argument("--viz", action="store_true", help="Dump CSV kadrove u ./seq_visualization_* (mali ulazi).")
    ap.add_argument(
        "--save-csv",
        type=Path,
        default=None,
        help="Opciono: putanja do jednog CSV-a (samo kanal 0) za poređenje / debug.",
    )
    args = ap.parse_args()

    x = load_csv(args.input)

    t0 = time.time()
    viz_dir = None
    if args.viz:
        p_tmp = load_params_json(args.params_dir)
        viz_dir = init_seq_viz_dir(Path("."), x.shape, args.kernel, args.stride, args.padding, p_tmp.out_features)

    y = sliding_window_paper_kan(
        x,
        args.params_dir,
        kernel=args.kernel,
        stride=args.stride,
        padding=args.padding,
        viz_dir=viz_dir,
        viz_channel=0,
    )
    dt = time.time() - t0

    print(f"Done in {dt:.4f}s. Output shape: {y.shape}")
    print(
        "Stats: "
        + f"min={float(np.min(y)):.6g} "
        + f"max={float(np.max(y)):.6g} "
        + f"mean={float(np.mean(y)):.6g} "
        + f"std={float(np.std(y)):.6g}"
    )

    if args.save_csv is not None:
        args.save_csv.parent.mkdir(parents=True, exist_ok=True)
        save_csv(y[:, :, 0], args.save_csv)
        print(f"Saved channel 0 to {args.save_csv}")


if __name__ == "__main__":
    main()

