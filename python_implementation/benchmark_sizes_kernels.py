import argparse
import csv
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

from sequential import load_csv as load_csv_seq
from sequential import sliding_window_paper_kan
from parallel import sliding_window_paper_kan_parallel


def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def measure_best(fn, repeats: int) -> float:
    best = float("inf")
    for _ in range(max(1, repeats)):
        t0 = time.perf_counter()
        _ = fn()
        dt = time.perf_counter() - t0
        best = min(best, dt)
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("../data"))
    ap.add_argument("--params-base", type=Path, default=Path("../data"))
    ap.add_argument("--sizes", type=str, default="16,24,32,40,48,56,64,80,96,112,128")
    ap.add_argument("--kernels", type=str, default="2,3,5,7")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--tiles-per-dim", type=int, default=2)
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("../outputs/sizes_kernels") / "python" / "workers{workers}",
        help="Base output directory. Results are written to <out-dir>/sizes_kernels.csv",
    )
    args = ap.parse_args()
    stride, padding, repeats = 1, 0, 1

    sizes = parse_int_list(args.sizes)
    kernels = parse_int_list(args.kernels)

    out_dir = Path(str(args.out_dir).format(workers=args.workers))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "sizes_kernels.csv"

    rows: List[Dict[str, object]] = []

    for k in kernels:
        params_dir = args.params_base / f"paper_kan_params_in{k*k}_out8_G5_k3_kernel{k}"
        for n in sizes:
            inp = args.data_dir / f"input_matrix_{n}.csv"
            x = load_csv_seq(inp)

            t_seq = measure_best(
                lambda: sliding_window_paper_kan(x, params_dir, kernel=k, stride=stride, padding=padding),
                repeats=repeats,
            )
            t_par = measure_best(
                lambda: sliding_window_paper_kan_parallel(
                    x,
                    params_dir,
                    kernel=k,
                    stride=stride,
                    padding=padding,
                    workers=args.workers,
                    tiles_per_dim=args.tiles_per_dim,
                ),
                repeats=repeats,
            )
            rows.append(
                {
                    "language": "python",
                    "H": int(x.shape[0]),
                    "W": int(x.shape[1]),
                    "kernel": int(k),
                    "stride": int(stride),
                    "padding": int(padding),
                    "workers": int(args.workers),
                    "tiles_per_dim": int(args.tiles_per_dim),
                    "t_seq_sec": float(t_seq),
                    "t_par_sec": float(t_par),
                    "speedup": float(t_seq / t_par) if t_par > 0 else float("nan"),
                    "efficiency": float((t_seq / t_par) / args.workers) if t_par > 0 else float("nan"),
                }
            )

    fieldnames = list({k for r in rows for k in r.keys()})
    fieldnames.sort()
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()

