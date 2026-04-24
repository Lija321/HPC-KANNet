import argparse
import csv
import math
import time
from pathlib import Path
from typing import Any, Callable, List, Tuple

import numpy as np

from sequential import load_csv as load_csv_seq
from sequential import sliding_window_paper_kan
from parallel import sliding_window_paper_kan_parallel


def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]

def parse_size_from_input_name(name: str) -> int | None:
    # expected: input_matrix_{N}.csv
    stem = Path(name).stem
    if not stem.startswith("input_matrix_"):
        return None
    tail = stem[len("input_matrix_") :]
    try:
        return int(tail)
    except ValueError:
        return None


def collect_timings(fn: Callable[[], Any], n_runs: int) -> List[float]:
    """Nezavisna merenja u sekundama (perf_counter), za mean/std/outlier analizu (NTP ~30 uzoraka)."""
    times: List[float] = []
    n = max(1, n_runs)
    for _ in range(n):
        t0 = time.perf_counter()
        _ = fn()
        times.append(time.perf_counter() - t0)
    return times


def sample_mean_std(xs: List[float]) -> tuple[float, float]:
    arr = np.asarray(xs, dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan")
    m = float(np.mean(arr))
    if arr.size < 2:
        return m, 0.0
    return m, float(np.std(arr, ddof=1))


def outlier_count_tukey(xs: List[float]) -> int:
    """Broj tačaka van [Q1 - 1.5*IQR, Q3 + 1.5*IQR]."""
    arr = np.asarray(xs, dtype=np.float64)
    if arr.size < 4:
        return 0
    q1, q3 = np.percentile(arr, [25.0, 75.0])
    iqr = q3 - q1
    if iqr <= 0.0:
        return 0
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return int(np.sum((arr < lo) | (arr > hi)))


def strong_scaling(
    x: np.ndarray,
    params_dir: Path,
    kernel: int,
    stride: int,
    padding: int,
    workers_list: List[int],
    tiles_per_dim: int,
    n_runs: int,
) -> List[dict]:
    # Match conv2d-style decomposition: measure a "serial/setup" portion separately
    # (loading params + padding + shape math), then total sequential runtime.
    def run_serial_setup() -> None:
        from paper_kan_core import load_params_json as _load_params_json

        _p = _load_params_json(params_dir)
        _xp = np.pad(x, ((padding, padding), (padding, padding)), mode="constant", constant_values=0.0)
        H, W = _xp.shape
        _ = ((H - kernel) // stride) + 1
        _ = ((W - kernel) // stride) + 1

    serial_times = collect_timings(run_serial_setup, n_runs)
    sm, ss = sample_mean_std(serial_times)
    so = outlier_count_tukey(serial_times)

    seq_times = collect_timings(
        lambda: sliding_window_paper_kan(x, params_dir, kernel, stride, padding),
        n_runs,
    )
    qm, qs = sample_mean_std(seq_times)
    qo = outlier_count_tukey(seq_times)

    rows = []
    for w in workers_list:
        par_times = collect_timings(
            lambda: sliding_window_paper_kan_parallel(
                x, params_dir, kernel, stride, padding, workers=w, tiles_per_dim=tiles_per_dim
            ),
            n_runs,
        )
        pm, ps = sample_mean_std(par_times)
        po = outlier_count_tukey(par_times)
        speedup = qm / pm if pm > 0 else float("nan")
        eff = speedup / w if w > 0 else float("nan")
        rows.append(
            {
                "mode": "strong",
                "H": x.shape[0],
                "W": x.shape[1],
                "kernel": kernel,
                "stride": stride,
                "padding": padding,
                "tiles_per_dim": tiles_per_dim,
                "workers": w,
                "n_runs": n_runs,
                "t_serial_sec": sm,
                "t_serial_std_sec": ss,
                "t_serial_outliers": so,
                "t_seq_sec": qm,
                "t_seq_std_sec": qs,
                "t_seq_outliers": qo,
                "t_par_sec": pm,
                "t_par_std_sec": ps,
                "t_par_outliers": po,
                "speedup": speedup,
                "efficiency": eff,
            }
        )
    return rows


def weak_scaling_sizes(base: int, workers_list: List[int]) -> List[Tuple[int, int]]:
    """
    Keep per-worker workload roughly constant by scaling area ~ workers.
    So size ~ base * sqrt(workers).
    """
    sizes = []
    for w in workers_list:
        scale = math.sqrt(max(1, w))
        sz = int(round(base * scale))
        # Keep sizes aligned with common HPC choices (avoid odd/unavailable pre-generated files).
        if sz % 8 != 0:
            sz = int(round(sz / 8) * 8)
        sz = max(8, sz)
        sizes.append((sz, sz))
    return sizes


def weak_scaling(
    data_dir: Path,
    params_dir: Path,
    base: int,
    kernel: int,
    stride: int,
    padding: int,
    workers_list: List[int],
    tiles_per_dim: int,
    n_runs: int,
) -> List[dict]:
    rows = []
    for (w, (h, ww)) in zip(workers_list, weak_scaling_sizes(base, workers_list)):
        inp = data_dir / f"input_matrix_{h}.csv"
        x = load_csv_seq(inp)
        par_times = collect_timings(
            lambda: sliding_window_paper_kan_parallel(
                x, params_dir, kernel, stride, padding, workers=w, tiles_per_dim=tiles_per_dim
            ),
            n_runs,
        )
        pm, ps = sample_mean_std(par_times)
        po = outlier_count_tukey(par_times)
        rows.append(
            {
                "mode": "weak",
                "H": x.shape[0],
                "W": x.shape[1],
                "kernel": kernel,
                "stride": stride,
                "padding": padding,
                "tiles_per_dim": tiles_per_dim,
                "workers": w,
                "n_runs": n_runs,
                "t_par_sec": pm,
                "t_par_std_sec": ps,
                "t_par_outliers": po,
            }
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("../data"))
    ap.add_argument("--params-dir", type=Path, required=True)
    ap.add_argument("--kernel", type=int, default=3)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--padding", type=int, default=0)
    ap.add_argument("--workers", type=str, default="1,2,4,8")
    ap.add_argument("--tiles-per-dim", type=int, default=2)
    ap.add_argument(
        "--runs",
        type=int,
        default=30,
        help="Broj nezavisnih merenja po tački (NTP: ~30; izveštaj: mean, std, Tukey outlier-i).",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("../outputs/scaling") / "kernel{kernel}" / "strong{strong}" / "weakbase{weakbase}",
        help="Base output directory. Results are written to <out-dir>/python_scaling.csv",
    )
    ap.add_argument("--strong-input", type=str, default="input_matrix_128.csv")
    ap.add_argument("--weak-base", type=int, default=64)
    args = ap.parse_args()

    workers_list = parse_int_list(args.workers)
    strong_n = parse_size_from_input_name(args.strong_input) or -1
    out_dir = Path(
        str(args.out_dir).format(kernel=args.kernel, strong=strong_n, weakbase=int(args.weak_base))
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "python_scaling.csv"

    results: List[dict] = []

    # Strong scaling: fixed input
    x_strong = load_csv_seq(args.data_dir / args.strong_input)
    results.extend(
        strong_scaling(
            x_strong,
            args.params_dir,
            kernel=args.kernel,
            stride=args.stride,
            padding=args.padding,
            workers_list=workers_list,
            tiles_per_dim=args.tiles_per_dim,
            n_runs=args.runs,
        )
    )

    # Weak scaling: input size scales with workers
    results.extend(
        weak_scaling(
            args.data_dir,
            args.params_dir,
            base=args.weak_base,
            kernel=args.kernel,
            stride=args.stride,
            padding=args.padding,
            workers_list=workers_list,
            tiles_per_dim=args.tiles_per_dim,
            n_runs=args.runs,
        )
    )

    # Write CSV
    fieldnames = sorted({k for r in results for k in r.keys()})
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            w.writerow(r)

    print(f"Wrote {len(results)} rows to {out_path}")

    # Markdown tabele za izveštaj (uz NTP grafike).
    md_path = out_path.with_name(out_path.stem + "_tables.md")
    write_scaling_markdown_tables(md_path, results)
    print(f"Wrote tables to {md_path}")


def write_scaling_markdown_tables(md_path: Path, results: List[dict]) -> None:
    strong = [r for r in results if r.get("mode") == "strong"]
    weak = [r for r in results if r.get("mode") == "weak"]

    def row_str(keys: List[str], r: dict) -> str:
        return "| " + " | ".join(str(r.get(k, "")) for k in keys) + " |"

    lines: List[str] = [
        "# Tabele merenja (mean / std / Tukey outlier-i)\n",
        "\n*Ubrzanje i efikasnost računati od srednjih vremena (`t_seq_sec` / `t_par_sec`).*\n",
    ]
    if strong:
        keys = [
            "workers",
            "n_runs",
            "t_serial_sec",
            "t_serial_std_sec",
            "t_serial_outliers",
            "t_seq_sec",
            "t_seq_std_sec",
            "t_seq_outliers",
            "t_par_sec",
            "t_par_std_sec",
            "t_par_outliers",
            "speedup",
            "efficiency",
        ]
        header = "| " + " | ".join(keys) + " |"
        sep = "|" + "|".join(["---"] * len(keys)) + "|"
        lines.append("\n## Jako skaliranje\n\n")
        lines.append(header + "\n")
        lines.append(sep + "\n")
        for r in sorted(strong, key=lambda x: int(x["workers"])):
            lines.append(row_str(keys, r) + "\n")
    if weak:
        keys = [
            "workers",
            "H",
            "W",
            "n_runs",
            "t_par_sec",
            "t_par_std_sec",
            "t_par_outliers",
        ]
        header = "| " + " | ".join(keys) + " |"
        sep = "|" + "|".join(["---"] * len(keys)) + "|"
        lines.append("\n## Slabo skaliranje (paralelno vreme)\n\n")
        lines.append(header + "\n")
        lines.append(sep + "\n")
        for r in sorted(weak, key=lambda x: int(x["workers"])):
            lines.append(row_str(keys, r) + "\n")

    md_path.write_text("".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()

