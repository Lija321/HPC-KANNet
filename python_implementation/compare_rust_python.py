import argparse
import subprocess
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--params-dir", type=Path, required=True)
    ap.add_argument("--workdir", type=Path, default=Path("../outputs/tmp_compare"))
    args = ap.parse_args()

    work = args.workdir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    py_csv = work / "py_c0.csv"
    rs_csv = work / "rs_c0.csv"
    kernel, stride, padding = 3, 1, 0

    # Python (channel 0)
    cmd_py = [
        "python",
        str(Path(__file__).parent / "sequential.py"),
        "--input",
        str(args.input),
        "--params-dir",
        str(args.params_dir),
        "--kernel",
        str(kernel),
        "--stride",
        str(stride),
        "--padding",
        str(padding),
        "--save-csv",
        str(py_csv),
    ]
    subprocess.check_call(cmd_py)

    # Rust (channel 0)
    rust_dir = Path(__file__).parents[1] / "rust_implementation"
    params_json = args.params_dir / "params.json"
    cmd_rs = [
        "cargo",
        "run",
        "-q",
        "--bin",
        "rust_implementation",
        "--",
        "--input",
        str(args.input),
        "--params",
        str(params_json),
        "--kernel",
        str(kernel),
        "--stride",
        str(stride),
        "--padding",
        str(padding),
        "--save-csv",
        str(rs_csv),
    ]
    subprocess.check_call(cmd_rs, cwd=str(rust_dir))

    a = np.loadtxt(str(py_csv), delimiter=",")
    b = np.loadtxt(str(rs_csv), delimiter=",")
    diff = np.abs(a - b)
    print("shape_py", a.shape, "shape_rust", b.shape)
    print("max_abs_diff", float(np.max(diff)))
    print("mean_abs_diff", float(np.mean(diff)))


if __name__ == "__main__":
    main()

