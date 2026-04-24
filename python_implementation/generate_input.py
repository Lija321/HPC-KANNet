import argparse
from pathlib import Path

import numpy as np


def save_csv(matrix: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, matrix, delimiter=",")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=Path("../data"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--H", type=int, default=128)
    ap.add_argument("--W", type=int, default=128)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    input_matrix = rng.random((args.H, args.W), dtype=np.float32)

    out_dir = args.out_dir.resolve()
    out_path = out_dir / f"input_matrix_{args.H}.csv"
    save_csv(input_matrix, out_path)

    print("Wrote:")
    print(f"- input: {out_path}")


if __name__ == "__main__":
    main()

