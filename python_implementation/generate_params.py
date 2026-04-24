import argparse
from pathlib import Path

import numpy as np

from paper_kan_core import PaperKANParams, make_initial_S_E, save_params_json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=Path("../data"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kernel", type=int, default=3)
    ap.add_argument("--out-channels", type=int, default=8)
    ap.add_argument("--G", type=int, default=5)
    ap.add_argument("--k", type=int, default=3)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    in_features = args.kernel * args.kernel  # single-channel patch flatten
    S, E = make_initial_S_E(in_features=in_features, G=args.G, k=args.k)

    basis = args.G + args.k
    W = rng.random((args.out_channels, in_features, basis), dtype=np.float32)

    params = PaperKANParams(
        G=args.G,
        k=args.k,
        S=S.astype(np.float32),
        E=E.astype(np.float32),
        W=W.astype(np.float32),
    )

    out_dir = args.out_dir.resolve()
    folder = f"paper_kan_params_in{in_features}_out{args.out_channels}_G{args.G}_k{args.k}_kernel{args.kernel}"
    params_dir = out_dir / folder
    save_params_json(params_dir, params, seed=args.seed)

    print("Wrote:")
    print(f"- params: {params_dir / 'params.json'}")


if __name__ == "__main__":
    main()

