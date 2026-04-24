import json
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class PaperKANParams:
    """
    ReLU-KAN single-layer parameters from arXiv:2406.02075 (ReLU-KAN).

    Shapes:
      - S: (in_features, G+k)
      - E: (in_features, G+k)
      - W: (out_features, in_features, G+k)
    """

    G: int
    k: int
    S: np.ndarray
    E: np.ndarray
    W: np.ndarray

    @property
    def in_features(self) -> int:
        return int(self.S.shape[0])

    @property
    def basis_count(self) -> int:
        return int(self.S.shape[1])

    @property
    def out_features(self) -> int:
        return int(self.W.shape[0])

    @property
    def r(self) -> float:
        # Eq. in HTML: r = 16 * G^4 / (k+1)^4
        return float(16 * (self.G**4) / ((self.k + 1) ** 4))


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def forward_paper_kan(x: np.ndarray, p: PaperKANParams) -> np.ndarray:
    """
    Forward for a single ReLU-KAN layer (inference-only), per Eqs (9)-(13) in the paper HTML.

    Input:
      x: (in_features,) or (batch, in_features)
    Output:
      y: (out_features,) or (batch, out_features)
    """
    if x.ndim == 1:
        xb = x[None, :]
    elif x.ndim == 2:
        xb = x
    else:
        raise ValueError(f"x must be 1D or 2D, got shape {x.shape}")

    if xb.shape[1] != p.in_features:
        raise ValueError(f"Expected in_features={p.in_features}, got x.shape={xb.shape}")

    # A = ReLU(E - x^T), B = ReLU(x^T - S)
    # Broadcasting: (batch, in_features, 1) vs (in_features, basis)
    xT = xb[:, :, None]
    A = relu(p.E[None, :, :] - xT)
    B = relu(xT - p.S[None, :, :])

    # Basis function (Eq. 6): R(x) = [ReLU(e-x) * ReLU(x-s)]^2 * 16/(e-s)^4
    # For the standard initialization, (e-s) is constant, so we use p.r as the normalization constant.
    # Result F has shape (batch, in_features, basis_count) matching W's last two dims.
    F = p.r * (A * B) ** 2

    # y_c = sum_{i,j} W[c,i,j] * F[i,j]  (equivalent to conv with equal-sized kernel)
    y = np.tensordot(F, p.W, axes=([1, 2], [1, 2]))  # (batch, out_features)

    if x.ndim == 1:
        return y[0]
    return y


def save_params_json(out_dir: Path, params: PaperKANParams, seed: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "paper": "2406.02075",
        "G": int(params.G),
        "k": int(params.k),
        "in_features": int(params.in_features),
        "basis_count": int(params.basis_count),
        "out_features": int(params.out_features),
        "r": float(params.r),
        "seed": int(seed),
        "dtype": "float32",
        # Store arrays directly in JSON (requested).
        "S": params.S.astype(np.float32, copy=False).tolist(),
        "E": params.E.astype(np.float32, copy=False).tolist(),
        "W": params.W.astype(np.float32, copy=False).tolist(),
    }
    (out_dir / "params.json").write_text(
        json.dumps(payload, indent=2)
        + "\n"
    )


def load_params_json(in_dir: Path) -> PaperKANParams:
    meta = json.loads((in_dir / "params.json").read_text())
    S = np.asarray(meta["S"], dtype=np.float32)
    E = np.asarray(meta["E"], dtype=np.float32)
    W = np.asarray(meta["W"], dtype=np.float32)
    return PaperKANParams(G=int(meta["G"]), k=int(meta["k"]), S=S, E=E, W=W)


def make_initial_S_E(in_features: int, G: int, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Initialize S and E as in the paper (Eq. 8/17x in HTML):
      s_{i,j} = (j - k - 1)/G
      e_{i,j} = j/G
    for j = 1..(G+k).
    """
    basis = G + k
    j = np.arange(1, basis + 1, dtype=np.float32)  # 1..basis
    s_row = (j - k - 1) / G
    e_row = j / G
    S = np.tile(s_row[None, :], (in_features, 1))
    E = np.tile(e_row[None, :], (in_features, 1))
    return S, E

