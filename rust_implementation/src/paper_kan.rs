use serde::Deserialize;

#[derive(Debug, Clone)]
pub struct PaperKanParams {
    pub g: usize,
    pub k: usize,
    pub in_features: usize,
    pub basis_count: usize,
    pub out_features: usize,
    pub r: f32,
    pub s: Vec<f32>, // row-major (in_features, basis_count)
    pub e: Vec<f32>, // row-major (in_features, basis_count)
    pub w: Vec<f32>, // row-major (out_features, in_features, basis_count)
}

#[derive(Debug, Deserialize)]
struct ParamsJson {
    #[allow(dead_code)]
    paper: Option<String>,
    #[serde(rename = "G")]
    g: usize,
    k: usize,
    in_features: usize,
    basis_count: usize,
    out_features: usize,
    r: f32,
    #[serde(rename = "S")]
    s: Vec<Vec<f32>>,
    #[serde(rename = "E")]
    e: Vec<Vec<f32>>,
    #[serde(rename = "W")]
    w: Vec<Vec<Vec<f32>>>,
}

fn flatten_2d(v: Vec<Vec<f32>>, rows: usize, cols: usize, name: &str) -> anyhow::Result<Vec<f32>> {
    if v.len() != rows {
        anyhow::bail!("{name} rows mismatch: expected {rows}, got {}", v.len());
    }
    let mut out = Vec::with_capacity(rows * cols);
    for (i, row) in v.into_iter().enumerate() {
        if row.len() != cols {
            anyhow::bail!("{name}[{i}] cols mismatch: expected {cols}, got {}", row.len());
        }
        out.extend_from_slice(&row);
    }
    Ok(out)
}

fn flatten_3d(
    v: Vec<Vec<Vec<f32>>>,
    a: usize,
    b: usize,
    c: usize,
    name: &str,
) -> anyhow::Result<Vec<f32>> {
    if v.len() != a {
        anyhow::bail!("{name} dim0 mismatch: expected {a}, got {}", v.len());
    }
    let mut out = Vec::with_capacity(a * b * c);
    for (i, vv) in v.into_iter().enumerate() {
        if vv.len() != b {
            anyhow::bail!("{name}[{i}] dim1 mismatch: expected {b}, got {}", vv.len());
        }
        for (j, row) in vv.into_iter().enumerate() {
            if row.len() != c {
                anyhow::bail!("{name}[{i}][{j}] dim2 mismatch: expected {c}, got {}", row.len());
            }
            out.extend_from_slice(&row);
        }
    }
    Ok(out)
}

pub fn load_params_json(path: &std::path::Path) -> anyhow::Result<PaperKanParams> {
    let bytes = std::fs::read(path)?;
    let pj: ParamsJson = serde_json::from_slice(&bytes)?;
    let s = flatten_2d(pj.s, pj.in_features, pj.basis_count, "S")?;
    let e = flatten_2d(pj.e, pj.in_features, pj.basis_count, "E")?;
    let w = flatten_3d(pj.w, pj.out_features, pj.in_features, pj.basis_count, "W")?;
    Ok(PaperKanParams {
        g: pj.g,
        k: pj.k,
        in_features: pj.in_features,
        basis_count: pj.basis_count,
        out_features: pj.out_features,
        r: pj.r,
        s,
        e,
        w,
    })
}

#[inline(always)]
fn relu(x: f32) -> f32 {
    if x > 0.0 { x } else { 0.0 }
}

#[inline(always)]
fn idx2(i: usize, j: usize, cols: usize) -> usize {
    i * cols + j
}

#[inline(always)]
fn idx3(o: usize, i: usize, j: usize, in_features: usize, basis: usize) -> usize {
    // (out, in, basis)
    o * in_features * basis + i * basis + j
}

/// Forward for one input vector x (length in_features).
/// Implements: F = r * (ReLU(E - x) * ReLU(x - S))^2, then y_o = sum_{i,j} W[o,i,j] * F[i,j].
pub fn forward_one(x: &[f32], p: &PaperKanParams, out: &mut [f32]) {
    debug_assert_eq!(x.len(), p.in_features);
    debug_assert_eq!(out.len(), p.out_features);

    // Compute y directly without storing full F (memory-friendly).
    out.fill(0.0);
    let basis = p.basis_count;
    for i in 0..p.in_features {
        let xi = x[i];
        for j in 0..basis {
            let a = relu(p.e[idx2(i, j, basis)] - xi);
            let b = relu(xi - p.s[idx2(i, j, basis)]);
            let f = p.r * (a * b) * (a * b);
            for o in 0..p.out_features {
                out[o] += p.w[idx3(o, i, j, p.in_features, basis)] * f;
            }
        }
    }
}

