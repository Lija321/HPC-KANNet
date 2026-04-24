use anyhow::Context;

pub fn load_csv_matrix_f32(path: &std::path::Path) -> anyhow::Result<(usize, usize, Vec<f32>)> {
    let s = std::fs::read_to_string(path)
        .with_context(|| format!("failed reading CSV {}", path.display()))?;
    let mut data: Vec<f32> = Vec::new();
    let mut rows = 0usize;
    let mut cols_expected: Option<usize> = None;
    for line in s.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let cols: Vec<&str> = line.split(',').collect();
        if cols_expected.is_none() {
            cols_expected = Some(cols.len());
        } else if cols_expected != Some(cols.len()) {
            anyhow::bail!(
                "inconsistent CSV columns in {}: expected {:?}, got {}",
                path.display(),
                cols_expected,
                cols.len()
            );
        }
        for c in cols {
            data.push(c.parse::<f32>().with_context(|| format!("parse float '{c}'"))?);
        }
        rows += 1;
    }
    let cols = cols_expected.unwrap_or(0);
    Ok((rows, cols, data))
}

#[inline(always)]
pub fn idx2(i: usize, j: usize, cols: usize) -> usize {
    i * cols + j
}

pub fn padded_matrix(
    h: usize,
    w: usize,
    input: &[f32],
    padding: usize,
) -> (usize, usize, Vec<f32>) {
    if padding == 0 {
        return (h, w, input.to_vec());
    }
    let hp = h + 2 * padding;
    let wp = w + 2 * padding;
    let mut out = vec![0.0f32; hp * wp];
    for i in 0..h {
        for j in 0..w {
            out[idx2(i + padding, j + padding, wp)] = input[idx2(i, j, w)];
        }
    }
    (hp, wp, out)
}

