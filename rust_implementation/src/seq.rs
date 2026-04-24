use crate::paper_kan;

pub fn run_sequential(
    hp: usize,
    wp: usize,
    x_padded: &[f32],
    p: &paper_kan::PaperKanParams,
    kernel: usize,
    stride: usize,
) -> (usize, usize, Vec<f32>) {
    let h_out = (hp.saturating_sub(kernel)) / stride + 1;
    let w_out = (wp.saturating_sub(kernel)) / stride + 1;
    let mut y = vec![0.0f32; h_out * w_out * p.out_features];

    let mut patch = vec![0.0f32; kernel * kernel];
    let mut tmp_out = vec![0.0f32; p.out_features];

    for oi in 0..h_out {
        let i0 = oi * stride;
        for oj in 0..w_out {
            let j0 = oj * stride;
            // Extract patch.
            for pi in 0..kernel {
                let src_row = i0 + pi;
                let dst_row = pi * kernel;
                let base = src_row * wp + j0;
                patch[dst_row..dst_row + kernel].copy_from_slice(&x_padded[base..base + kernel]);
            }
            paper_kan::forward_one(&patch, p, &mut tmp_out);
            let base_y = (oi * w_out + oj) * p.out_features;
            y[base_y..base_y + p.out_features].copy_from_slice(&tmp_out);
        }
    }
    (h_out, w_out, y)
}

