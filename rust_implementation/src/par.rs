use crate::paper_kan;

use std::sync::{
    atomic::{AtomicUsize, Ordering},
    mpsc, Arc,
};
use std::thread;

pub fn run_parallel_tiles_halo(
    hp: usize,
    wp: usize,
    x_padded: Vec<f32>,
    p: paper_kan::PaperKanParams,
    kernel: usize,
    stride: usize,
    threads: usize,
    tiles_per_dim: usize,
) -> (usize, usize, Vec<f32>) {
    let h_out = (hp.saturating_sub(kernel)) / stride + 1;
    let w_out = (wp.saturating_sub(kernel)) / stride + 1;
    let mut y = vec![0.0f32; h_out * w_out * p.out_features];

    let ti = tiles_per_dim.max(1);
    let tile_h = (h_out + ti - 1) / ti;
    let tile_w = (w_out + ti - 1) / ti;

    let x_arc = Arc::new(x_padded);
    let p_arc = Arc::new(p);

    let (tx, rx) = mpsc::channel::<(usize, usize, usize, usize, Vec<f32>)>();
    let mut tasks = Vec::new();
    for bi in 0..ti {
        let oi0 = bi * tile_h;
        let oi1 = ((bi + 1) * tile_h).min(h_out);
        if oi0 >= oi1 {
            continue;
        }
        for bj in 0..ti {
            let oj0 = bj * tile_w;
            let oj1 = ((bj + 1) * tile_w).min(w_out);
            if oj0 >= oj1 {
                continue;
            }
            tasks.push((oi0, oi1, oj0, oj1));
        }
    }

    let tasks = Arc::new(tasks);
    let next_idx = Arc::new(AtomicUsize::new(0));

    let mut handles = Vec::new();
    for _ in 0..threads.max(1) {
        let x = Arc::clone(&x_arc);
        let p = Arc::clone(&p_arc);
        let tasks = Arc::clone(&tasks);
        let next_idx = Arc::clone(&next_idx);
        let tx_out = tx.clone();
        let handle = thread::spawn(move || {
            let mut patch = vec![0.0f32; kernel * kernel];
            let mut tmp_out = vec![0.0f32; p.out_features];
            loop {
                let idx = next_idx.fetch_add(1, Ordering::Relaxed);
                if idx >= tasks.len() {
                    break;
                }
                let (oi0, oi1, oj0, oj1) = tasks[idx];

                // Minimal input region for halo.
                let i0 = oi0 * stride;
                let i1 = (oi1 - 1) * stride + kernel;
                let j0 = oj0 * stride;
                let j1 = (oj1 - 1) * stride + kernel;
                let tile_hp = i1 - i0;
                let tile_wp = j1 - j0;
                let mut x_tile = vec![0.0f32; tile_hp * tile_wp];
                for ii in 0..tile_hp {
                    let src = (i0 + ii) * wp + j0;
                    let dst = ii * tile_wp;
                    x_tile[dst..dst + tile_wp].copy_from_slice(&x[src..src + tile_wp]);
                }

                let out_h = oi1 - oi0;
                let out_w = oj1 - oj0;
                let mut y_tile = vec![0.0f32; out_h * out_w * p.out_features];

                for local_oi in 0..out_h {
                    let ii = local_oi * stride;
                    for local_oj in 0..out_w {
                        let jj = local_oj * stride;
                        for pi in 0..kernel {
                            let base = (ii + pi) * tile_wp + jj;
                            let dst_row = pi * kernel;
                            patch[dst_row..dst_row + kernel]
                                .copy_from_slice(&x_tile[base..base + kernel]);
                        }
                        paper_kan::forward_one(&patch, &p, &mut tmp_out);
                        let base_y = (local_oi * out_w + local_oj) * p.out_features;
                        y_tile[base_y..base_y + p.out_features].copy_from_slice(&tmp_out);
                    }
                }

                let _ = tx_out.send((oi0, oi1, oj0, oj1, y_tile));
            }
        });
        handles.push(handle);
    }
    drop(tx);

    for (oi0, oi1, oj0, oj1, y_tile) in rx {
        let out_h = oi1 - oi0;
        let out_w = oj1 - oj0;
        for local_oi in 0..out_h {
            for local_oj in 0..out_w {
                let src_base = (local_oi * out_w + local_oj) * p_arc.out_features;
                let dst_base = ((oi0 + local_oi) * w_out + (oj0 + local_oj)) * p_arc.out_features;
                y[dst_base..dst_base + p_arc.out_features]
                    .copy_from_slice(&y_tile[src_base..src_base + p_arc.out_features]);
            }
        }
    }

    for h in handles {
        let _ = h.join();
    }

    (h_out, w_out, y)
}

