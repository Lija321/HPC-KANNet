mod io;
mod paper_kan;
mod par;
mod seq;
mod stats;

use clap::Parser;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(name = "hpc-kannet", about = "ReLU-KAN (2406.02075) sliding-window benchmark (Rust)")]
struct Args {
    /// Input CSV matrix (single-channel).
    #[arg(long)]
    input: PathBuf,

    /// Params JSON path (paper KAN params.json).
    #[arg(long)]
    params: PathBuf,

    /// Kernel size n (n x n).
    #[arg(long, default_value_t = 3)]
    kernel: usize,

    /// Stride.
    #[arg(long, default_value_t = 1)]
    stride: usize,

    /// Zero padding.
    #[arg(long, default_value_t = 0)]
    padding: usize,

    /// Run parallel version (threads + spatial tiles + halo).
    #[arg(long)]
    parallel: bool,

    /// Number of worker threads (parallel mode).
    #[arg(long, default_value_t = 4)]
    threads: usize,

    /// Number of tiles per dimension (parallel mode).
    #[arg(long, default_value_t = 2)]
    tiles_per_dim: usize,

    /// Opciono: putanja do CSV-a (samo kanal 0).
    #[arg(long)]
    save_csv: Option<PathBuf>,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let (h, w, x) = io::load_csv_matrix_f32(&args.input)?;
    let (hp, wp, x_padded) = io::padded_matrix(h, w, &x, args.padding);
    let params = paper_kan::load_params_json(&args.params)?;
    if params.in_features != args.kernel * args.kernel {
        anyhow::bail!(
            "params in_features {} != kernel^2 {}",
            params.in_features,
            args.kernel * args.kernel
        );
    }
    if params.basis_count != params.g + params.k {
        anyhow::bail!(
            "params basis_count {} != G+k {}+{}",
            params.basis_count,
            params.g,
            params.k
        );
    }
    let out_features = params.out_features;
    let g = params.g;
    let k = params.k;

    let t0 = Instant::now();
    let (h_out, w_out, y) = if args.parallel {
        par::run_parallel_tiles_halo(
            hp,
            wp,
            x_padded,
            params,
            args.kernel,
            args.stride,
            args.threads,
            args.tiles_per_dim,
        )
    } else {
        seq::run_sequential(hp, wp, &x_padded, &params, args.kernel, args.stride)
    };
    let dt = t0.elapsed().as_secs_f64();

    let (minv, maxv, mean, std) = stats::output_stats(&y);
    println!(
        "Done in {:.4}s. Output shape: ({}, {}, {}). G={} k={}. Stats: min={:.6} max={:.6} mean={:.6} std={:.6}",
        dt, h_out, w_out, out_features, g, k, minv, maxv, mean, std
    );

    if let Some(path) = args.save_csv {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut s = String::new();
        for i in 0..h_out {
            for j in 0..w_out {
                let base = (i * w_out + j) * out_features;
                let v = y[base]; // channel 0
                if j > 0 {
                    s.push(',');
                }
                s.push_str(&format!("{v}"));
            }
            s.push('\n');
        }
        std::fs::write(&path, s)?;
        println!("Saved output channel0 to {}", path.display());
    }

    Ok(())
}
