use clap::Parser;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use rust_implementation::{io, par, paper_kan, seq};

#[derive(Parser, Debug)]
#[command(name = "benchmark_sizes_kernels", about = "Time vs input size and kernel size (Rust)")]
struct Args {
    #[arg(long, default_value = "../data")]
    data_dir: PathBuf,

    #[arg(long, default_value = "../data")]
    params_base: PathBuf,

    #[arg(long, default_value = "16,24,32,40,48,56,64,80,96,112,128")]
    sizes: String,

    #[arg(long, default_value = "2,3,5,7")]
    kernels: String,

    #[arg(long, default_value_t = 4)]
    threads: usize,

    #[arg(long, default_value_t = 2)]
    tiles_per_dim: usize,

    /// Base output directory.
    /// Results are written to <out-dir>/sizes_kernels.csv
    #[arg(long, default_value = "../outputs/sizes_kernels/rust/workers{threads}")]
    out_dir: PathBuf,
}

fn parse_list(s: &str) -> anyhow::Result<Vec<usize>> {
    let mut v = Vec::new();
    for part in s.split(',') {
        let p = part.trim();
        if p.is_empty() {
            continue;
        }
        v.push(p.parse::<usize>()?);
    }
    Ok(v)
}

fn measure_best(mut f: impl FnMut() -> anyhow::Result<()>, repeats: usize) -> anyhow::Result<f64> {
    let mut best = f64::INFINITY;
    for _ in 0..repeats.max(1) {
        let t0 = Instant::now();
        f()?;
        let dt = t0.elapsed().as_secs_f64();
        best = best.min(dt);
    }
    Ok(best)
}

fn write_csv(path: &Path, rows: &[BTreeMap<String, String>]) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut headers = std::collections::BTreeSet::<String>::new();
    for r in rows {
        for k in r.keys() {
            headers.insert(k.clone());
        }
    }
    let headers: Vec<String> = headers.into_iter().collect();
    let mut out = String::new();
    out.push_str(&headers.join(","));
    out.push('\n');
    for r in rows {
        for (i, h) in headers.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            out.push_str(r.get(h).cloned().unwrap_or_default().as_str());
        }
        out.push('\n');
    }
    std::fs::write(path, out)?;
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let stride = 1usize;
    let padding = 0usize;
    let repeats = 1usize;

    let sizes = parse_list(&args.sizes)?;
    let kernels = parse_list(&args.kernels)?;

    let mut rows: Vec<BTreeMap<String, String>> = Vec::new();
    for &k in &kernels {
        let params_path = args
            .params_base
            .join(format!("paper_kan_params_in{}_out8_G5_k3_kernel{}/params.json", k * k, k));
        let params = paper_kan::load_params_json(&params_path)?;
        if params.in_features != k * k {
            anyhow::bail!("params in_features mismatch for kernel={k}");
        }

        for &n in &sizes {
            let input_path = args.data_dir.join(format!("input_matrix_{n}.csv"));
            let (h, w, x) = io::load_csv_matrix_f32(&input_path)?;
            let (hp, wp, x_pad) = io::padded_matrix(h, w, &x, padding);

            let t_seq = measure_best(
                || {
                    let _ = seq::run_sequential(hp, wp, &x_pad, &params, k, stride);
                    Ok(())
                },
                repeats,
            )?;

            let t_par = measure_best(
                || {
                    let _ = par::run_parallel_tiles_halo(
                        hp,
                        wp,
                        x_pad.clone(),
                        params.clone(),
                        k,
                        stride,
                        args.threads,
                        args.tiles_per_dim,
                    );
                    Ok(())
                },
                repeats,
            )?;

            let speedup = t_seq / t_par;
            let eff = speedup / (args.threads as f64);

            let mut r = BTreeMap::new();
            r.insert("language".into(), "rust".into());
            r.insert("H".into(), h.to_string());
            r.insert("W".into(), w.to_string());
            r.insert("kernel".into(), k.to_string());
            r.insert("stride".into(), stride.to_string());
            r.insert("padding".into(), padding.to_string());
            r.insert("workers".into(), args.threads.to_string());
            r.insert("tiles_per_dim".into(), args.tiles_per_dim.to_string());
            r.insert("t_seq_sec".into(), format!("{t_seq:.6}"));
            r.insert("t_par_sec".into(), format!("{t_par:.6}"));
            r.insert("speedup".into(), format!("{speedup:.6}"));
            r.insert("efficiency".into(), format!("{eff:.6}"));
            rows.push(r);
        }
    }

    let out_dir_s = args.out_dir.to_string_lossy().to_string();
    let out_dir_s = out_dir_s.replace("{threads}", &args.threads.to_string());
    let out_dir = PathBuf::from(out_dir_s);
    let out_path = out_dir.join("sizes_kernels.csv");
    write_csv(&out_path, &rows)?;
    println!("Wrote {} rows to {}", rows.len(), out_path.display());
    Ok(())
}

