use clap::Parser;
use std::path::{Path, PathBuf};
use std::time::Instant;

use rust_implementation::{io, par, paper_kan, seq};

#[derive(Parser, Debug)]
#[command(name = "benchmark_scaling", about = "Strong/weak scaling experiments (Rust)")]
struct Args {
    /// Directory containing input_matrix_{N}.csv files.
    #[arg(long, default_value = "../data")]
    data_dir: PathBuf,

    /// Params JSON path.
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

    /// Comma-separated workers list, e.g. "1,2,4,8".
    #[arg(long, default_value = "1,2,4,8")]
    workers: String,

    /// Tiles per dimension (parallel).
    #[arg(long, default_value_t = 2)]
    tiles_per_dim: usize,

    /// Broj nezavisnih merenja po tački (NTP: ~30; mean, std, Tukey outlier-i).
    #[arg(long, default_value_t = 30, alias = "repeats")]
    runs: usize,

    /// Strong scaling input size (expects input_matrix_{N}.csv).
    #[arg(long, default_value_t = 128)]
    strong_size: usize,

    /// Weak scaling base size (expects input_matrix_{N}.csv for generated sizes).
    #[arg(long, default_value_t = 64)]
    weak_base: usize,

    /// Base output directory.
    /// Results are written to <out-dir>/rust_scaling.csv
    #[arg(
        long,
        default_value = "../outputs/scaling/kernel{kernel}/strong{strong_size}/weakbase{weak_base}"
    )]
    out_dir: PathBuf,
}

fn parse_workers(s: &str) -> anyhow::Result<Vec<usize>> {
    let mut v = Vec::new();
    for part in s.split(',') {
        let p = part.trim();
        if p.is_empty() {
            continue;
        }
        v.push(p.parse::<usize>()?);
    }
    if v.is_empty() {
        anyhow::bail!("workers list is empty");
    }
    Ok(v)
}

fn measure_samples(mut f: impl FnMut() -> anyhow::Result<()>, n: usize) -> anyhow::Result<Vec<f64>> {
    let mut v = Vec::with_capacity(n.max(1));
    for _ in 0..n.max(1) {
        let t0 = Instant::now();
        f()?;
        v.push(t0.elapsed().as_secs_f64());
    }
    Ok(v)
}

fn mean_std_sample(xs: &[f64]) -> (f64, f64) {
    let n = xs.len();
    if n == 0 {
        return (f64::NAN, f64::NAN);
    }
    let m = xs.iter().sum::<f64>() / n as f64;
    if n < 2 {
        return (m, 0.0);
    }
    let var: f64 = xs.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (n as f64 - 1.0);
    (m, var.sqrt())
}

fn percentile_linear(sorted: &[f64], p: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return f64::NAN;
    }
    if n == 1 {
        return sorted[0];
    }
    let rank = (n - 1) as f64 * p.clamp(0.0, 1.0);
    let lo = rank.floor() as usize;
    let hi = rank.ceil() as usize;
    let hi = hi.min(n - 1);
    let frac = rank - lo as f64;
    if lo == hi {
        sorted[lo]
    } else {
        sorted[lo] * (1.0 - frac) + sorted[hi] * frac
    }
}

fn tukey_outliers_count(xs: &[f64]) -> usize {
    if xs.len() < 4 {
        return 0;
    }
    let mut v = xs.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let q1 = percentile_linear(&v, 0.25);
    let q3 = percentile_linear(&v, 0.75);
    let iqr = q3 - q1;
    if !iqr.is_finite() || iqr <= 0.0 {
        return 0;
    }
    let lo = q1 - 1.5 * iqr;
    let hi = q3 + 1.5 * iqr;
    xs.iter().filter(|&&t| t < lo || t > hi).count()
}

fn load_input(data_dir: &Path, size: usize) -> anyhow::Result<(usize, usize, Vec<f32>)> {
    let path = data_dir.join(format!("input_matrix_{size}.csv"));
    io::load_csv_matrix_f32(&path)
}

fn weak_size(base: usize, workers: usize) -> usize {
    let scale = (workers.max(1) as f64).sqrt();
    let mut sz = ((base as f64) * scale).round() as usize;
    // Match Python rounding to avoid missing pre-generated files.
    if sz % 8 != 0 {
        sz = ((sz as f64) / 8.0).round() as usize * 8;
    }
    sz.max(8)
}

fn write_csv(path: &Path, rows: &[std::collections::BTreeMap<String, String>]) -> anyhow::Result<()> {
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
            let v = r.get(h).cloned().unwrap_or_default();
            // Minimal CSV escaping (values are numeric/simple)
            out.push_str(&v);
        }
        out.push('\n');
    }
    std::fs::write(path, out)?;
    Ok(())
}

fn write_markdown_tables(out_csv: &Path, rows: &[std::collections::BTreeMap<String, String>]) -> anyhow::Result<()> {
    let stem = out_csv
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("scaling");
    let md_path = out_csv.with_file_name(format!("{stem}_tables.md"));
    let mut lines: Vec<String> = vec![
        "# Tabele merenja (mean / std / Tukey outlier-i)\n\n".into(),
        "*Ubrzanje i efikasnost od srednjih vremena (`t_seq_sec` / `t_par_sec`).*\n\n".into(),
    ];

    let strong: Vec<_> = rows.iter().filter(|r| r.get("mode").map(|s| s.as_str()) == Some("strong")).collect();
    if !strong.is_empty() {
        lines.push("## Jako skaliranje\n\n".into());
        let keys = [
            "workers",
            "n_runs",
            "t_serial_sec",
            "t_serial_std_sec",
            "t_serial_outliers",
            "t_seq_sec",
            "t_seq_std_sec",
            "t_seq_outliers",
            "t_par_sec",
            "t_par_std_sec",
            "t_par_outliers",
            "speedup",
            "efficiency",
        ];
        lines.push(format!("| {} |\n", keys.join(" | ")));
        lines.push(format!("|{}|\n", keys.iter().map(|_| "---").collect::<Vec<_>>().join("|")));
        let mut sorted = strong.clone();
        sorted.sort_by_key(|r| r.get("workers").and_then(|s| s.parse::<usize>().ok()).unwrap_or(0));
        for r in sorted {
            let cells: Vec<String> = keys.iter().map(|k| r.get(*k).cloned().unwrap_or_default()).collect();
            lines.push(format!("| {} |\n", cells.join(" | ")));
        }
        lines.push("\n".into());
    }

    let weak: Vec<_> = rows.iter().filter(|r| r.get("mode").map(|s| s.as_str()) == Some("weak")).collect();
    if !weak.is_empty() {
        lines.push("## Slabo skaliranje (paralelno vreme)\n\n".into());
        let keys = ["workers", "H", "W", "n_runs", "t_par_sec", "t_par_std_sec", "t_par_outliers"];
        lines.push(format!("| {} |\n", keys.join(" | ")));
        lines.push(format!("|{}|\n", keys.iter().map(|_| "---").collect::<Vec<_>>().join("|")));
        let mut sorted = weak.clone();
        sorted.sort_by_key(|r| r.get("workers").and_then(|s| s.parse::<usize>().ok()).unwrap_or(0));
        for r in sorted {
            let cells: Vec<String> = keys.iter().map(|k| r.get(*k).cloned().unwrap_or_default()).collect();
            lines.push(format!("| {} |\n", cells.join(" | ")));
        }
    }

    std::fs::write(&md_path, lines.concat())?;
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let workers = parse_workers(&args.workers)?;

    let params = paper_kan::load_params_json(&args.params)?;
    if params.in_features != args.kernel * args.kernel {
        anyhow::bail!(
            "params in_features {} != kernel^2 {}",
            params.in_features,
            args.kernel * args.kernel
        );
    }
    if params.basis_count != params.g + params.k {
        anyhow::bail!("params basis_count != G+k");
    }

    let mut rows: Vec<std::collections::BTreeMap<String, String>> = Vec::new();

    // Strong scaling (fixed input, compare seq vs par).
    let (h, w, x) = load_input(&args.data_dir, args.strong_size)?;
    // conv2d-style: measure "serial/setup" part (padding + shape math) separately
    let serial_samples = measure_samples(
        || {
            let (hp, wp, _x_padded) = io::padded_matrix(h, w, &x, args.padding);
            let _h_out = (hp.saturating_sub(args.kernel)) / args.stride + 1;
            let _w_out = (wp.saturating_sub(args.kernel)) / args.stride + 1;
            Ok(())
        },
        args.runs,
    )?;
    let (sm, ss) = mean_std_sample(&serial_samples);
    let so = tukey_outliers_count(&serial_samples);

    let (hp, wp, x_padded) = io::padded_matrix(h, w, &x, args.padding);

    let seq_samples = measure_samples(
        || {
            let _ = seq::run_sequential(hp, wp, &x_padded, &params, args.kernel, args.stride);
            Ok(())
        },
        args.runs,
    )?;
    let (qm, qs) = mean_std_sample(&seq_samples);
    let qo = tukey_outliers_count(&seq_samples);

    for &pcount in &workers {
        let par_samples = measure_samples(
            || {
                let _ = par::run_parallel_tiles_halo(
                    hp,
                    wp,
                    x_padded.clone(),
                    params.clone(),
                    args.kernel,
                    args.stride,
                    pcount,
                    args.tiles_per_dim,
                );
                Ok(())
            },
            args.runs,
        )?;
        let (pm, ps) = mean_std_sample(&par_samples);
        let po = tukey_outliers_count(&par_samples);
        let speedup = qm / pm;
        let eff = speedup / (pcount as f64);

        let mut r = std::collections::BTreeMap::new();
        r.insert("mode".into(), "strong".into());
        r.insert("H".into(), h.to_string());
        r.insert("W".into(), w.to_string());
        r.insert("kernel".into(), args.kernel.to_string());
        r.insert("stride".into(), args.stride.to_string());
        r.insert("padding".into(), args.padding.to_string());
        r.insert("tiles_per_dim".into(), args.tiles_per_dim.to_string());
        r.insert("workers".into(), pcount.to_string());
        r.insert("n_runs".into(), args.runs.to_string());
        r.insert("t_serial_sec".into(), format!("{sm:.6}"));
        r.insert("t_serial_std_sec".into(), format!("{ss:.6}"));
        r.insert("t_serial_outliers".into(), so.to_string());
        r.insert("t_seq_sec".into(), format!("{qm:.6}"));
        r.insert("t_seq_std_sec".into(), format!("{qs:.6}"));
        r.insert("t_seq_outliers".into(), qo.to_string());
        r.insert("t_par_sec".into(), format!("{pm:.6}"));
        r.insert("t_par_std_sec".into(), format!("{ps:.6}"));
        r.insert("t_par_outliers".into(), po.to_string());
        r.insert("speedup".into(), format!("{speedup:.6}"));
        r.insert("efficiency".into(), format!("{eff:.6}"));
        rows.push(r);
    }

    // Weak scaling (input scales with workers; measure parallel time).
    for &pcount in &workers {
        let sz = weak_size(args.weak_base, pcount);
        let (h, w, x) = load_input(&args.data_dir, sz)?;
        let (hp, wp, x_padded) = io::padded_matrix(h, w, &x, args.padding);

        let par_samples = measure_samples(
            || {
                let _ = par::run_parallel_tiles_halo(
                    hp,
                    wp,
                    x_padded.clone(),
                    params.clone(),
                    args.kernel,
                    args.stride,
                    pcount,
                    args.tiles_per_dim,
                );
                Ok(())
            },
            args.runs,
        )?;
        let (pm, ps) = mean_std_sample(&par_samples);
        let po = tukey_outliers_count(&par_samples);

        let mut r = std::collections::BTreeMap::new();
        r.insert("mode".into(), "weak".into());
        r.insert("H".into(), h.to_string());
        r.insert("W".into(), w.to_string());
        r.insert("kernel".into(), args.kernel.to_string());
        r.insert("stride".into(), args.stride.to_string());
        r.insert("padding".into(), args.padding.to_string());
        r.insert("tiles_per_dim".into(), args.tiles_per_dim.to_string());
        r.insert("workers".into(), pcount.to_string());
        r.insert("n_runs".into(), args.runs.to_string());
        r.insert("t_par_sec".into(), format!("{pm:.6}"));
        r.insert("t_par_std_sec".into(), format!("{ps:.6}"));
        r.insert("t_par_outliers".into(), po.to_string());
        rows.push(r);
    }

    let mut out_dir_s = args.out_dir.to_string_lossy().to_string();
    out_dir_s = out_dir_s.replace("{kernel}", &args.kernel.to_string());
    out_dir_s = out_dir_s.replace("{strong_size}", &args.strong_size.to_string());
    out_dir_s = out_dir_s.replace("{weak_base}", &args.weak_base.to_string());
    let out_dir = PathBuf::from(out_dir_s);
    let out_path = out_dir.join("rust_scaling.csv");
    write_csv(&out_path, &rows)?;
    write_markdown_tables(&out_path, &rows)?;
    let md_path = out_path.with_file_name(format!(
        "{}_tables.md",
        out_path.file_stem().and_then(|s| s.to_str()).unwrap_or("rust_scaling")
    ));
    println!("Wrote {} rows to {}", rows.len(), out_path.display());
    println!("Wrote tables to {}", md_path.display());
    Ok(())
}

