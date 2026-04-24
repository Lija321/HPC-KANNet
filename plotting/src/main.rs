use anyhow::Context;
use plotters::prelude::*;
use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

const CHART_WIDTH: u32 = 900;
const CHART_HEIGHT: u32 = 520;

#[derive(Debug, Deserialize, Clone)]
struct Row {
    mode: String, // "strong" | "weak"
    #[serde(default)]
    kernel: Option<usize>,
    #[serde(default)]
    workers: Option<usize>,
    #[serde(default)]
    t_serial_sec: Option<f64>,
    #[serde(default)]
    t_seq_sec: Option<f64>,
    #[serde(default)]
    t_par_sec: Option<f64>,
    #[serde(default)]
    speedup: Option<f64>,
}

#[derive(Debug, clap::Parser)]
#[command(name = "kannet_plotting", about = "Plot KANNet scaling results (Rust + Plotters)")]
struct Args {
    /// Rust input CSV (e.g. outputs/scaling_results_rust.csv)
    #[arg(long, default_value = "../outputs/scaling_results_rust.csv")]
    rust_input: PathBuf,

    /// Python input CSV (e.g. outputs/scaling_results.csv)
    #[arg(long, default_value = "../outputs/scaling_results.csv")]
    python_input: PathBuf,

    /// Output directory for PNGs
    #[arg(long, default_value = "../outputs/plots_rust")]
    out_dir: PathBuf,
}

fn read_rows(path: &Path) -> anyhow::Result<Vec<Row>> {
    let mut rdr = csv::Reader::from_path(path).with_context(|| format!("open csv {}", path.display()))?;
    let mut out = Vec::new();
    for rec in rdr.deserialize() {
        let row: Row = rec?;
        out.push(row);
    }
    Ok(out)
}

fn ensure_out_dir(path: &Path) -> anyhow::Result<()> {
    std::fs::create_dir_all(path).with_context(|| format!("create out dir {}", path.display()))?;
    Ok(())
}

fn grouped_strong(rows: &[Row]) -> BTreeMap<usize, Vec<Row>> {
    // group by kernel, default 0 if missing
    let mut m: BTreeMap<usize, Vec<Row>> = BTreeMap::new();
    for r in rows.iter().filter(|r| r.mode == "strong") {
        let k = r.kernel.unwrap_or(0);
        m.entry(k).or_default().push(r.clone());
    }
    for v in m.values_mut() {
        v.sort_by_key(|r| r.workers.unwrap_or(0));
    }
    m
}

fn grouped_weak(rows: &[Row]) -> BTreeMap<usize, Vec<Row>> {
    let mut m: BTreeMap<usize, Vec<Row>> = BTreeMap::new();
    for r in rows.iter().filter(|r| r.mode == "weak") {
        let k = r.kernel.unwrap_or(0);
        m.entry(k).or_default().push(r.clone());
    }
    for v in m.values_mut() {
        v.sort_by_key(|r| r.workers.unwrap_or(0));
    }
    m
}

fn fit_amdahl_fraction(points: &[(f64, f64)]) -> Option<f64> {
    // points: (P, S_measured)
    // Fit f in [0, 1] by brute-force grid search (stable, no extra deps).
    if points.len() < 2 {
        return None;
    }
    let mut best_f = 0.0;
    let mut best_err = f64::INFINITY;
    for step in 0..=2000 {
        let f = (step as f64) / 2000.0;
        let mut err = 0.0;
        for &(p, s) in points {
            let pred = 1.0 / (f + (1.0 - f) / p);
            let d = pred - s;
            err += d * d;
        }
        if err < best_err {
            best_err = err;
            best_f = f;
        }
    }
    Some(best_f)
}

fn gustafson_speedup(p: f64, f: f64) -> f64 {
    // Gustafson's law with serial fraction f (often denoted alpha):
    // S_G(P) = P - f*(P-1)
    p - f * (p - 1.0)
}

fn amdahl_speedup(p: f64, f: f64) -> f64 {
    1.0 / (f + (1.0 - f) / p)
}

/// NTP pravilnik: x = broj jezgara, y = ubrzanje; mereno + idealno skaliranje (S=P) + teorija.
fn plot_ntp_speedup_triple(
    out_path: &Path,
    title: &str,
    measured: &[(f64, f64)],
    ideal: &[(f64, f64)],
    theory: &[(f64, f64)],
    theory_legend: &str,
    width: u32,
    height: u32,
) -> anyhow::Result<()> {
    if measured.is_empty() {
        return Ok(());
    }

    let root = BitMapBackend::new(out_path, (width, height)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut y_max = 1.0_f64;
    for (_, y) in measured.iter().chain(ideal.iter()).chain(theory.iter()) {
        y_max = y_max.max(*y);
    }
    let x_max = measured
        .iter()
        .map(|(x, _)| *x)
        .fold(1.0_f64, f64::max)
        .max(ideal.last().map(|(x, _)| *x).unwrap_or(1.0))
        + 0.5;
    y_max = (y_max * 1.12).max(1.0);
    let x_min = 0.5_f64;
    let y_min = 0.0_f64;

    let mut chart = ChartBuilder::on(&root)
        .caption(title, ("sans-serif", 22).into_font())
        .margin(15)
        .x_label_area_size(45)
        .y_label_area_size(55)
        .build_cartesian_2d(x_min..x_max, y_min..y_max)?;

    chart
        .configure_mesh()
        .x_desc("Broj procesorskih jezgara")
        .y_desc("Ubrzanje")
        .light_line_style(&RGBColor(220, 220, 220))
        .draw()?;

    // Idealno skaliranje (S = P)
    chart
        .draw_series(LineSeries::new(ideal.to_vec(), GREEN.stroke_width(2)))?
        .label("Idealno skaliranje (S = P)")
        .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], GREEN.stroke_width(2)));

    // Teorija (Amdahl ili Gustafson)
    chart
        .draw_series(LineSeries::new(theory.to_vec(), RED.stroke_width(2)))?
        .label(theory_legend)
        .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED.stroke_width(2)));

    // Mereno
    chart
        .draw_series(LineSeries::new(measured.to_vec(), BLUE.stroke_width(3)))?
        .label("Mereno ubrzanje")
        .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE.stroke_width(3)));
    chart.draw_series(
        measured
            .iter()
            .map(|(x, y)| Circle::new((*x, *y), 5, BLUE.filled())),
    )?;

    chart.configure_series_labels().border_style(&BLACK).draw()?;
    root.present()?;
    Ok(())
}

fn plot_from_csv(label: &str, input: &Path, out_dir: &Path) -> anyhow::Result<()> {
    if !input.exists() {
        return Ok(());
    }
    let rows = read_rows(input)?;
    let strong = grouped_strong(&rows);
    let weak = grouped_weak(&rows);

    for (k, rs) in &strong {
        let kout = out_dir.join(format!("kernel{k}"));
        ensure_out_dir(&kout)?;

        let mut sp_pts = Vec::new();
        for r in rs {
            if let (Some(w), Some(v)) = (r.workers, r.speedup) {
                sp_pts.push((w as f64, v));
            }
        }

        // Jako skaliranje: mereno ubrzanje, S=P, Amdahl
        if !sp_pts.is_empty() {
            let f_measured = rs
                .iter()
                .find(|r| r.workers == Some(1))
                .and_then(|r| match (r.t_serial_sec, r.t_seq_sec) {
                    (Some(ts), Some(tseq)) if tseq > 0.0 => Some((ts / tseq).clamp(0.0, 1.0)),
                    _ => None,
                });
            let f_fit = fit_amdahl_fraction(&sp_pts);
            if let Some(f) = f_measured.or(f_fit) {
                sp_pts.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
                let ideal: Vec<(f64, f64)> = sp_pts.iter().map(|(p, _)| (*p, *p)).collect();
                let theory: Vec<(f64, f64)> = sp_pts
                    .iter()
                    .map(|(p, _)| (*p, amdahl_speedup(*p, f)))
                    .collect();
                let slug = label.to_lowercase();
                let out = kout.join(format!("ntp_strong_{slug}_amdahl_kernel{k}.png"));
                let leg = format!("Amdahl (f ≈ {f:.3})");
                plot_ntp_speedup_triple(
                    &out,
                    &format!("Jako skaliranje — {label}, kernel={k}"),
                    &sp_pts,
                    &ideal,
                    &theory,
                    &leg,
                    CHART_WIDTH,
                    CHART_HEIGHT,
                )?;
            }
        }
    }

    for (k, rs) in &weak {
        let kout = out_dir.join(format!("kernel{k}"));
        ensure_out_dir(&kout)?;

        // Slabo skaliranje: P·T(1)/T(P), S=P, Gustafson
        let strong_rows_for_k = grouped_strong(&rows).get(k).cloned().unwrap_or_default();
        let mut sp_pts = Vec::new();
        for r in &strong_rows_for_k {
            if let (Some(w), Some(v)) = (r.workers, r.speedup) {
                sp_pts.push((w as f64, v));
            }
        }
        let f_measured = strong_rows_for_k
            .iter()
            .find(|r| r.workers == Some(1))
            .and_then(|r| match (r.t_serial_sec, r.t_seq_sec) {
                (Some(ts), Some(tseq)) if tseq > 0.0 => Some((ts / tseq).clamp(0.0, 1.0)),
                _ => None,
            });
        if let Some(f) = f_measured.or_else(|| fit_amdahl_fraction(&sp_pts)) {
            let t1 = rs
                .iter()
                .find(|r| r.workers == Some(1))
                .and_then(|r| r.t_par_sec);
            if let Some(t1) = t1 {
                let mut meas = Vec::new();
                for r in rs {
                    if let (Some(w), Some(tp)) = (r.workers, r.t_par_sec) {
                        if tp > 0.0 {
                            let p = w as f64;
                            meas.push((p, p * t1 / tp));
                        }
                    }
                }
                meas.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
                if !meas.is_empty() {
                    let ideal: Vec<(f64, f64)> = meas.iter().map(|(p, _)| (*p, *p)).collect();
                    let theory: Vec<(f64, f64)> = meas
                        .iter()
                        .map(|(p, _)| (*p, gustafson_speedup(*p, f)))
                        .collect();
                    let slug = label.to_lowercase();
                    let out = kout.join(format!("ntp_weak_{slug}_gustafson_kernel{k}.png"));
                    let leg = format!("Gustafson (f ≈ {f:.3})");
                    plot_ntp_speedup_triple(
                        &out,
                        &format!("Slabo skaliranje — {label}, kernel={k}"),
                        &meas,
                        &ideal,
                        &theory,
                        &leg,
                        CHART_WIDTH,
                        CHART_HEIGHT,
                    )?;
                }
            }
        }
    }
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let args = <Args as clap::Parser>::parse();
    ensure_out_dir(&args.out_dir)?;

    plot_from_csv("Rust", &args.rust_input, &args.out_dir)?;
    plot_from_csv("Python", &args.python_input, &args.out_dir)?;

    println!("Wrote plots to {}", args.out_dir.display());
    Ok(())
}

