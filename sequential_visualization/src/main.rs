use csv::ReaderBuilder;
use ndarray::Array2;
use std::fs::File;
use std::path::Path;
use std::{thread, time};

extern crate matrix_display;
use matrix_display::*;

const SLEEP_MS: u64 = 150;

#[derive(Debug, clap::Parser)]
#[command(name = "kannet_seq_viz", about = "Terminal animation for KANNet sequential run")]
struct Args {
    /// Path to seq_visualization_* directory produced by python sequential.py --viz
    #[arg(long)]
    dir: String,
}

fn read_csv<P: AsRef<Path>>(filename: P) -> anyhow::Result<Array2<f64>> {
    let file = File::open(&filename)?;
    let mut rdr = ReaderBuilder::new().has_headers(false).from_reader(file);
    let mut rows = Vec::new();
    for result in rdr.records() {
        let record = result?;
        let row: Result<Vec<f64>, _> = record.iter().map(|f| f.parse::<f64>()).collect();
        rows.push(row?);
    }
    let num_cols = rows.get(0).map_or(0, |r| r.len());
    let flat: Vec<f64> = rows.into_iter().flatten().collect();
    let num_rows = if num_cols == 0 { 0 } else { flat.len() / num_cols };
    Ok(Array2::from_shape_vec((num_rows, num_cols), flat)?)
}

fn clear() {
    print!("{}[2J", 27 as char);
}

fn display_matrix(
    matrix: &Array2<f64>,
    rows: usize,
    cols: usize,
    highlight_cell: Option<(usize, usize)>,
    highlight_rect: Option<(usize, usize, usize, usize)>, // (i0,j0,h,w)
) {
    let format = Format::new(5, 5);
    let board = matrix
        .iter()
        .enumerate()
        .map(|(idx, &x)| {
            let mut ansi_fg = 0;
            let mut ansi_bg = 70;
            let i = idx / cols;
            let j = idx % cols;

            if let Some((hi, hj)) = highlight_cell {
                if i == hi && j == hj {
                    ansi_bg = 7;
                    ansi_fg = 33;
                }
            }

            if let Some((i0, j0, h, w)) = highlight_rect {
                if i >= i0 && i < i0 + h && j >= j0 && j < j0 + w {
                    ansi_bg = 7;
                    ansi_fg = 33;
                }
            }
            cell::Cell::new((x * 100.0).round() / 100.0, ansi_fg, ansi_bg)
        })
        .collect::<Vec<_>>();
    let data = matrix::Matrix::new(rows, board);
    let display = MatrixDisplay::new(format, data);
    display.print(&mut std::io::stdout(), &style::BordersStyle::Thin);
}

fn parse_from_dir(dir: &str, key: &str) -> Option<usize> {
    // crude parse: "..._k3_..."; key = "k" or "s"
    let needle = format!("_{key}");
    let pos = dir.find(&needle)?;
    let rest = &dir[(pos + needle.len())..];
    let num: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
    num.parse::<usize>().ok()
}

fn main() -> anyhow::Result<()> {
    let args = <Args as clap::Parser>::parse();
    let dir = args.dir;
    let sleep = time::Duration::from_millis(SLEEP_MS);

    let input = read_csv(format!("{dir}/input/input_matrix.csv"))?;
    let in_rows = input.nrows();
    let in_cols = input.ncols();

    let kernel = parse_from_dir(&dir, "k").unwrap_or(3);
    let stride = parse_from_dir(&dir, "s").unwrap_or(1);

    // Determine number of steps by counting step_*.csv files.
    let mut steps = Vec::new();
    for entry in std::fs::read_dir(format!("{dir}/kan"))? {
        let e = entry?;
        let name = e.file_name().to_string_lossy().to_string();
        if name.starts_with("step_") && name.ends_with(".csv") {
            steps.push(name);
        }
    }
    steps.sort_by_key(|n| n.trim_start_matches("step_").trim_end_matches(".csv").parse::<usize>().unwrap_or(0));

    clear();
    println!("==============INPUT MATRIX ON START================");
    display_matrix(&input, in_rows, in_cols, None, None);
    thread::sleep(sleep);

    // Output size inferred from first step
    let first = read_csv(format!("{dir}/kan/{}", steps[0]))?;
    let out_rows = first.nrows();
    let out_cols = first.ncols();

    for (frame_idx, fname) in steps.iter().enumerate() {
        clear();
        println!("==============INPUT MATRIX================");
        let oi = frame_idx / out_cols;
        let oj = frame_idx % out_cols;
        let i0 = oi * stride;
        let j0 = oj * stride;
        display_matrix(
            &input,
            in_rows,
            in_cols,
            None,
            Some((i0, j0, kernel, kernel)),
        );

        println!("\n==============KAN OUTPUT STEP {}================", frame_idx);
        let out = read_csv(format!("{dir}/kan/{fname}"))?;
        // highlight the current output cell
        display_matrix(&out, out_rows, out_cols, Some((oi, oj)), None);
        thread::sleep(sleep);
    }

    Ok(())
}
