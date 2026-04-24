use csv::ReaderBuilder;
use ndarray::Array2;
use std::fs::File;
use std::path::Path;
use std::{thread, time};

extern crate matrix_display;
use matrix_display::*;

const SLEEP_MS: u64 = 250;

#[derive(Debug, clap::Parser)]
#[command(name = "kannet_par_viz", about = "Terminal animation for KANNet parallel tiling")]
struct Args {
    /// Path to par_visualization_* directory produced by python parallel.py --viz
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

fn display_matrix(matrix: &Array2<f64>, rows: usize, cols: usize, halo: Option<usize>) {
    let format = Format::new(5, 5);
    let board = matrix
        .iter()
        .enumerate()
        .map(|(idx, &x)| {
            let mut ansi_fg = 0;
            let mut ansi_bg = 70;
            if let Some(h) = halo {
                let i = idx / cols;
                let j = idx % cols;
                // highlight halo border with bright background
                if i < h || j < h || i + h >= rows || j + h >= cols {
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

fn display_matrix_with_grid(matrix: &Array2<f64>, rows: usize, cols: usize, tiles_per_dim: usize) {
    let format = Format::new(5, 5);
    let ti = tiles_per_dim.max(1);
    let tile_h = (rows + ti - 1) / ti;
    let tile_w = (cols + ti - 1) / ti;

    let board = matrix
        .iter()
        .enumerate()
        .map(|(idx, &x)| {
            let i = idx / cols;
            let j = idx % cols;
            let mut ansi_fg = 0;
            let mut ansi_bg = 70;
            // highlight tile boundaries
            if i % tile_h == 0 || j % tile_w == 0 {
                ansi_bg = 7;
                ansi_fg = 33;
            }
            cell::Cell::new((x * 100.0).round() / 100.0, ansi_fg, ansi_bg)
        })
        .collect::<Vec<_>>();
    let data = matrix::Matrix::new(rows, board);
    let display = MatrixDisplay::new(format, data);
    display.print(&mut std::io::stdout(), &style::BordersStyle::Thin);
}

fn parse_from_dir(dir: &str, key: &str) -> Option<usize> {
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
    let kernel = parse_from_dir(&dir, "k").unwrap_or(3);
    let halo = kernel.saturating_sub(1);
    let tiles_per_dim = parse_from_dir(&dir, "t").unwrap_or(2);

    let input = read_csv(format!("{dir}/input/input_matrix.csv"))?;

    let mut tiles_in = Vec::new();
    for entry in std::fs::read_dir(format!("{dir}/tiles_input"))? {
        let e = entry?;
        let name = e.file_name().to_string_lossy().to_string();
        if name.starts_with("tile_") && name.ends_with(".csv") {
            tiles_in.push(name);
        }
    }
    tiles_in.sort_by_key(|n| n.trim_start_matches("tile_").trim_end_matches(".csv").parse::<usize>().unwrap_or(0));

    clear();
    println!("==============INPUT MATRIX================");
    display_matrix(&input, input.nrows(), input.ncols(), None);
    thread::sleep(sleep);

    for (i, t) in tiles_in.iter().enumerate() {
        clear();
        println!("==============TILE INPUT {}================", i);
        let tile = read_csv(format!("{dir}/tiles_input/{t}"))?;
        display_matrix(&tile, tile.nrows(), tile.ncols(), Some(halo));
        thread::sleep(sleep);

        println!("\n==============TILE OUTPUT {} (channel)================", i);
        let out = read_csv(format!("{dir}/tiles_output/{t}"))?;
        display_matrix(&out, out.nrows(), out.ncols(), None);
        thread::sleep(sleep);
    }

    // Final merged output (optional, only if dumped by python).
    let merged_path = format!("{dir}/output/output_matrix.csv");
    if Path::new(&merged_path).exists() {
        clear();
        println!("==============MERGED OUTPUT (channel)================");
        let out = read_csv(merged_path)?;
        display_matrix_with_grid(&out, out.nrows(), out.ncols(), tiles_per_dim);
        thread::sleep(sleep);
    }

    Ok(())
}
