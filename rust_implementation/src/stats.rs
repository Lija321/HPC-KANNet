pub fn output_stats(y: &[f32]) -> (f32, f32, f32, f32) {
    let mut minv = f32::INFINITY;
    let mut maxv = f32::NEG_INFINITY;
    let mut sum = 0.0f64;
    for &v in y {
        if v < minv {
            minv = v;
        }
        if v > maxv {
            maxv = v;
        }
        sum += v as f64;
    }
    let mean = (sum / (y.len() as f64)) as f32;
    let mut var = 0.0f64;
    for &v in y {
        let d = v as f64 - mean as f64;
        var += d * d;
    }
    let std = ((var / (y.len() as f64)).sqrt()) as f32;
    (minv, maxv, mean, std)
}

