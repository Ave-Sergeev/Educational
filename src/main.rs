use std::iter::Iterator;

fn main() {
    let x = vec![1.0, 1.0, 1.0];
    let mut w = vec![0.5, 0.5, 0.5];
    let number_of_iterations: i8 = 10;
    let learning_rate_w = 0.1;

    fn f(x: &[f32]) -> f32 {
        x[0].powi(2) + 2.0 * x[1].powi(2) + 3.0 * x[2].powi(2) + &x[0] * &x[1] + &x[1] * &x[2] + 5.0
    }

    fn gradient_f(x: &[f32]) -> Vec<f32> {
        vec![
            (2.0 * &x[0] + &x[1]),
            (4.0 * &x[1] + &x[0] + &x[2]),
            (6.0 * &x[2] + &x[1]),
        ]
    }

    fn gradient_f_w(w: &[f32], x: &[f32]) -> Vec<f32> {
        let gradient = gradient_f(x);
        let conv_result: Vec<f32> = w.iter().zip(&gradient).map(|(w, g)| w * g).collect();

        x.iter()
            .zip(conv_result.iter())
            .zip(gradient.iter())
            .map(
                |((xi, conv), g)| {
                    g * (xi - (xi - conv))
                }
            )
            .collect()
    }

    for k in 0..number_of_iterations {
        let gradient_w: Vec<f32> = gradient_f_w(&w, &x);

        for (wi, gw) in w.iter_mut().zip(gradient_w.iter()) {
            *wi -= &learning_rate_w * gw;
        }

        let gradient: Vec<f32> = gradient_f(&x);
        let conv_result: Vec<f32> = gradient.iter().zip(w.iter()).map(|(g, w)| g * w).collect();
        let x_new: Vec<f32> = x.iter().zip(conv_result.iter()).map(|(xi, conv)| xi - conv).collect();

        println!("Iteration number {}: W = {:?}, f(x_new) = {:.4}", k + 1, w, f(&x_new));
    }
}
