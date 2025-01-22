fn main() {
    let x: Vec<f64> = vec![1.0, 1.0, 1.0];
    let mut w: Vec<f64> = vec![0.5, 0.5, 0.5];
    let number_of_iterations: i32 = 10;
    let learning_rate_w: f64 = 0.1;

    fn f(x: &[f64]) -> f64 {
        x[0].powi(2) + 2.0 * x[1].powi(2) + 3.0 * x[2].powi(2) + &x[0] * &x[1] + &x[1] * &x[2] + 5.0
    }

    fn gradient_f(x: &[f64]) -> Vec<f64> {
        let mut result: Vec<f64> = Vec::new();

        result.push(2.0 * &x[0] + &x[1]);
        result.push(4.0 * &x[1] + &x[0] + &x[2]);
        result.push(16.0 * &x[2] + &x[1]);
        result
    }

    fn gradient_f_w(w: &[f64], x: &[f64]) -> Vec<f64> {
        let gradient: Vec<f64> = gradient_f(x);
        let conv_result: Vec<f64> = gradient.iter().zip(w.iter()).map(|(g, w)| g * w).collect();

        x.iter()
            .zip(conv_result.iter())
            .zip(gradient.iter())
            .map(|((xi, conv), g)| g * (xi - (xi - conv)))
            .collect()
    }

    for k in 0..number_of_iterations {
        let gradient_w: Vec<f64> = gradient_f_w(&w, &x);

        for (wi, gw) in w.iter_mut().zip(gradient_w.iter()) {
            *wi -= &learning_rate_w * gw;
        }

        let gradient: Vec<f64> = gradient_f(&x);
        let conv_result: Vec<f64> = gradient.iter().zip(w.iter()).map(|(g, w)| g * w).collect();
        let x_new: Vec<f64> = x.iter().zip(conv_result.iter()).map(|(xi, conv)| xi - conv).collect();

        println!("Iteration number {}: W = {:?}, f(x_new) = {:.4}", k + 1, w, f(&x_new));
    }
}
