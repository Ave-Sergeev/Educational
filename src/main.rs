use std::iter::Iterator;

const NUM_ITERATIONS: usize = 30;
const LEARNING_RATE_W: f32 = 0.1;

fn main() {
    let mut x: Vec<f32> = vec![1.0, 1.0, 1.0];
    let mut w: Vec<f32> = vec![1.0, 2.0, 3.0];
    let kernel: Vec<f32> = generate_gauss_kernel(x.len(), 1.0);

    // Оптимизация W
    for k in 1..=NUM_ITERATIONS {
        let gradient_w: Vec<f32> = gradient_f_w(&w, &x);
        update_weights(&mut w, &gradient_w);

        // Вычисляем новое значение после свёртки
        let gradient: Vec<f32> = gradient_f(&x);
        let conv_result: Vec<f32> = conv_grad(&w, &kernel, &gradient);

        // Обновляем X
        for i in 0..x.len() {
            x[i] -= conv_result[i];
        }

        println!("Iteration number {}: W = {:?}, f(x) = {:.4}", k, w, f(&x));
    }
}

// Задаем функцию f(x) над которой экспериментируем
fn f(x: &[f32]) -> f32 {
    x[0].powi(2) + 2.0 * x[1].powi(2) + 3.0 * x[2].powi(2) + x[0] * x[1] + x[1] * x[2] + 5.0
}

// Вычисляем градиент функции f(x) (берем частные производные)
fn gradient_f(x: &[f32]) -> Vec<f32> {
    vec![2.0 * x[0] + x[1], 4.0 * x[1] + x[0] + x[2], 6.0 * x[2] + x[1]]
}

// Градиент функции по параметрам W
fn gradient_f_w(w: &[f32], x: &[f32]) -> Vec<f32> {
    let gradient: Vec<f32> = gradient_f(x);
    w.iter().zip(&gradient).map(|(w, g)| w * g).collect()
}

// Обновление весов
fn update_weights(w: &mut [f32], gradient_w: &[f32]) {
    for (wi, gw) in w.iter_mut().zip(gradient_w.iter()) {
        *wi -= LEARNING_RATE_W * gw; // возможно * (-gw), не уверен
    }
}

// Свертка ядра с градиентом
fn conv_grad(x: &[f32], kernel: &[f32], grad: &[f32]) -> Vec<f32> {
    let n = x.len();
    let m = kernel.len();
    let mut result = vec![0.0; n];

    for i in 0..n {
        for j in 0..m {
            if i + j < n {
                result[i] += kernel[j] * grad[i + j];
            } else {
                break;
            }
        }
    }

    result
}

// Генерация ядра (чем больше отклонение, тем больше будет сглаживание)
fn generate_gauss_kernel(size: usize, sigma: f32) -> Vec<f32> {
    let mut kernel = vec![0.0; size];
    let mean = size as f32 / 2.0;
    let sum: f32 = (0..size)
        .map(|i| {
            let x = i as f32 - mean;
            let value = (-0.5 * (x / sigma).powi(2)).exp();
            kernel[i] = value;
            value
        })
        .sum();

    for i in 0..size {
        kernel[i] /= sum;
    }

    kernel
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conv_grad() {
        let x: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let kernel: Vec<f32> = vec![0.5, 0.3, 0.2];
        let grad: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4, 0.5];

        let result: Vec<f32> = conv_grad(&x, &kernel, &grad);
        println!("result: {:?}", result);

        let expected = vec![
            0.1 * 0.5 + 0.2 * 0.3 + 0.3 * 0.2,
            0.2 * 0.5 + 0.3 * 0.3 + 0.4 * 0.2,
            0.3 * 0.5 + 0.4 * 0.3 + 0.5 * 0.2,
            0.4 * 0.5 + 0.5 * 0.3,
            0.5 * 0.5,
        ];
        println!("expected: {:?}", expected);

        for (res, exp) in result.iter().zip(expected.iter()) {
            assert!((res - exp).abs() < 1e-6, "Got {}, expected {}", res, exp);
        }
    }

    #[test]
    fn test_gauss_kernel() {
        let kernel: Vec<f32> = generate_gauss_kernel(5, 1.0);

        assert_eq!(kernel.len(), 5);
        assert_eq!(kernel[0], 0.017873362);
        assert_eq!(kernel[1], 0.13206728);
        assert_eq!(kernel[2], 0.35899606);
        assert_eq!(kernel[3], 0.35899606);
        assert_eq!(kernel[4], 0.13206728);
    }
}
