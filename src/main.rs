use std::iter::zip;
use std::iter::Iterator;

const KERNEL: [f32; 3] = [0.1, 0.2, 0.3];
const NUM_ITERATIONS: usize = 30;
const LEARNING_RATE_W: f32 = 0.1;

fn main() {
    let mut x: Vec<f32> = vec![1.0, 1.0, 1.0];
    let mut w: Vec<f32> = vec![1.0, 2.0, 3.0];

    // Оптимизация W
    for k in 1..=NUM_ITERATIONS {
        let gradient_w: Vec<f32> = gradient_f_w(&w, &x);
        update_weights(&mut w, &gradient_w);

        // Вычисляем новое значение после свёртки
        let gradient: Vec<f32> = gradient_f(&x);
        let conv_result: Vec<f32> = conv_grad(&w, &KERNEL, &gradient);

        // Обновляем x
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

// Градиент функции по параметрам w
fn gradient_f_w(w: &[f32], x: &[f32]) -> Vec<f32> {
    let gradient: Vec<f32> = gradient_f(x);
    let conv_result: Vec<f32> = conv_grad(w, &KERNEL, &gradient);

    zip(x.iter(), conv_result.iter())
        .zip(gradient.iter())
        .map(|((_, conv), g)| g * conv)
        .collect()
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conv_grad() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let kernel = vec![0.5, 0.3, 0.2];
        let grad = vec![0.1, 0.2, 0.3, 0.4, 0.5];

        let result = conv_grad(&x, &kernel, &grad);
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
}
