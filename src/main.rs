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

// Обновляем веса
fn update_weights(w: &mut [f32], gradient_w: &[f32]) {
    for (wi, gw) in w.iter_mut().zip(gradient_w.iter()) {
        *wi -= LEARNING_RATE_W * gw;
    }
}

// Сворачиваем ядро с градиентом
fn conv_grad(x: &[f32], kernel: &[f32], grad: &[f32]) -> Vec<f32> {
    let x_len: usize = x.len();
    let kernel_len: usize = kernel.len();
    let mut result: Vec<f32> = vec![0.0; x_len];

    for i in 0..x_len {
        for j in 0..kernel_len {
            if i + j < x_len {
                result[i] += kernel[j] * grad[i + j];
            } else {
                break;
            }
        }
    }

    result
}

// Генерируем ядро (чем больше отклонение, тем больше будет сглаживание)
fn generate_gauss_kernel(size: usize, sigma: f32) -> Vec<f32> {
    assert!(
        !(size == 0 || sigma <= 0.0),
        "The `size` and `sigma` fields cannot be equal to or less than 0."
    );

    let mut kernel: Vec<f32> = vec![0.0; size];
    let mean: f32 = size as f32 / 2.0;
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
    use approx::assert_relative_eq;

    use super::*;

    // Проверяем корректность работы функции свертки ядра с градиентом
    #[test]
    fn test_conv_grad() {
        let x: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let kernel: Vec<f32> = vec![0.5, 0.3, 0.2];
        let grad: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4, 0.5];

        let result: Vec<f32> = conv_grad(&x, &kernel, &grad);
        println!("result: {:?}", result);

        let expected: Vec<f32> = vec![
            0.1 * 0.5 + 0.2 * 0.3 + 0.3 * 0.2,
            0.2 * 0.5 + 0.3 * 0.3 + 0.4 * 0.2,
            0.3 * 0.5 + 0.4 * 0.3 + 0.5 * 0.2,
            0.4 * 0.5 + 0.5 * 0.3,
            0.5 * 0.5,
        ];

        for (res, exp) in result.iter().zip(expected.iter()) {
            assert!((res - exp).abs() < 1e-6, "Got {}, expected {}", res, exp);
        }
    }

    // Проверяем, что размер возвращаемого ядра соответствует заданному размеру, и значения возвращаемые функцией, совпадают с референсными
    #[test]
    fn test_kernel() {
        let size: usize = 5;
        let sigma: f32 = 1.0;
        let kernel: Vec<f32> = generate_gauss_kernel(size, sigma);

        assert_eq!(kernel.len(), size);
        assert_eq!(kernel[0], 0.017873362);
        assert_eq!(kernel[1], 0.13206728);
        assert_eq!(kernel[2], 0.35899606);
        assert_eq!(kernel[3], 0.35899606);
        assert_eq!(kernel[4], 0.13206728);
        assert!((kernel.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }

    // Проверяем, что сумма всех элементов ядра равна 1 (нормализация).
    #[test]
    fn test_kernel_normalization() {
        let size: usize = 9;
        let sigma: f32 = 2.0;
        let kernel: Vec<f32> = generate_gauss_kernel(size, sigma);
        let sum: f32 = kernel.iter().sum();

        assert_relative_eq!(sum, 1.0, epsilon = 1e-6);
    }

    // Проверяем, что максимальное значение находится в центре ядра.
    #[test]
    fn test_kernel_peak() {
        let size: usize = 5;
        let sigma: f32 = 1.0;
        let kernel: Vec<f32> = generate_gauss_kernel(size, sigma);

        assert_eq!(kernel.iter().max_by(|a, b| a.partial_cmp(b).unwrap()), Some(&kernel[size / 2]));
    }

    // Сравниваем ядра с разными значениями сигмы, чтобы убедиться, что большая сигма дает более гладкое распределение.
    #[test]
    fn test_different_sigmas() {
        let size: usize = 7;
        let sigma1: f32 = 1.0;
        let sigma2: f32 = 2.0;
        let kernel1: Vec<f32> = generate_gauss_kernel(size, sigma1);
        let kernel2: Vec<f32> = generate_gauss_kernel(size, sigma2);

        assert!(kernel1[size / 2] > kernel2[size / 2]);
    }

    // Проверяем обработку некорректных данных
    #[test]
    #[should_panic]
    fn test_zero_size() {
        generate_gauss_kernel(0, 1.0);
    }

    // Проверяем обработку некорректных данных
    #[test]
    #[should_panic]
    fn test_negative_sigma() {
        generate_gauss_kernel(5, -1.0);
    }

    // Проверяем обработку некорректных данных
    #[test]
    #[should_panic]
    fn test_zero_sigma() {
        generate_gauss_kernel(5, 0.0);
    }
}
