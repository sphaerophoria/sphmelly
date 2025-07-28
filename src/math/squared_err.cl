__kernel void squared_err(
        __global float* a,
        __global float* b,
        __global float* output,
        uint n
) {
    uint global_id = get_global_id(0);
    if (global_id >= n) return;
    float diff = a[global_id] - b[global_id];
    output[global_id] = diff * diff;
}

__kernel void squared_err_grad(
        __global float* downstream_grad,
        __global float* a,
        __global float* b,
        __global float* a_grad,
        __global float* b_grad,
        uint n
) {
    uint global_id = get_global_id(0);
    if (global_id >= n) return;

    float diff = a[global_id] - b[global_id];
    // 2 * diff from power rule, multiplied by downstream effect
    float grad = downstream_grad[global_id] * 2 * diff;

    // Merging power and subtraction gradients, any change in a is the same as
    // if we were just doing a normal power rule, any change in b has an inverse
    // effect on the difference, so invert the gradient
    a_grad[global_id] = grad;
    b_grad[global_id] = -grad;
}

