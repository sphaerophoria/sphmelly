constant float B1 = 0.9;
constant float B2 = 0.999;
constant float eps = 1e-8;

__kernel void adam(
        __global float* weights,
        __global float* adam_params,
        __global float* gradients,
        float alpha,
        uint t,
        uint n
) {
    uint global_id = get_global_id(0);
    if (global_id >= n) return;

    __global float* m = adam_params + (global_id * 2);
    __global float* v = m + 1;

    float grad = gradients[global_id];
    *m = B1 * *m + (1.0f - B1) * grad;
    *v = B2 * *v + (1.0f - B2) * grad * grad;

    float b1_t = pow(B1, (float)t);
    float b2_t = pow(B2, (float)t);

    float m_hat = *m  / (1 - b1_t);
    float v_hat = *v  / (1 - b2_t);

    weights[global_id] = weights[global_id] - m_hat / (sqrt(v_hat) + eps) * alpha;
}
