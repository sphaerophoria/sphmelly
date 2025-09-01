__kernel void sigmoid(
        __global float* in,
        __global float* output,
        uint n
) {
    uint global_id = get_global_id(0);
    if (global_id >= n) return;
    output[global_id] = 1.0f / (1.0f + exp(-in[global_id]));
}

__kernel void sigmoid_grad(
        __global float* downstream_grad,
        __global float* in,
        __global float* output,
        uint n
) {
    uint global_id = get_global_id(0);
    if (global_id >= n) return;

    float enx = exp(-in[global_id]);
    if (enx == INFINITY) {
        output[global_id] = 0;
        return;
    }

    float enxp1 = enx + 1;
    float grad = enx / enxp1 / enxp1 * downstream_grad[global_id];
    output[global_id] = grad;
}
