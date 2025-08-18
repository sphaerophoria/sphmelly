__kernel void relu(
    __global float* in,
    __global float* out,
    uint n
) {
    uint id = get_global_id(0);
    if (id >= n) return;


    float in_val = in[id];

    out[id] = (in_val > 0) ? in_val : 0;
}

__kernel void relu_grad(
    __global float* downstream_grad,
    __global float* in,
    __global float* out,
    uint n
) {
    uint id = get_global_id(0);
    if (id >= n) return;

    float in_val = in[id];

    out[id] = (in_val > 0) ? downstream_grad[id] : 0;
}
