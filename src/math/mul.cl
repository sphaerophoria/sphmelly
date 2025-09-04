__kernel void mul_scalar(
        __global float* a,
        float b,
        __global float* output,
        uint n

) {
    uint global_id = get_global_id(0);
    if (global_id >= n) return;
    output[global_id] = a[global_id] * b;
}

__kernel void elem_mul(
    __global float* a,
    __global float* b,
    uint b_len,
    __global float* output,
    uint out_len
) {
    uint global_id = get_global_id(0);
    if (global_id >= out_len) return;

    output[global_id] = a[global_id] * b[global_id % b_len];
}
