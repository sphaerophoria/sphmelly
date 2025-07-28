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
