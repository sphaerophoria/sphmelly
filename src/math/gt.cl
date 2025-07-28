__kernel void gt(
        __global float* in,
        uint stride,
        __global float* output,
        uint n

) {
    uint global_id = get_global_id(0);
    if (global_id >= n) return;

    float a = in[global_id];
    float b = in[global_id + stride];
    output[global_id] = (a > b) ? 1.0 : 0.0;
}
