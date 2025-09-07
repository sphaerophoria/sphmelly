__kernel void add_assign(
        __global float* a,
        __global float* b,
        uint n
) {
    uint global_id = get_global_id(0);
    if (global_id >= n) return;
    a[global_id] += b[global_id];
}

__kernel void sub(
        __global float* a,
        __global float* b,
        __global float* res,
        uint n
) {
    uint global_id = get_global_id(0);
    if (global_id >= n) return;
    res[global_id] = a[global_id] - b[global_id];
}
