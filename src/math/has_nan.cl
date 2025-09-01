__kernel void has_nan(
    __global float* input,
    __global float* out,
    uint n
) {
    uint global_id = get_global_id(0);

    float val = (global_id < n) ? input[global_id] : 0.0;
    int is_nan = isnan(val);

    uint local_size = get_local_size(0);
    uint local_id = get_local_id(0);

    int any_nan = work_group_any(is_nan);

    if (local_id == 0) {
        out[global_id / local_size] = (any_nan) ? NAN : 0.0f;
    }
}
