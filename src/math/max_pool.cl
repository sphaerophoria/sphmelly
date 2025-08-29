uint calc_max(
    uint global_id,
    __global float* in,
    uint in_width,
    uint in_height,
    uint stride,
    float* out_val
) {
    uint output_x = global_id % (in_width / stride);
    uint output_y = global_id / (in_width / stride);

    uint in_x = output_x * stride;
    uint in_y = (output_y * stride) % in_height;
    uint in_channel = (output_y * stride) / in_height;

    uint max_idx;
    *out_val = -INFINITY;

    for (int y = in_y; y < stride + in_y; y++) {
        if (y >= in_height) continue;

        for (int x = in_x; x < stride + in_x; x++) {
            if (x >= in_width) continue;

            uint idx = in_channel * in_width * in_height + y * in_width + x;
            float val = in[idx];
            if (val > *out_val) {
                *out_val = val;
                max_idx = idx;
            }
        }
    }

    return max_idx;
}

__kernel void maxpool(
        __global float* in,
        __global float* out,
        uint in_width,
        uint in_height,
        uint stride,
        uint n
) {
    // (w, h, c, n)

    uint global_id = get_global_id(0);
    if (global_id >= n) return;

    float out_val;
    calc_max(global_id, in, in_width, in_height, stride, &out_val);

    out[global_id] = out_val;
}

__kernel void maxpool_grad(
        __global float* downstream_grads,
        __global float* in,
        __global float* out,
        uint in_width,
        uint in_height,
        uint stride,
        uint n
) {
    uint global_id = get_global_id(0);
    if (global_id >= n) return;

    float out_val;
    uint max_idx = calc_max(global_id, in, in_width, in_height, stride, &out_val);
    out[max_idx] = downstream_grads[global_id];
}
