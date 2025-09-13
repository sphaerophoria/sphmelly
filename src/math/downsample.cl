
float lerp(float a, float b, float t) {
    return (1.0f - t) * a + t * b;
}

__kernel void downsample(
        __global float* in_imgs,
        __global float* out_imgs,
        uint in_w,
        uint in_h,
        uint channels,
        uint num_images,
        uint out_size
) {
    uint global_id = get_global_id(0);
    if (global_id >= out_size * out_size * channels * num_images) {
        return;
    }

    uint out_channel_size = out_size * out_size;
    // (out_w, out_h, c, n)
    // But we can view it as (out_w, out_h, c * n) with no consequence :)
    uint out_x = global_id % out_size;
    uint out_y = (global_id % out_channel_size) / out_size;
    uint channel = global_id / out_channel_size;

    float x_norm = (float)out_x / (float)out_size;
    float y_norm = (float)out_y / (float)out_size;

    float x_lerp = fmod(x_norm, 1.0f);
    float y_lerp = fmod(y_norm, 1.0f);

    float in_x_per_out = (float)in_w / (float)out_size;
    float in_y_per_out = (float)in_h / (float)out_size;

    uint in_x = x_norm * in_w;
    uint in_y = y_norm * in_w;
    uint in_channel_size = in_w * in_h;

    out_imgs[global_id] = in_imgs[channel * in_channel_size + in_y * in_w + in_x];
}
