
__kernel void masked_conv(
    __global float* in_buf,
    __global float* mask,
    __global float* out_buf,
    uint img_width,
    uint img_height,
    uint n,
    __global float* img_kerns,
    uint kernel_width
) {
    uint global_id = get_global_id(0);
    if (global_id >= n) return;

    uint cx = global_id % img_width;
    uint cy = global_id / img_width;

    if (mask[cy * img_width + cx] < 0.5) {
        out_buf[global_id] = in_buf[global_id];
        return;
    }

    uint img_id = global_id / img_width / img_height;
    __global float* img_kern = img_kerns + img_id * kernel_width * kernel_width;
    float sum = 0;
    float kernel_total = 0;


    int min_y = img_id * img_height;
    int max_y = min_y + img_height - 1;
    for (int kx = 0; kx < kernel_width; kx++) {
        int img_x = cx + kx - kernel_width / 2;
        img_x = min((int)img_width - 1, max(0, img_x));

        for (int ky = 0; ky < kernel_width; ky++) {
            int img_y = cy + ky - kernel_width / 2;
            img_y = min(max_y, max(min_y, img_y));

            float a = img_kern[ky * kernel_width + kx];
            float b = in_buf[img_y * img_width + img_x];

            sum += a * b;
            kernel_total += a;
        }
    }

    out_buf[global_id] = sum / kernel_total;
}
