

struct conv_many_idx {
    uint img_cx;
    uint img_cy;
    uint out_channel_idx;
    uint img_idx;
};

struct kernel_iter {
    uint kx;
    uint ky;
    int img_x;
    int img_y;
    int channel_idx;

    const uint kw;
    const uint kh;
    const uint w;
    const uint h;
    const uint img_cx;
    const uint img_cy;
    const uint num_channels;

    const __global float* kernel_start;
    const __global float* img_start;
};


struct kernel_iter kernel_iter_init(struct conv_many_idx idx, uint kw, uint kh, uint in_c, uint w, uint h, __global float *kernel_buf, __global float* img_buf) {
    __global float* kernel_start = kernel_buf + kw * kh * in_c * idx.out_channel_idx;
    __global float* img_start = img_buf + w * h * in_c * idx.img_idx;
    // initialize with kx = -1 so that first call to kernel_iter_advance will start us at 0, 0, 0
    return (struct kernel_iter) {
        -1, 0, idx.img_cx - kw / 2, idx.img_cy - kh / 2, 0,
        kw, kh, w, h, idx.img_cx, idx.img_cy, in_c,
        kernel_start,
        img_start
    };
}

bool kernel_iter_advance(struct kernel_iter* it) {
    it->kx += 1;
    if (it->kx < it->kw) goto out_success;

    it->kx = 0;
    it->ky += 1;

    if (it->ky < it->kh) goto out_success;

    it->ky = 0;
    it->channel_idx += 1;

    if (it->channel_idx >= it->num_channels) {
        return false;
    }

out_success:
    it->img_x = it->img_cx + it->kx - it->kw / 2;
    it->img_y = it->img_cy + it->ky - it->kh / 2;

    return true;
}

float kernel_iter_sample_kernel(struct kernel_iter* it) {
    uint idx = it->kx + it->ky * it->kw + it->channel_idx * it->kw * it->kh;
    return it->kernel_start[idx];
}

float kernel_iter_sample_img_pad_extend(struct kernel_iter* it) {
    int img_x = max(0, min((int)it->w - 1, it->img_x));
    int img_y = max(0, min((int)it->h - 1, it->img_y));
    uint idx = img_x + img_y * it->w + it->channel_idx * it->w * it->h;
    return it->img_start[idx];
}

float kernel_iter_sample_img_pad_zero(struct kernel_iter* it) {
    if (it->img_x < 0 || it->img_y < 0 || it->img_x >= it->w || it->img_y >= it->h) {
        return 0.0f;
    }
    uint idx = it->img_x + it->img_y * it->w + it->channel_idx * it->w * it->h;
    return it->img_start[idx];
}

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

struct conv_many_idx conv_many_idx_from_id(uint id, uint w, uint h, uint out_c) {
    uint channel_size = w * h;
    uint out_img_size = channel_size * out_c;

    uint img_idx = id / out_img_size;
    uint channel_idx = (id % out_img_size) / channel_size;
    uint img_cy = (id % channel_size) / w;
    uint img_cx = id % w;

    return (struct conv_many_idx){
        img_cx,
        img_cy,
        channel_idx,
        img_idx,
    };
}

void conv_many_pad_extend(
    __global float* in_buf,
    __global float* kernel_buf,
    __global float* out_buf,
    uint w,
    uint h,
    uint in_c,
    uint n,
    uint kw,
    uint kh,
    uint out_c
) {
    uint global_id = get_global_id(0);
    struct conv_many_idx idx = conv_many_idx_from_id(global_id, w, h, out_c);
    if (idx.img_idx >= n) return;

    float sum = 0;
    struct kernel_iter it = kernel_iter_init(idx, kw, kh, in_c, w, h, kernel_buf, in_buf);

    while (kernel_iter_advance(&it)) {
        float kernel_val = kernel_iter_sample_kernel(&it);
        float img_val = kernel_iter_sample_img_pad_extend(&it);
        sum += img_val * kernel_val;
    }

    out_buf[global_id] = sum;
}

void conv_many_pad_zero(
    __global float* in_buf,
    __global float* kernel_buf,
    __global float* out_buf,
    uint w,
    uint h,
    uint in_c,
    uint n,
    uint kw,
    uint kh,
    uint out_c
) {
    uint global_id = get_global_id(0);
    struct conv_many_idx idx = conv_many_idx_from_id(global_id, w, h, out_c);
    if (idx.img_idx >= n) return;

    float sum = 0;
    struct kernel_iter it = kernel_iter_init(idx, kw, kh, in_c, w, h, kernel_buf, in_buf);

    while (kernel_iter_advance(&it)) {
        float kernel_val = kernel_iter_sample_kernel(&it);
        float img_val = kernel_iter_sample_img_pad_zero(&it);
        sum += img_val * kernel_val;
    }

    out_buf[global_id] = sum;
}

__kernel void conv_many(
    __global float* in_buf,
    __global float* kernel_buf,
    __global float* out_buf,
    uint w,
    uint h,
    uint in_c,
    uint n,
    uint kw,
    uint kh,
    uint out_c
) {
    conv_many_pad_extend(
        in_buf,
        kernel_buf,
        out_buf,
        w,
        h,
        in_c,
        n,
        kw,
        kh,
        out_c
    );
}

__kernel void make_grad_mirrored_kernel(
    __global float* in_kernel,
    __global float* grad_kernel,
    uint kw,
    uint kh,
    uint in_in_num_c,
    uint in_out_num_c
) {
    uint n = kw * kh * in_in_num_c * in_out_num_c;

    uint global_id = get_global_id(0);
    if (global_id >= n) return;

    uint channel_size = kw * kh;
    uint out_kernel_size = channel_size * in_out_num_c;
    uint in_kernel_size = channel_size * in_in_num_c;

    const uint out_x = global_id % kw;
    const uint out_y = (global_id % channel_size) / kw;
    const uint out_in_c = (global_id % out_kernel_size) / channel_size;
    const uint out_out_c = global_id / out_kernel_size;

    const uint in_x = kw - 1 - out_x;
    const uint in_y = kh - 1 - out_y;
    const uint in_in_c = out_out_c;
    const uint in_out_c = out_in_c;

    grad_kernel[global_id] = in_kernel[
        in_out_c * in_kernel_size +
        in_in_c * channel_size +
        in_y * kw +
        in_x
    ];

}

// FIXME: This whole thing can probably done by using the gradients as a kernel
// that is run over the input img
__kernel void conv_many_grad_kernel(
    __global float* in_grad_buf,
    __global float* in_img_buf,
    __global float* out_grad,
    uint w,
    uint h,
    uint in_c,
    uint n,
    uint kw,
    uint kh,
    uint out_c
) {
    // Run once per cell in the input kernel
    // in_img (w, h, in_c, n)
    // in_grad (w, y, out_c, n)
    // out_grad (kw, kh, in_c, out_c)

    // Each cell in the kernel touches every pixel (ish, not on the borders)

    uint global_id = get_global_id(0);

    uint kernel_channel_size = kw * kh;
    uint kernel_size = kernel_channel_size * in_c;

    if (global_id >= kernel_size * out_c) return;

    uint img_channel_size = w * h;
    uint in_img_size = img_channel_size * in_c;
    uint out_img_size = img_channel_size * out_c;

    uint kx = global_id % kw;
    uint ky = (global_id % kernel_channel_size) / kw;
    uint input_channel = (global_id % kernel_size) / kernel_channel_size;
    uint output_channel = global_id / kernel_size;

    int img_x_offs = kx - kw / 2;
    int img_y_offs = ky - kh / 2;

    float sum = 0;
    for (uint img_idx = 0; img_idx < n; img_idx++) {
        __global float* in_grad_img = in_grad_buf + img_idx * out_img_size;
        __global float* in_grad_channel = in_grad_img + output_channel * img_channel_size;

        __global float* in_img = in_img_buf + img_idx * in_img_size;
        __global float* in_img_channel = in_img + input_channel * img_channel_size;

        for (uint in_grad_y = 0; in_grad_y < h; in_grad_y++) {
            __global float* in_grad_row = in_grad_channel + w * in_grad_y;

            int img_y = in_grad_y + img_y_offs;
            img_y = max(0, min((int)h - 1, img_y));
            __global float* in_img_row = in_img_channel + w * img_y;

            for (uint in_grad_x = 0; in_grad_x < w; in_grad_x++) {
                float in_grad = *(in_grad_row + in_grad_x);

                int img_x = in_grad_x + img_x_offs;
                img_x = max(0, min((int)w - 1, img_x));
                float in_img = *(in_img_row + img_x);

                sum += in_img * in_grad;
            }
        }
    }

    out_grad[global_id] = sum;
}

__kernel void conv_many_grad_img(
    __global float* in_grad_buf,
    __global float* grad_kernel_buf,
    __global float* out_grad,
    uint w,
    uint h,
    uint in_c,
    uint n,
    uint kw,
    uint kh,
    uint out_c
) {
    conv_many_pad_zero(
        in_grad_buf,
        grad_kernel_buf,
        out_grad,
        w,
        h,
        out_c,
        n,
        kw,
        kh,
        in_c
    );
}
