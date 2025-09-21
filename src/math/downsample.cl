
float lerp(float a, float b, float t) {
    return (1.0f - t) * a + t * b;
}


float sample_normal_dist(float x) {
    return exp(-(x * x) / 2.0f) / sqrt(2.0f * M_PI);
}

float sample_gaussian_blur_kernel(
    uint kx,
    uint ky,
    uint kernel_size,
    float stddev
) {
    // Truncating div, center of 3 == 1
    uint kernel_center = kernel_size / 2;

    // Small standard deviations should result in less blur, which means we
    // should be sampling farther away from the center as we get further from
    // the kernel size. We kinda just stretch some gaussian dist over our
    // kernel, imagine a very wide stretch resulting in a box blur, and a very
    // narrow stretch resulting in all the mass of the kernel ending up in the
    // center. Divide by stddev to get desired "blurrier when bigger number"
    // effect. 12.0 so that a stddev of 1.0 will result in a gaussian that
    // samples from [-3, 3]
    float sample_width = 12.0f / kernel_size / stddev;

    int kx_offs = kx - kernel_center;
    int ky_offs = ky - kernel_center;

    const float norm_x_sample = sample_normal_dist(kx_offs * sample_width);
    const float norm_y_sample = sample_normal_dist(ky_offs * sample_width);

    return norm_x_sample * norm_y_sample * sample_width;
}

__kernel void make_downsample_kernel(
        __global float* ret,
        // width or height, not width * height
        uint kernel_size,
        float stddev
) {
    uint global_id = get_global_id(0);
    uint kernel_elems = kernel_size * kernel_size;
    float res = 0;
    if (global_id < kernel_elems) {
        uint kx = global_id % kernel_size;
        uint ky = global_id / kernel_size;

        if (kernel_size * kernel_size > get_local_size(0)) {
            ret[global_id] = INFINITY;
            return;
        }

        res = sample_gaussian_blur_kernel(kx, ky, kernel_size, stddev);
    }

    float total = work_group_reduce_add(res);

    if (global_id < kernel_elems) {
        ret[global_id] = res / total;
    }
}

struct downsample_box {
    float x;
    float y;
    float w;
    float h;
    float r;
};

__kernel void downsample_box_inner(
        __global float* in_imgs,
        __global float* out_imgs,
        struct downsample_box box,
        uint global_id,
        uint in_w,
        uint in_h,
        uint channels,
        uint num_images,
        uint out_width,
        uint out_height,
        uint num_samples_1d
) {
    // in_imgs (in_w, in_h, channels, num_images)
    // out_imgs (out_width, out_height, channels, num_images)

    uint out_channel_size = out_width * out_height;
    // (out_w, out_h, c, n)
    // But we can view it as (out_w, out_h, c * n) with no consequence :)
    uint out_x = global_id % out_width;
    uint out_y = (global_id % out_channel_size) / out_width;
    uint channel = global_id / out_channel_size;

    float sum = 0.0f;

    for (int i = 0; i < num_samples_1d; i++) {
        float x_offs = (1.0f / num_samples_1d / 2.0f) + (float)i / num_samples_1d - 0.5;
        for (int j = 0; j < num_samples_1d; j++) {
            float y_offs = (1.0f / num_samples_1d / 2.0f) + (float)j / num_samples_1d - 0.5;

            float x_norm = ((float)out_x + x_offs) / (float)out_width;
            float y_norm = ((float)out_y + y_offs) / (float)out_height;

            // Output space, x, y pixel in out img
            // normalized space, x,y pixel in out img normlaized from [0,1]
            // box space, x,y pixel but relative to rotated space
            // in img space, take box space and unrotated and offset

            // Relative to box center, which box pixel do we want

            if (in_w != in_h) {
                out_imgs[global_id] = INFINITY;
                return;
            }

            float x_box = (x_norm - 0.5) * box.w * in_w;
            float y_box = (y_norm - 0.5) * box.h * in_w;

            float2 box_x_axis = {cos(box.r), sin(box.r)};
            float2 box_y_axis = {-box_x_axis[1], box_x_axis[0]};

            float2 in_box_center = {box.x * in_w + (float)in_w / 2.0f, box.y * in_h + (float)in_h / 2.0f};

            float2 in_px = in_box_center + box_x_axis * x_box + box_y_axis * y_box;

            uint in_channel_size = in_w * in_h;

            uint in_channel_offs = (uint)in_px[1] * in_w + (uint)in_px[0];
            bool in_bounds = in_px[0] >= 0 && in_px[0] < in_w && in_px[1] < in_h && in_px[1] >= 0;
            sum += (in_bounds)
                ? in_imgs[channel * in_channel_size + in_channel_offs]
                : 0.0;
        }
    }

    out_imgs[global_id] = sum / num_samples_1d / num_samples_1d;
}

__kernel void downsample(
        __global float* in_imgs,
        __global float* out_imgs,
        uint multisample,
        uint in_w,
        uint in_h,
        uint channels,
        uint num_images,
        uint out_size
) {
    // in_imgs (in_w, in_h, channels, num_images)
    // out_imgs (out_size, out_size, channels, num_images)

    uint global_id = get_global_id(0);
    if (global_id >= out_size * out_size * channels * num_images) {
        return;
    }

    struct downsample_box box = {0, 0, 1.0, 1.0, 0};
    downsample_box_inner(
        in_imgs,
        out_imgs,
        box,
        global_id,
        in_w,
        in_h,
        channels,
        num_images,
        out_size,
        out_size,
        multisample
    );
}

__kernel void downsample_box(
        __global float* in_imgs,
        __global float* out_imgs,
        __global float* boxes,
        uint multisample,
        uint in_w,
        uint in_h,
        uint channels,
        uint num_images,
        uint out_width,
        uint out_height
) {
    // in_imgs (in_w, in_h, channels, num_images)
    // out_imgs (out_width, out_height, channels, num_images)
    //
    // x_offs,y_offs,width,height,angle
    // boxes (5, num_images)

    uint global_id = get_global_id(0);
    if (global_id >= out_width * out_height * channels * num_images) {
        return;
    }

    __global float* thread_box_data = boxes + global_id / out_width / out_height / channels * 5;
    struct downsample_box box = {
        thread_box_data[0],
        thread_box_data[1],
        thread_box_data[2],
        thread_box_data[3],
        thread_box_data[4]
    };
    downsample_box_inner(
        in_imgs,
        out_imgs,
        box,
        global_id,
        in_w,
        in_h,
        channels,
        num_images,
        out_width,
        out_height,
        multisample
    );
}
