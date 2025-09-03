float lerp(float a, float b, float t) {
    return t * (b - a) + a;
}

bool sample1d(__global float* input, uint len, float x1, float x2, float* out) {
    if (x1 < 0 || x2 < 0 || x1 > len) return false;
    uint left_input_id = x1;
    uint right_input_id = min(x2, (float)len - 1);

    float right_ratio = fmod(x2, 1.0f);
    float left_ratio = 1.0f - right_ratio;

    float left_mod_value = input[left_input_id];
    float right_mod_value = input[right_input_id];

    *out = left_mod_value * left_ratio + right_mod_value * right_ratio;
    return true;
}

struct barcode_sampler {
    float2 x_axis;
    float module_width;
    float x;
    float y;
};

int pixel_offs_to_barcode_idx(struct barcode_sampler const* pc, float2 offs) {
    return pc->x + dot(offs, pc->x_axis) / pc->module_width;
}

float2 sample_perlin_gradient_vector(uint cell_x, uint cell_y, uint seed) {
    uint seed2 = philox2x32(cell_x, seed);
    float vx = ulongToFloat(philox2x32(cell_y * 2, seed2)) * 2.0f - 1.0f;
    float vy = ulongToFloat(philox2x32(cell_y * 2 + 1, seed2)) * 2.0f - 1.0f;

    return (float2){vx, vy} / sqrt(2.0f);
}

float smoothstep(float t) {
    float t2 = t * t;
    float t3 = t2 * t;
    return 3 * t2 - 2 * t3;
}

float perlin_interp(float a, float b, float t) {
    return a + smoothstep(t) * (b - a);
}

float sample_perlin_noise(uint x, uint y, uint seed, uint cell_width) {
    float vals[4];
    bool has_0 = false;
    for (int i = 0; i < 4; i++) {
        uint grad_x = (x / cell_width) + i % 2;
        uint grad_y = (y / cell_width) + i / 2;

        int offs_x_px = x - grad_x * cell_width;
        int offs_y_px = y - grad_y * cell_width;

        if (offs_x_px == 0 && offs_y_px == 0) {
            has_0 = true;
        }

        float offs_x_norm = (float)offs_x_px / cell_width;
        float offs_y_norm = (float)offs_y_px / cell_width;

        float2 grad = sample_perlin_gradient_vector(grad_x, grad_y, seed);
        vals[i] = dot((float2){offs_x_norm, offs_y_norm}, grad) * 2.0f;
    }

    float x_lerp_t = (float)(x % cell_width) / cell_width;
    float y_lerp_t = (float)(y % cell_width) / cell_width;

    float a = perlin_interp(vals[0], vals[1], x_lerp_t);
    float b = perlin_interp(vals[2], vals[3], x_lerp_t);

    float ret = perlin_interp(a, b, y_lerp_t) / 2.0f + 0.5f;

    return ret;
}

#define BARCODE_NUM_DIGITS 12
#define BARCODE_START_END_WIDTH 3
#define BARCODE_MIDDLE_WIDTH 5
#define BARCODE_QUIET_ZONE_WIDTH 5
#define BARCODE_DIGIT_WIDTH 7
#define BARCODE_HALF_DIGITS_WIDTH (BARCODE_DIGIT_WIDTH * BARCODE_NUM_DIGITS / 2)
#define BARCODE_NUM_MODULES (BARCODE_NUM_DIGITS * BARCODE_DIGIT_WIDTH + BARCODE_START_END_WIDTH * 2 + BARCODE_MIDDLE_WIDTH + BARCODE_QUIET_ZONE_WIDTH * 2)

enum pattern_gen_purpose {
    PGP_QUIET,
    PGP_LEFT_DIGIT,
    PGP_RIGHT_DIGIT,
    PGP_START_END,
    PGP_MIDDLE,
    PGP_INVALID,
};

struct pattern_gen_info {
    enum pattern_gen_purpose purpose;
    uint offs;
    // Only valid if purpose == PGP_*_DIGIT
    uint digit;
};

struct pattern_gen_info get_pattern_gen_info(uint id, ulong ctr, uint seed) {
    uint acc = BARCODE_QUIET_ZONE_WIDTH;
    if (id < acc) {
        return (struct pattern_gen_info) {
            PGP_QUIET, id, 0,
        };
    }

    acc += BARCODE_START_END_WIDTH;

    if (id < acc) {
        return (struct pattern_gen_info) {
            PGP_START_END, id - (acc - BARCODE_START_END_WIDTH), 0,
        };
    }

    acc += BARCODE_HALF_DIGITS_WIDTH;

    if (id < acc) {
        uint start_segment_offset = id - (acc - BARCODE_HALF_DIGITS_WIDTH);
        uint digit_offs = start_segment_offset % BARCODE_DIGIT_WIDTH;
        uint digit_id = start_segment_offset / BARCODE_DIGIT_WIDTH;
        return (struct pattern_gen_info) {
            PGP_LEFT_DIGIT, digit_offs, philox2x32(ctr - digit_offs, seed) % 10,
        };
    }

    acc += BARCODE_MIDDLE_WIDTH;

    if (id < acc) {
        return (struct pattern_gen_info) {
            PGP_MIDDLE, id - (acc - BARCODE_MIDDLE_WIDTH), 0,
        };
    }

    acc += BARCODE_HALF_DIGITS_WIDTH;

    if (id < acc) {
        uint start_segment_offset = id - (acc - BARCODE_HALF_DIGITS_WIDTH);
        uint digit_offs = start_segment_offset % BARCODE_DIGIT_WIDTH;
        uint digit_id = start_segment_offset / BARCODE_DIGIT_WIDTH + BARCODE_NUM_DIGITS / 2;
        return (struct pattern_gen_info) {
            PGP_RIGHT_DIGIT, digit_offs, philox2x32(ctr - digit_offs, seed) % 10,
        };
    }
    acc += BARCODE_START_END_WIDTH;

    if (id < acc) {
        return (struct pattern_gen_info) {
            PGP_START_END, id - (acc - BARCODE_START_END_WIDTH), 0,
        };
    }

    acc += BARCODE_QUIET_ZONE_WIDTH;
    if (id < acc) {
        return (struct pattern_gen_info) {
            PGP_QUIET, id - (acc - BARCODE_QUIET_ZONE_WIDTH), 0,
        };
    }

    return (struct pattern_gen_info) {
        PGP_INVALID, 0, 0,
    };
}

constant bool L_PATTERN_TABLE[70] ={
     1, 1, 1, 0, 0, 1, 0,
     1, 1, 0, 0, 1, 1, 0,
     1, 1, 0, 1, 1, 0, 0,
     1, 0, 0, 0, 0, 1, 0,
     1, 0, 1, 1, 1, 0, 0,
     1, 0, 0, 1, 1, 1, 0,
     1, 0, 1, 0, 0, 0, 0,
     1, 0, 0, 0, 1, 0, 0,
     1, 0, 0, 1, 0, 0, 0,
     1, 1, 1, 0, 1, 0, 0,
};

bool module_color_from_info(struct pattern_gen_info info) {
    switch (info.purpose) {
        case PGP_QUIET:
            return true;
        case PGP_LEFT_DIGIT:
            return L_PATTERN_TABLE[info.digit * 7 + info.offs];
        case PGP_RIGHT_DIGIT:
            return !L_PATTERN_TABLE[info.digit * 7 + info.offs];
        case PGP_START_END:
            info.offs += 1;
            // fallthrough
        case PGP_MIDDLE:
            return info.offs % 2 == 0;
        case PGP_INVALID:
        default:
            return false;
    }
}

__kernel void generate_module_patterns(
    __global float* out,
    uint num_patterns,
    uint seed,
    ulong ctr_start
) {
    // Each thread is responsible for writing a single bar
    uint global_id = get_global_id(0);
    if (global_id >= num_patterns * BARCODE_NUM_MODULES) return;

    ulong ctr = ctr_start + global_id;

    struct pattern_gen_info info = get_pattern_gen_info(global_id % BARCODE_NUM_MODULES, ctr, seed);
    out[global_id] = module_color_from_info(info);
}

struct concrete_params {
    __global float* sample_buf;
    uint img_width;
    uint img_height;
    uint img_idx;
    float x_scale;
    float y_scale;
    float x_offs;
    float y_offs;
    float rot;
    uint seed;
    uint perlin_cell_width;
    float noise_x_scale;
    float noise_y_scale;
    float dark_color;
    float light_color;
    float background_color;
};

__kernel void sample_barcode_params(
    __global struct concrete_params* ret,
    uint expected_concrete_params_size,
    // size of barcode * n
    __global float* sample_buf_space,
    // Explicit separated output for training labels
    __global float* bbox_out,
    uint n,
    uint img_width,
    uint img_height,
    float min_x_offs,
    float max_x_offs,
    float min_y_offs,
    float max_y_offs,
    float min_x_scale,
    float max_x_scale,
    float min_aspect,
    float max_aspect,
    float min_contrast,
    uint min_perlin_grid_size,
    uint max_perlin_grid_size,
    float min_x_noise_multiplier,
    float max_x_noise_multiplier,
    float min_y_noise_multiplier,
    float max_y_noise_multiplier,
    float min_background_color,
    float max_background_color,
    uint num_images,
    uint seed,
    ulong ctr_start
) {

    uint global_id = get_global_id(0);
    if (global_id >= n) return;

    if (sizeof(struct concrete_params) != expected_concrete_params_size) {
        if (global_id == 0) {
            printf("Concrete params struct size is %d, expected %d\n", sizeof(struct concrete_params), expected_concrete_params_size);
        }

        ret[global_id] = (struct concrete_params){0};
    }

    struct philox_thread_rng rng = rngInit(ctr_start + global_id, seed);

    float x_offs = randFloatBetween(&rng, min_x_offs, max_x_offs);
    float y_offs = randFloatBetween(&rng, min_y_offs, max_y_offs);
    float x_scale = randFloatBetween(&rng, min_x_scale, max_x_scale);
    float y_scale = randFloatBetween(&rng, min_aspect * x_scale, max_aspect * x_scale);
    float rot = randFloatBetween(&rng, -M_PI / 2.0, M_PI / 2.0);

    float contrast = randFloatBetween(&rng, min_contrast, 1.0);
    float dark_color = randFloatBetween(&rng, 0.0, 1.0 - contrast);
    float light_color = dark_color + contrast;
    uint perlin_cell_width = randUlongBetween(&rng, min_perlin_grid_size, max_perlin_grid_size);

    float noise_x_scale = randFloatBetween(&rng, min_x_noise_multiplier, max_x_noise_multiplier);
    float noise_y_scale = randFloatBetween(&rng, min_y_noise_multiplier, max_y_noise_multiplier);

    float background_color = randFloatBetween(&rng, min_background_color, max_background_color);

    __global float* sample_buf = sample_buf_space + global_id * BARCODE_NUM_MODULES;
    uint image_idx = randUlongBetween(&rng, 0, num_images);

    ret[global_id] = (struct concrete_params) {
        .img_width = img_width,
        .img_height = img_height,
        .img_idx = image_idx,
        .sample_buf = sample_buf,
        .x_scale = x_scale,
        .y_scale = y_scale,
        .x_offs = x_offs,
        .y_offs = y_offs,
        .rot = rot,
        .seed = seed,
        .perlin_cell_width = perlin_cell_width,
        .noise_x_scale = noise_x_scale,
        .noise_y_scale = noise_y_scale,
        .dark_color = dark_color,
        .light_color = light_color,
        .background_color = background_color,
    };

    bbox_out[global_id * 6 + 0] = x_offs / (float)img_width;
    bbox_out[global_id * 6 + 1] = y_offs / (float)img_height;
    bbox_out[global_id * 6 + 2] = sqrt(x_scale);
    bbox_out[global_id * 6 + 3] = sqrt(y_scale);
    bbox_out[global_id * 6 + 4] = cos(rot);
    bbox_out[global_id * 6 + 5] = sin(rot);
}

bool multisample_barcode(
    struct barcode_sampler sampler,
    float light_color,
    float dark_color,
    __global float* sample_buf,
    float* out
) {
    const int num_samples_x = 4;

    float sum = 0;
    for (int i = 0; i < num_samples_x; i++) {
        float x_offs = (1.0f / num_samples_x / 2.0f) + (float)i / num_samples_x - 0.5;
        for (int j = 0; j < num_samples_x; j++) {
            float y_offs = (1.0f / num_samples_x / 2.0f) + (float)j / num_samples_x - 0.5;

            float2 offs = {x_offs, y_offs};

            int barcode_idx = pixel_offs_to_barcode_idx(&sampler, offs);
            if (barcode_idx < 0 || barcode_idx >= BARCODE_NUM_MODULES) {
                return false;
            }
            sum += lerp(dark_color, light_color, sample_buf[barcode_idx]);
        }
    }

    *out = sum / num_samples_x / num_samples_x;
    return true;
}

__kernel void generate_barcode(
        __global struct concrete_params* all_params,
        __global float* ret,
        __global float* mask_ret,
        __global float* background_buf,
        uint width,
        uint height,
        uint enable_backgrounds,
        uint num_barcodes
) {
    uint global_id = get_global_id(0);
    if (global_id >= width * height * num_barcodes) return;

    uint barcode_id = global_id / width / height;
    uint pixel_id = global_id % (width * height);

    struct concrete_params our_params = all_params[barcode_id];
    uint image_size = width * height;

    // Just so early returns don't fall over, maybe causing extra memory io
    // which could be bad maybe i dunno whatever
    ret[global_id] = (enable_backgrounds)
        ? background_buf[our_params.img_idx * width * height + pixel_id]
        : our_params.background_color;

    float crot = cos(our_params.rot);
    float srot = sin(our_params.rot);

    float2 x_axis = {crot, srot};
    float2 y_axis = {-srot, crot};

    int image_x = pixel_id % width;
    int image_y = pixel_id / width;

    float image_cx = width / 2.0f;
    float image_cy = height / 2.0f;

    float barcode_cx_img = image_cx + our_params.x_offs;
    float barcode_cy_img = image_cy + our_params.y_offs;

    float barcode_width = (float)width * our_params.x_scale;
    float barcode_height = (float)height * our_params.y_scale;

    float barcode_cx_barcode = barcode_width / 2.0;
    float barcode_cy_barcode = barcode_height / 2.0;

    float2 offs = {image_x - barcode_cx_img, image_y - barcode_cy_img};

    // Surprisingly global_id here makes more sense than pixel_id, as we will
    // naturally get different noise per image
    float noise_x_offs = sample_perlin_noise(global_id % width, global_id / width, our_params.seed, our_params.perlin_cell_width);
    float noise_y_offs = sample_perlin_noise(global_id % width, global_id / width, our_params.seed + 1, our_params.perlin_cell_width);
    float x = dot(offs, x_axis) + barcode_cx_barcode + noise_x_offs * our_params.noise_x_scale;
    float y = dot(offs, y_axis) + barcode_cy_barcode + noise_y_offs * our_params.noise_y_scale;

    if (y > height * our_params.y_scale || y < 0) {
        return;
    }

    float module_width = (float)width / (float)BARCODE_NUM_MODULES * our_params.x_scale;
    struct barcode_sampler barcode_sampler = {
        x_axis, module_width, x / module_width, y,
    };

    float val = 0.0f;

    // Initial testing is that AA greatly improves the look of the thing, even
    // with a blur kernel on top
    if (multisample_barcode(
            barcode_sampler,
            our_params.light_color,
            our_params.dark_color,
            our_params.sample_buf,
            &val)) {
        ret[global_id] = val;
        mask_ret[global_id] = 1.0;
    }
}

float sample_normal_dist(float x) {
    return exp(-(x * x) / 2.0f) / sqrt(2.0f * M_PI);
}

__kernel void generate_blur_kernels(
        __global float* ret,
        // width or height, not width * height
        uint kernel_size,
        // 1 per cell per kernel
        uint n,
        float min_sigmoid,
        float max_sigmoid,
        uint seed,
        ulong ctr
) {
    uint global_id = get_global_id(0);
    if (global_id >= n) return;

    uint kernel_elems = kernel_size * kernel_size;
    uint kernel_id = global_id / kernel_elems;
    uint kernel_offs = global_id % kernel_elems;

    uint kx = kernel_offs % kernel_size;
    uint ky = kernel_offs / kernel_size;
    // Truncating div, center of 3 == 1
    uint kernel_center = kernel_size / 2;

    struct philox_thread_rng rng = rngInit(ctr + kernel_id, seed);
    float stddev = randFloatBetween(&rng, min_sigmoid, max_sigmoid);

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

    // Multiply x/y gaussians, should have same shape, volume will not be 1
    ret[global_id] = norm_x_sample * norm_y_sample * sample_width;
}
