
ulong philox2x32round(ulong ctr, uint key) {
    ulong M = 0xd256d193;
    uint ctr_low = ctr;
    uint ctr_high = ctr >> 32;

    ulong multiplied = (ulong)ctr_high * M;
    uint multiplied_high = multiplied >> 32;
    uint multiplied_low = multiplied;

    ulong out_high = multiplied_high ^ key ^ ctr_low;
    out_high <<= 32;

    return out_high | multiplied_low;
}

uint philox2x32bumpkey(uint key) {
    return key + 0x9E3779B9;
}

ulong philox2x32(ulong ctr, uint key) {

    ctr = philox2x32round(ctr, key);
    // philox 2x32 rounds was 10 in philox.h
    for (int i = 0; i < 9; i++) {
        key = philox2x32bumpkey(key);
        ctr = philox2x32round(ctr, key);
    }

    return ctr;
}

union fu {
    float f;
    uint u;
};

float ulongToFloat(ulong val) {
    union fu ret;
    uint lz = __builtin_clz(val);
    uint mantissa = val & 0x7fffff;

    // This puts us in subnormal numbers which i don't want to think about,
    // round to smallest normal number or 0
    if (lz >= 41) {
        // 64 bits - 41 is 23 bits left in mantissa. Half way is 22
        //
        // FIXME: This is a crazy amount of entropy to be losing
        //
        ulong thresh = 1 << 22;
        if (val < thresh) return 0.0f;

        uint exponent = 1;
        ret.u = exponent << 23;
        return ret.f;
    }
    // if val is 0xffff...
    // We want to be 0.99999999
    // This is (binary) 1.1111111111 * 2^-1. i.e. almost 2 * 0.5 == almost 1
    // Therefore if lz == 0, we want -1 exponent represented as 126 - 127,
    // increasing for each leading 0
    uint exponent = 126 - lz;

    ret.u = exponent << 23 | mantissa;
    return ret.f;
}

struct philox_thread_rng {
    ulong ctr;
    uint seed;
};

struct philox_thread_rng rngInit(ulong ctr, uint seed) {
  // Make a new rng based off the output of this one. This way if we have
  // sequential RNG accesses we don't have to do anything special
  return (struct philox_thread_rng) { 0, philox2x32(ctr, seed), };
}

ulong rngGenerate(struct philox_thread_rng* rng) {
    return philox2x32(rng->ctr++, rng->seed);
}

float randFloatBetween(struct philox_thread_rng* rng, float min, float max) {
    float t = ulongToFloat(rngGenerate(rng));
    return t * (max - min) + min;
}

ulong randUlongBetween(struct philox_thread_rng* rng, ulong min, ulong max) {
    return rngGenerate(rng) % (max - min) + min;
}


__kernel void rand(
        __global float* output,
        ulong initial_ctr,
        uint seed,
        uint n
) {
    uint global_id = get_global_id(0);
    if (global_id >= n) return;

    ulong ctr = global_id + initial_ctr;

    ulong val = philox2x32(ctr, seed);

    output[global_id] = ulongToFloat(val);
}

float gaussianRandVal(ulong ctr, uint seed) {
    float u1 = ulongToFloat(philox2x32(ctr, seed));
    float u2 = ulongToFloat(philox2x32(ctr + 1, seed));

    // Box muller transform using only 1 value
    return sqrt(-2.0 * log(u1)) * cospi(u2);
}

__kernel void gaussian(
        __global float* output,
        ulong initial_ctr,
        uint seed,
        uint n
) {
    uint global_id = get_global_id(0);
    if (global_id >= n) return;

    ulong ctr = (global_id * 2) + initial_ctr;
    output[global_id] = gaussianRandVal(ctr, seed);
}

__kernel void gaussian_noise(
        __global float* input,
        __global float* output,
        ulong initial_ctr,
        uint seed,
        uint n
) {
    uint global_id = get_global_id(0);
    if (global_id >= n) return;

    ulong ctr = (global_id * 2) + initial_ctr;
    output[global_id] = fmax(0.0f, fmin(255.0f / 256.0f, input[global_id] + gaussianRandVal(ctr, seed) / 50.0f));
}
