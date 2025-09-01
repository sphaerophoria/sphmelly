float sigmoid(float in) {
    return  1.0f / (1.0f + exp(-in));
}

__kernel void bce_with_logits(
        __global float* input,
        __global float* expected,
        __global float* ret,
        uint n
) {
    uint global_id = get_global_id(0);
    if (global_id >= n) return;

    float sig_x = sigmoid(input[global_id]);
    float this_expected = expected[global_id];

    ret[global_id] = - this_expected * max(-100.0f, log(sig_x)) - (1 - this_expected) * max(-100.0f, log(1 - sig_x));
}

__kernel void bce_with_logits_grad(
        __global float* downstream_grads,
        __global float* input,
        __global float* expected,
        __global float* ret,
        uint n
) {
    uint global_id = get_global_id(0);
    if (global_id >= n) return;

    float this_input = input[global_id];
    float this_expected = expected[global_id];

    float grad = 1 / (exp(-this_input) + 1) - this_expected;
    ret[global_id] = downstream_grads[global_id] * grad;
}
