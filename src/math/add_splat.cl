__kernel void add_splat_outer(
        __global float* a,
        __global float* b,
        uint b_stride,
        __global float* out,
        uint n
) {
    uint global_id = get_global_id(0);
    if (global_id >= n) return;

    uint a_idx = global_id % b_stride;
    out[global_id] = a[a_idx] + b[global_id];
}


__kernel void add_splat_outer_grad_a(
        __global float* downstream,
        uint downstream_size,
        __global float* a_grad,
        uint n
) {
    // Downstream gradients match shape of B, we want to output shape of A
    uint a_offs = get_global_id(0);
    if (a_offs >= n) return;

    uint repetitions = downstream_size / n;

    float acc = 0.0f;
    // Each element in this row in B is added to A, how does A's change effect
    // these elements? We move each element by the amount A changes
    //
    // A change in A results in each B changing 1-1. If we sum the downstream
    // gradients we should have the effect a change in A will have overall
    for (uint i = 0; i < repetitions; i++) {
        acc += downstream[i * n + a_offs];
    }

    a_grad[a_offs] = acc;
}

// B gradients are just a copy of downstream, as any change in B will have a 1-1
// change on the output
