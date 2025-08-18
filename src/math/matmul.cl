__kernel void matmul(
        __global float* a,
        uint a_width,
        uint a_height,
        __global float* b,
        uint b_width,
        uint b_height,
        __global float* output,
        uint n
) {
    uint global_id = get_global_id(0);
    if (global_id >= n) return;

    uint out_row = global_id / b_width;
    uint out_col = global_id % b_width;

    const uint b_mat_idx = global_id / (b_width * a_height);
    const __global float* b_mat_base = b + (b_mat_idx * b_width * b_height);

    float accumulator = 0.0f;
    for (uint i = 0; i < a_width; i++) {
        float a_val = a[(out_row  % a_height) * a_width + i];
        float b_val = b_mat_base[i * b_width + out_col];

        accumulator += a_val * b_val;
    }

    output[global_id] = accumulator;
}

__kernel void matmul_grad_a(
        __global float* downstream_gradients,
        uint num_mats,
        __global float* a,
        uint a_width,
        uint a_height,
        __global float* b,
        uint b_width,
        __global float* output,
        uint n
) {
  // Each element in A is multiplied by every element in a row of B
  //
  // E.g. the top left element in A is multiplied by every element in the first
  // row of B
  //
  // Each element modifies the downstream gradient at the output location by
  // out_(a_row,b_col)
  //
  // Each relevant output ends up in the equivalent row of the output, at the column
  // of B we were multiplying by
  uint global_id = get_global_id(0);
  if (global_id >= n) return;

  uint a_row = global_id / a_width;
  uint a_col = global_id % a_width;
  uint b_row = a_col;

  uint a_idx = a_row * a_width + a_col;
  uint downstream_gradient_row = global_id % b_width;

  uint downstream_grad_size = b_width * a_height;
  // b_height == a_width
  uint b_size = b_width * a_width;

  float acc = 0.0;
  for (uint mat_idx = 0; mat_idx < num_mats; mat_idx++) {
      const __global float* downstream_grad_base = downstream_gradients + downstream_grad_size * mat_idx;
      const __global float* b_base = b + b_size * mat_idx;

      for (uint i = 0; i < b_width; i++) {
          uint b_idx = b_row * b_width + i;
          uint downstream_idx = a_row * b_width + i;
          acc += b_base[b_idx] * downstream_grad_base[downstream_idx];
      }
  }


  output[a_idx] = acc;
}

__kernel void matmul_grad_b(
        __global float* downstream_gradients,
        __global float* a,
        uint a_width,
        uint a_height,
        __global float* b,
        uint b_width,
        uint b_height,
        __global float* output,
        uint n
) {
    // Each element in B is multiplied by every element in a col of A
    //
    // E.g. the top left element in B is multiplied by every element in the first
    // col of A
    //
    // Each element modifies the downstream gradient at the location by
    // out(a_row, b_col)
    //
    // Each relevant output ends up in the equivalent col of the output, at the row
    // of A we were multiplying by

    uint global_id = get_global_id(0);
    if (global_id >= n) return;

    uint b_row = global_id / b_width;
    uint b_col = global_id % b_width;
    uint a_col = b_row % a_width;

    uint mat_idx = global_id / b_width / b_height;
    __global float* downstream_grad_base = downstream_gradients + mat_idx * b_width * a_height;

    float acc = 0.0;
    for (uint i = 0; i < a_height; i++) {
        uint a_idx = i * a_width + a_col;
        uint downstream_idx = i * b_width + b_col;

        acc += a[a_idx] * downstream_grad_base[downstream_idx];
    }

    output[global_id] = acc;
}

__kernel void transpose(
    __global float* in,
    __global float* out,
    uint in_w,
    uint in_h
) {

    uint global_id = get_global_id(0);
    if (global_id >= in_w * in_h) return;

    uint out_x = global_id % in_h;
    uint out_y = global_id / in_h;

    out[global_id] = *(in + out_x * in_w + out_y);
}
