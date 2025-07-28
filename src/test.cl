__kernel void add(__global float* as, __global float* bs, __global float* output, int n) {
	int i = get_global_id(0);

	if ((i >=0) && (i<n)) {
		output[i] = as[i] + bs[i];
	}
}

__kernel void mul(__global float* as, __global float* bs, __global float* output, int n) {
	int i = get_global_id(0);

	if ((i >=0) && (i<n)) {
		output[i] = as[i] * bs[i];
	}
}
