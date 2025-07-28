int ceilDiv(int num, int denom) {
    return (num + (denom - 1)) / denom;
}

struct reduce_it {
    int remaining;
    bool should_run;
    int other_id;
};

struct reduce_it reduce_it_init(int num_elems) {
    int local_size = get_local_size(0);
    int initial_size = min(local_size * 2, num_elems);

    return (struct reduce_it){
        initial_size,
        // Other fields set on first call to _next()
    };
}

bool reduce_it_next(struct reduce_it* it, int local_id) {
    if (it->remaining <= 1) return false;

    it->remaining  = ceilDiv(it->remaining, 2);
    it->should_run = local_id < it->remaining / 2;
    int add_offset = ceilDiv(it->remaining, 2);
    it->other_id = local_id + add_offset;

    return true;
}

__kernel void sum(__global float* numbers, __global float* output, __local float* tmp, int n, int log) {
    int global_id = get_global_id(0);
    int initial_idx = global_id * 2;
    int initial_neighbor = global_id * 2 + 1;
    int local_id = get_local_id(0);
    int local_size = get_local_size(0);

    if (log) {
        tmp[local_id] = 99999.0;
    }

    if (initial_idx < n) {
        if (initial_neighbor < n) {
            tmp[local_id] = numbers[initial_idx] + numbers[initial_neighbor];
        } else {
            tmp[local_id] = numbers[initial_idx];
        }
    }

    struct reduce_it reduce_it = reduce_it_init(n);
    if (local_id == 0) {
        printf("tmp size %d local size %d n %d", reduce_it.remaining, local_size, n);
    }
    if (initial_idx < n && log > 0) {
        printf("tmp buffer %d == %f", local_id, tmp[local_id]);
    }

    while (reduce_it_next(&reduce_it, local_id)) {
        barrier(CLK_LOCAL_MEM_FENCE);

        if (reduce_it.should_run) {
            if (log > 0) {
                printf("id %d %d (< %d ) Adding %f to %f\n", local_id, global_id, reduce_it.remaining, tmp[local_id], tmp[reduce_it.other_id]);
            }
            tmp[local_id] = tmp[local_id] + tmp[reduce_it.other_id];
        }
    }

    if (local_id == 0) {
        output[global_id / local_size] = tmp[0];
    }
}
