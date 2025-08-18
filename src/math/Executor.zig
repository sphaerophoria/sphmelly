const std = @import("std");
const cl = @import("../cl.zig");
const math = @import("../math.zig");

pub const Tensor = math.Tensor(cl.Executor.Buffer);
const Executor = @This();

const sum_program_content = @embedFile("sum.cl");
const matmul_program_content = @embedFile("matmul.cl");
const sigmoid_program_content = @embedFile("sigmoid.cl");
const add_splat_program_content = @embedFile("add_splat.cl");
const gt_program_content = @embedFile("gt.cl");
const squared_err_program_content = @embedFile("squared_err.cl");
const mul_program_content = @embedFile("mul.cl");
const rand_program_content = @embedFile("rand.cl");

matmul_kernel: cl.Executor.Kernel,
matmul_grad_a_kernel: cl.Executor.Kernel,
matmul_grad_b_kernel: cl.Executor.Kernel,
sigmoid_kernel: cl.Executor.Kernel,
sigmoid_grad_kernel: cl.Executor.Kernel,
add_splat_kernel: cl.Executor.Kernel,
add_splat_grad_a_kernel: cl.Executor.Kernel,
gt_kernel: cl.Executor.Kernel,
squared_err_kernel: cl.Executor.Kernel,
squared_err_grad_kernel: cl.Executor.Kernel,
add_assign_kernel: cl.Executor.Kernel,
mul_scalar_kernel: cl.Executor.Kernel,
rand_kernel: cl.Executor.Kernel,
gaussian_kernel: cl.Executor.Kernel,
gaussian_noise_kernel: cl.Executor.Kernel,
executor: cl.Executor,

pub fn init(cl_alloc: *cl.Alloc, executor: cl.Executor) !Executor {
    const sum_program = try executor.createProgram(cl_alloc, sum_program_content);
    const add_assign_kernel = try sum_program.createKernel(cl_alloc, "add_assign");

    const matmul_program = try executor.createProgram(cl_alloc, matmul_program_content);
    const matmul_kernel = try matmul_program.createKernel(cl_alloc, "matmul");
    const matmul_grad_a_kernel = try matmul_program.createKernel(cl_alloc, "matmul_grad_a");
    const matmul_grad_b_kernel = try matmul_program.createKernel(cl_alloc, "matmul_grad_b");

    const sigmoid_program = try executor.createProgram(cl_alloc, sigmoid_program_content);
    const sigmoid_kernel = try sigmoid_program.createKernel(cl_alloc, "sigmoid");
    const sigmoid_grad_kernel = try sigmoid_program.createKernel(cl_alloc, "sigmoid_grad");

    const add_splat_program = try executor.createProgram(cl_alloc, add_splat_program_content);
    const add_splat_kernel = try add_splat_program.createKernel(cl_alloc, "add_splat_horizontal");
    const add_splat_grad_a_kernel = try add_splat_program.createKernel(cl_alloc, "add_splat_horizontal_grad_a");

    const mul_program = try executor.createProgram(cl_alloc, mul_program_content);
    const mul_scalar_kernel = try mul_program.createKernel(cl_alloc, "mul_scalar");

    const gt_program = try executor.createProgram(cl_alloc, gt_program_content);
    const gt_kernel = try gt_program.createKernel(cl_alloc, "gt");

    const squared_err_program = try executor.createProgram(cl_alloc, squared_err_program_content);
    const squared_err_kernel = try squared_err_program.createKernel(cl_alloc, "squared_err");
    const squared_err_grad_kernel = try squared_err_program.createKernel(cl_alloc, "squared_err_grad");

    const rand_program = try executor.createProgram(cl_alloc, rand_program_content);
    const rand_kernel = try rand_program.createKernel(cl_alloc, "rand");
    const gaussian_kernel = try rand_program.createKernel(cl_alloc, "gaussian");
    const gaussian_noise_kernel = try rand_program.createKernel(cl_alloc, "gaussian_noise");

    return .{
        .matmul_kernel = matmul_kernel,
        .matmul_grad_a_kernel = matmul_grad_a_kernel,
        .matmul_grad_b_kernel = matmul_grad_b_kernel,
        .sigmoid_kernel = sigmoid_kernel,
        .sigmoid_grad_kernel = sigmoid_grad_kernel,
        .add_splat_kernel = add_splat_kernel,
        .add_splat_grad_a_kernel = add_splat_grad_a_kernel,
        .gt_kernel = gt_kernel,
        .squared_err_kernel = squared_err_kernel,
        .squared_err_grad_kernel = squared_err_grad_kernel,
        .add_assign_kernel = add_assign_kernel,
        .mul_scalar_kernel = mul_scalar_kernel,
        .rand_kernel = rand_kernel,
        .gaussian_kernel = gaussian_kernel,
        .gaussian_noise_kernel = gaussian_noise_kernel,
        .executor = executor,
    };
}

pub fn addAssign(self: Executor, a: Tensor, b: Tensor) !void {
    if (!a.dims.eql(b.dims)) {
        std.log.err("{any} does not match {any}\n", .{ a.dims, b.dims });
        return error.InvalidDims;
    }

    const n = a.dims.numElems();
    try executeKernel(self.executor, self.add_assign_kernel, n, &.{
        .{ .buf = a.buf },
        .{ .buf = b.buf },
        .{ .uint = n },
    });
}

pub fn mulScalar(self: Executor, cl_alloc: *cl.Alloc, a: Tensor, b: f32) !Tensor {
    const ret = try self.createTensorUninitialized(cl_alloc, a.dims);
    const n = a.dims.numElems();

    try executeKernel(self.executor, self.mul_scalar_kernel, n, &.{
        .{ .buf = a.buf },
        .{ .float = b },
        .{ .buf = ret.buf },
        .{ .uint = n },
    });

    return ret;
}

pub fn reshape(_: Executor, cl_alloc: *cl.Alloc, val: Tensor, new_dims_in: anytype) !Tensor {
    const new_dims = switch (@TypeOf(new_dims_in)) {
        math.TensorDims => try new_dims_in.clone(cl_alloc.heap()),
        else => try math.TensorDims.init(cl_alloc.heap(), new_dims_in),
    };

    if (new_dims.numElems() != val.dims.numElems()) {
        return error.InvalidDims;
    }

    return .{
        .buf = val.buf,
        .dims = new_dims,
    };
}

pub fn rand(self: Executor, cl_alloc: *cl.Alloc, dims_in: anytype, source: *math.RandSource) !Tensor {
    const out_dims = try math.TensorDims.init(cl_alloc.heap(), dims_in);

    const ret = try self.executor.createBuffer(cl_alloc, .read_write, out_dims.byteSize());

    const n = out_dims.numElems();
    try executeKernel(self.executor, self.rand_kernel, n, &.{
        .{ .buf = ret },
        .{ .ulong = source.ctr },
        .{ .uint = source.seed },
        .{ .uint = n },
    });

    source.ctr += n;

    return .{
        .buf = ret,
        .dims = out_dims,
    };
}

pub fn randGaussian(self: Executor, cl_alloc: *cl.Alloc, dims_in: anytype, source: *math.RandSource) !Tensor {
    const out_dims = try math.TensorDims.init(cl_alloc.heap(), dims_in);

    const ret = try self.executor.createBuffer(cl_alloc, .read_write, out_dims.byteSize());

    const n = out_dims.numElems();
    try executeKernel(self.executor, self.gaussian_kernel, n, &.{
        .{ .buf = ret },
        .{ .ulong = source.ctr },
        .{ .uint = source.seed },
        .{ .uint = n },
    });

    // Box muller algo uses 2 PRNG uniform inputs per 2 outputs, however our
    // impl discards the second one, resulting in 2 consumed numbers per number
    source.ctr += 2 * n;

    return .{
        .buf = ret,
        .dims = out_dims,
    };
}

pub fn gaussianNoise(self: Executor, cl_alloc: *cl.Alloc, input: Tensor, seed: u32, start_count: u64) !Tensor {
    const out_dims = try input.dims.clone(cl_alloc.heap());

    const ret = try self.executor.createBuffer(cl_alloc, .read_write, out_dims.byteSize());

    const n = out_dims.numElems();
    try executeKernel(self.executor, self.gaussian_noise_kernel, n, &.{
        .{ .buf = input.buf },
        .{ .buf = ret },
        .{ .ulong = start_count },
        .{ .uint = seed },
        .{ .uint = n },
    });

    return .{
        .buf = ret,
        .dims = out_dims,
    };
}

pub fn gt(self: Executor, cl_alloc: *cl.Alloc, in: Tensor, dim: u32) !Tensor {
    if (in.dims.get(dim) != 2) {
        return error.InvalidDims;
    }

    const stride = in.dims.stride(dim);
    const out_dims = try math.TensorDims.initEmpty(cl_alloc.heap(), in.dims.len() - 1);
    var out_dim_idx: usize = 0;
    for (0..in.dims.len()) |i| {
        if (i == dim) continue;
        out_dims.getPtr(out_dim_idx).* = in.dims.get(i);
        out_dim_idx += 1;
    }

    const ret = try self.executor.createBuffer(cl_alloc, .read_write, out_dims.byteSize());

    const n = out_dims.numElems();
    try executeKernel(self.executor, self.gt_kernel, n, &.{
        .{ .buf = in.buf },
        .{ .uint = stride },
        .{ .buf = ret },
        .{ .uint = n },
    });

    return .{
        .buf = ret,
        .dims = out_dims,
    };
}

pub fn squaredErr(self: *Executor, cl_alloc: *cl.Alloc, a: Tensor, b: Tensor) !Tensor {
    if (!a.dims.eql(b.dims)) {
        std.log.err("{any} does not match {any}\n", .{ a.dims, b.dims });
        return error.InvalidDims;
    }

    const ret = try self.executor.createBuffer(cl_alloc, .read_write, a.dims.byteSize());

    const n = a.dims.numElems();
    try executeKernel(self.executor, self.squared_err_kernel, n, &.{
        .{ .buf = a.buf },
        .{ .buf = b.buf },
        .{ .buf = ret },
        .{ .uint = n },
    });

    return .{
        .buf = ret,
        .dims = try a.dims.clone(cl_alloc.heap()),
    };
}

pub fn squaredErrGrad(self: Executor, cl_alloc: *cl.Alloc, downstream_grad: Tensor, a: Tensor, b: Tensor) ![2]Tensor {
    if (!a.dims.eql(b.dims) or !a.dims.eql(downstream_grad.dims)) {
        return error.InvalidDims;
    }

    const a_grad = try self.createTensorUninitialized(cl_alloc, a.dims);
    const b_grad = try self.createTensorUninitialized(cl_alloc, a.dims);

    const n = a.dims.numElems();
    try executeKernel(self.executor, self.squared_err_grad_kernel, n, &.{
        .{ .buf = downstream_grad.buf },
        .{ .buf = a.buf },
        .{ .buf = b.buf },
        .{ .buf = a_grad.buf },
        .{ .buf = b_grad.buf },
        .{ .uint = n },
    });

    return .{
        a_grad, b_grad,
    };
}

const TensorRes = struct {
    event: cl.Executor.Event,
    val: Tensor,
};

pub fn createTensor(self: Executor, cl_alloc: *cl.Alloc, initial_data: []const f32, dims_in: []const u32) !TensorRes {
    const params = try self.createTensorCommon(cl_alloc, initial_data, dims_in);
    const event = try self.executor.writeBuffer(cl_alloc, params.buf, std.mem.sliceAsBytes(initial_data));

    return .{
        .event = event,
        .val = .{
            .buf = params.buf,
            .dims = params.dims,
        },
    };
}

pub fn createTensorUntracked(self: Executor, cl_alloc: *cl.Alloc, initial_data: []const f32, dims_in: []const u32) !Tensor {
    const params = try self.createTensorCommon(cl_alloc, initial_data, dims_in);
    try self.executor.writeBufferUntracked(params.buf, std.mem.sliceAsBytes(initial_data));

    return .{
        .buf = params.buf,
        .dims = params.dims,
    };
}

pub fn createTensorUninitialized(self: Executor, cl_alloc: *cl.Alloc, dims_in: anytype) !Tensor {
    const dims = try math.TensorDims.init(cl_alloc.heap(), dims_in);
    const buf = try self.executor.createBuffer(cl_alloc, .read_write, dims.byteSize());

    return .{
        .buf = buf,
        .dims = dims,
    };
}

const TensorCommonParams = struct {
    dims: math.TensorDims,
    buf: cl.Executor.Buffer,
};

pub fn createTensorCommon(self: Executor, cl_alloc: *cl.Alloc, initial_data: []const f32, dims_in: []const u32) !TensorCommonParams {
    const dims = try math.TensorDims.init(cl_alloc.heap(), dims_in);

    if (initial_data.len % dims.numElems() != 0) {
        return error.InvalidDims;
    }

    const buf = try self.executor.createBuffer(cl_alloc, .read_write, initial_data.len * @sizeOf(f32));

    return .{
        .dims = dims,
        .buf = buf,
    };
}

pub fn addSplatHorizontal(self: Executor, cl_alloc: *cl.Alloc, a: Tensor, b: Tensor) !Tensor {
    if (a.dims.len() != 1) {
        return error.InvalidDims;
    }

    if (a.dims.get(0) != b.dims.get(1)) {
        return error.InvalidDims;
    }

    const out = try self.executor.createBuffer(cl_alloc, .read_write, b.dims.byteSize());

    const n = b.dims.numElems();
    try executeKernel(self.executor, self.add_splat_kernel, n, &.{
        .{ .buf = a.buf },
        .{ .buf = b.buf },
        .{ .uint = b.dims.get(0) },
        .{ .buf = out },
        .{ .uint = n },
    });

    return .{
        .buf = out,
        .dims = try b.dims.clone(cl_alloc.heap()),
    };
}

pub fn addSplatHorizontalGrad(self: Executor, cl_alloc: *cl.Alloc, downstream_gradients: Tensor, a: Tensor, b: Tensor) ![2]Tensor {
    if (a.dims.len() != 1) {
        return error.InvalidDims;
    }

    if (a.dims.get(0) != b.dims.get(1)) {
        return error.InvalidDims;
    }

    if (!b.dims.eql(downstream_gradients.dims)) {
        return error.InvalidDims;
    }

    const a_grads = try self.createTensorUninitialized(cl_alloc, a.dims);

    const n = a.dims.numElems();
    try executeKernel(self.executor, self.add_splat_grad_a_kernel, n, &.{
        .{ .buf = downstream_gradients.buf },
        .{ .uint = downstream_gradients.dims.get(0) },
        .{ .buf = a.buf },
        .{ .buf = a_grads.buf },
        .{ .uint = a.dims.numElems() },
    });

    return .{
        a_grads,
        // B changes effect downstream 1-1
        downstream_gradients,
    };
}

pub fn toCpu(self: Executor, alloc: std.mem.Allocator, scratch_cl: *cl.Alloc, tensor: Tensor) ![]f32 {
    const res_cpu = try alloc.alloc(f32, tensor.dims.numElems());

    const event = try self.executor.readBuffer(scratch_cl, tensor.buf, std.mem.sliceAsBytes(res_cpu));
    try event.wait();

    return res_cpu;
}

pub fn matmul(self: Executor, cl_alloc: *cl.Alloc, a: Tensor, b: Tensor) !Tensor {
    if (a.dims.len() != 2 or b.dims.len() != 2) {
        return error.InvalidMatMul;
    }

    if (a.dims.get(0) != b.dims.get(1)) {
        std.log.err("{} cannot matmul {}\n", .{ a.dims, b.dims });
        return error.InvalidDims;
    }

    const out_dims = try math.TensorDims.init(cl_alloc.heap(), &.{ b.dims.get(0), a.dims.get(1) });

    const ret = try self.executor.createBuffer(cl_alloc, .read_write, out_dims.byteSize());

    const n = out_dims.numElems();
    try executeKernel(self.executor, self.matmul_kernel, n, &.{
        .{ .buf = a.buf },
        .{ .uint = a.dims.get(0) },
        .{ .uint = a.dims.get(1) },
        .{ .buf = b.buf },
        .{ .uint = b.dims.get(0) },
        .{ .buf = ret },
        .{ .uint = n },
    });

    return .{
        .buf = ret,
        .dims = out_dims,
    };
}

pub fn matmulGrad(self: Executor, cl_alloc: *cl.Alloc, downstream_gradients: Tensor, a: Tensor, b: Tensor) ![2]Tensor {
    const a_grad = try self.createTensorUninitialized(cl_alloc, a.dims);
    const b_grad = try self.createTensorUninitialized(cl_alloc, b.dims);

    const a_n = a_grad.dims.numElems();
    try executeKernel(self.executor, self.matmul_grad_a_kernel, a_n, &.{
        .{ .buf = downstream_gradients.buf },
        .{ .buf = a.buf },
        .{ .uint = a.dims.get(0) },
        .{ .buf = b.buf },
        .{ .uint = b.dims.get(0) },
        .{ .buf = a_grad.buf },
        .{ .uint = a_n },
    });

    const b_n = b_grad.dims.numElems();
    try executeKernel(self.executor, self.matmul_grad_b_kernel, b_n, &.{
        .{ .buf = downstream_gradients.buf },
        .{ .buf = a.buf },
        .{ .uint = a.dims.get(0) },
        .{ .uint = a.dims.get(1) },
        .{ .buf = b.buf },
        .{ .uint = b.dims.get(0) },
        .{ .buf = b_grad.buf },
        .{ .uint = b_n },
    });

    return .{ a_grad, b_grad };
}

pub fn sigmoid(self: Executor, cl_alloc: *cl.Alloc, in: Tensor) !Tensor {
    const ret = try self.executor.createBuffer(cl_alloc, .read_write, in.dims.byteSize());

    const n = in.dims.numElems();
    try executeKernel(self.executor, self.sigmoid_kernel, n, &.{
        .{ .buf = in.buf },
        .{ .buf = ret },
        .{ .uint = n },
    });

    return .{
        .buf = ret,
        .dims = try in.dims.clone(cl_alloc.heap()),
    };
}

pub fn sigmoidGrad(self: Executor, cl_alloc: *cl.Alloc, downstream_grad: Tensor, in: Tensor) !Tensor {
    if (!downstream_grad.dims.eql(in.dims)) {
        return error.InvalidDims;
    }

    const ret = try self.executor.createBuffer(cl_alloc, .read_write, in.dims.byteSize());

    const n = in.dims.numElems();
    try executeKernel(self.executor, self.sigmoid_grad_kernel, n, &.{
        .{ .buf = downstream_grad.buf },
        .{ .buf = in.buf },
        .{ .buf = ret },
        .{ .uint = n },
    });

    return .{
        .buf = ret,
        .dims = try in.dims.clone(cl_alloc.heap()),
    };
}

fn executeKernel(executor: cl.Executor, kernel: cl.Executor.Kernel, n: u32, args: []const cl.Executor.Kernel.Arg) !void {
    for (args, 0..) |arg, i| {
        try kernel.setArg(@intCast(i), arg);
    }

    try executor.executeKernelUntracked(kernel, n);
}

const ClExecutorFixture = struct {
    buf: []u8,
    cl_alloc: cl.Alloc,
    executor: cl.Executor,
    cl_math: Executor,
    rand_source: math.RandSource,

    fn initPinned(self: *ClExecutorFixture) !void {
        self.buf = try std.heap.page_allocator.alloc(u8, 10 * 1024 * 1024);

        self.executor = try cl.Executor.init();
        errdefer self.executor.deinit();

        try self.cl_alloc.initPinned(self.buf);
        errdefer self.cl_alloc.deinit();

        self.cl_math = try Executor.init(&self.cl_alloc, self.executor);

        self.rand_source = .{
            .ctr = 0,
            .seed = 0,
        };
    }

    fn deinit(self: *ClExecutorFixture) void {
        self.cl_alloc.deinit();
        self.executor.deinit();
        std.heap.page_allocator.free(self.buf);
    }
};

test "addAssign" {
    var fixture: ClExecutorFixture = undefined;
    try fixture.initPinned();
    defer fixture.deinit();

    const a = try fixture.cl_math.createTensorUntracked(&fixture.cl_alloc, &.{
        1, 2, 3, 4,
    }, &.{4});

    const b = try fixture.cl_math.createTensorUntracked(&fixture.cl_alloc, &.{
        5, 6, 7, 8,
    }, &.{4});

    try fixture.cl_math.addAssign(a, b);

    const actual = try fixture.cl_math.toCpu(fixture.cl_alloc.heap(), &fixture.cl_alloc, a);
    const expected: []const f32 = &.{ 6, 8, 10, 12 };

    for (expected, actual) |ex, ac| {
        try std.testing.expectApproxEqAbs(ex, ac, 0.0001);
    }
}

test "mulScalar" {
    var fixture: ClExecutorFixture = undefined;
    try fixture.initPinned();
    defer fixture.deinit();

    const a = try fixture.cl_math.createTensorUntracked(&fixture.cl_alloc, &.{
        1, 2,
        3, 4,
    }, &.{ 2, 2 });

    const res = try fixture.cl_math.mulScalar(&fixture.cl_alloc, a, 4);
    const res_cpu = try fixture.cl_math.toCpu(fixture.cl_alloc.heap(), &fixture.cl_alloc, res);

    const expected: []const f32 = &.{ 4, 8, 12, 16 };

    for (expected, res_cpu) |ex, ac| {
        try std.testing.expectApproxEqAbs(ex, ac, 0.0001);
    }
}

test "matmul" {
    var fixture: ClExecutorFixture = undefined;
    try fixture.initPinned();
    defer fixture.deinit();

    const a = try fixture.cl_math.createTensorUntracked(&fixture.cl_alloc, &.{
        1, 2, 3,
        4, 5, 6,
    }, &.{ 3, 2 });

    const b = try fixture.cl_math.createTensorUntracked(&fixture.cl_alloc, &.{
        7,  8,  9,  10,
        11, 12, 13, 14,
        15, 16, 17, 18,
    }, &.{ 4, 3 });

    const c = try fixture.cl_math.matmul(&fixture.cl_alloc, a, b);

    try std.testing.expectEqual(4, c.dims.get(0));
    try std.testing.expectEqual(2, c.dims.get(1));

    const expected: []const f32 = &.{
        74,  80,  86,  92,
        173, 188, 203, 218,
    };

    var actual: [8]f32 = undefined;
    const finish = try fixture.executor.readBuffer(&fixture.cl_alloc, c.buf, std.mem.sliceAsBytes(&actual));
    try finish.wait();

    for (expected, actual) |ex, ac| {
        try std.testing.expectApproxEqAbs(ex, ac, 0.0001);
    }
}

test "matmulGrad" {
    var fixture: ClExecutorFixture = undefined;
    try fixture.initPinned();
    defer fixture.deinit();

    const grads = try fixture.cl_math.createTensorUntracked(
        &fixture.cl_alloc,
        &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9 },
        &.{ 3, 3 },
    );

    const a = try fixture.cl_math.createTensorUntracked(
        &fixture.cl_alloc,
        &.{
            10, 11,
            12, 13,
            14, 15,
        },
        &.{ 2, 3 },
    );

    const b = try fixture.cl_math.createTensorUntracked(
        &fixture.cl_alloc,
        &.{
            16, 17, 18,
            19, 20, 21,
        },
        &.{ 3, 2 },
    );

    const a_grads, const b_grads = try fixture.cl_math.matmulGrad(&fixture.cl_alloc, grads, a, b);

    const a_expected: []const f32 = &.{
        1 * 16 + 2 * 17 + 3 * 18, 1 * 19 + 2 * 20 + 3 * 21,
        4 * 16 + 5 * 17 + 6 * 18, 4 * 19 + 5 * 20 + 6 * 21,
        7 * 16 + 8 * 17 + 9 * 18, 7 * 19 + 8 * 20 + 9 * 21,
    };

    const b_expected: []const f32 = &.{
        1 * 10 + 4 * 12 + 7 * 14, 2 * 10 + 5 * 12 + 8 * 14, 3 * 10 + 6 * 12 + 9 * 14,
        1 * 11 + 4 * 13 + 7 * 15, 2 * 11 + 5 * 13 + 8 * 15, 3 * 11 + 6 * 13 + 9 * 15,
    };

    const a_actual = try fixture.cl_math.toCpu(fixture.cl_alloc.heap(), &fixture.cl_alloc, a_grads);

    for (a_expected, a_actual) |ex, ac| {
        try std.testing.expectApproxEqAbs(ex, ac, 0.001);
    }

    const b_actual = try fixture.cl_math.toCpu(fixture.cl_alloc.heap(), &fixture.cl_alloc, b_grads);

    for (b_expected, b_actual) |ex, ac| {
        try std.testing.expectApproxEqAbs(ex, ac, 0.001);
    }
}

test "sigmoid" {
    var fixture: ClExecutorFixture = undefined;
    try fixture.initPinned();
    defer fixture.deinit();

    const input: []const f32 = &.{
        -10.0,
        -5.0,
        0.0,
        5.0,
        10.0,
    };

    const in_gpu = try fixture.cl_math.createTensorUntracked(&fixture.cl_alloc, input, &.{input.len});
    const out_gpu = try fixture.cl_math.sigmoid(&fixture.cl_alloc, in_gpu);

    var actual: [5]f32 = undefined;
    const final = try fixture.executor.readBuffer(&fixture.cl_alloc, out_gpu.buf, std.mem.sliceAsBytes(&actual));
    try final.wait();

    const expected: []const f32 = &.{ 4.53978687e-05, 6.69285092e-03, 5.00000000e-01, 9.93307149e-01, 9.99954602e-01 };

    for (expected, actual) |ex, ac| {
        try std.testing.expectApproxEqAbs(ex, ac, 0.0001);
    }
}

test "sigmoidGrad" {
    var fixture: ClExecutorFixture = undefined;
    try fixture.initPinned();
    defer fixture.deinit();

    const downstream_grads = try fixture.cl_math.createTensorUntracked(
        &fixture.cl_alloc,
        &.{
            1, 2, 3, 4, 5,
        },
        &.{5},
    );

    const inputs = try fixture.cl_math.createTensorUntracked(
        &fixture.cl_alloc,
        &.{ -10, -5, 0, 5, 10 },
        &.{5},
    );

    const gradients = try fixture.cl_math.sigmoidGrad(
        &fixture.cl_alloc,
        downstream_grads,
        inputs,
    );

    const actual = try fixture.cl_math.toCpu(fixture.cl_alloc.heap(), &fixture.cl_alloc, gradients);
    const expected: []const f32 = &.{
        0.0000453958 * 1,
        0.00664806 * 2,
        0.25 * 3,
        0.00664806 * 4,
        0.0000453958 * 5,
    };

    for (expected, actual) |ex, ac| {
        try std.testing.expectApproxEqAbs(ex, ac, 0.0001);
    }
}

test "addSplatHorizontal" {
    var fixture: ClExecutorFixture = undefined;
    try fixture.initPinned();
    defer fixture.deinit();

    const a = try fixture.cl_math.createTensorUntracked(&fixture.cl_alloc, &.{ 1, 2, 3, 4 }, &.{4});
    const b = try fixture.cl_math.createTensorUntracked(&fixture.cl_alloc, &.{
        5,  6,  7,  8,
        9,  10, 11, 12,
        13, 14, 15, 16,
        17, 18, 19, 20,
    }, &.{ 4, 4 });

    const expected: []const f32 = &.{
        6,  7,  8,  9,
        11, 12, 13, 14,
        16, 17, 18, 19,
        21, 22, 23, 24,
    };

    const actual_gpu = try fixture.cl_math.addSplatHorizontal(&fixture.cl_alloc, a, b);
    var actual: [16]f32 = undefined;
    const final = try fixture.executor.readBuffer(&fixture.cl_alloc, actual_gpu.buf, std.mem.sliceAsBytes(&actual));
    try final.wait();

    for (expected, actual) |ex, ac| {
        try std.testing.expectApproxEqAbs(ex, ac, 0.0001);
    }
}

test "addSplatHorizontalGrad" {
    var fixture: ClExecutorFixture = undefined;
    try fixture.initPinned();
    defer fixture.deinit();

    const a = try fixture.cl_math.createTensorUntracked(&fixture.cl_alloc, &.{ 1, 2, 3, 4 }, &.{4});
    const b = try fixture.cl_math.createTensorUntracked(&fixture.cl_alloc, &.{
        5,  6,  7,  8,
        9,  10, 11, 12,
        13, 14, 15, 16,
        17, 18, 19, 20,
    }, &.{ 4, 4 });

    const downstream_gradients = try fixture.cl_math.createTensorUntracked(&fixture.cl_alloc, &.{
        21, 22, 23, 24,
        25, 26, 27, 28,
        29, 30, 31, 32,
        33, 34, 35, 36,
    }, &.{ 4, 4 });

    const expected_a: []const f32 = &.{
        21 + 22 + 23 + 24,
        25 + 26 + 27 + 28,
        29 + 30 + 31 + 32,
        33 + 34 + 35 + 36,
    };

    const expected_b: []const f32 = &.{
        21, 22, 23, 24,
        25, 26, 27, 28,
        29, 30, 31, 32,
        33, 34, 35, 36,
    };

    const a_grad, const b_grad = try fixture.cl_math.addSplatHorizontalGrad(&fixture.cl_alloc, downstream_gradients, a, b);

    const a_grad_cpu = try fixture.cl_math.toCpu(fixture.cl_alloc.heap(), &fixture.cl_alloc, a_grad);
    const b_grad_cpu = try fixture.cl_math.toCpu(fixture.cl_alloc.heap(), &fixture.cl_alloc, b_grad);

    for (expected_a, a_grad_cpu) |ex, ac| {
        try std.testing.expectApproxEqAbs(ex, ac, 0.0001);
    }

    for (expected_b, b_grad_cpu) |ex, ac| {
        try std.testing.expectApproxEqAbs(ex, ac, 0.0001);
    }
}

test "gt" {
    var fixture: ClExecutorFixture = undefined;
    try fixture.initPinned();
    defer fixture.deinit();

    const vals = try fixture.cl_math.createTensorUntracked(
        &fixture.cl_alloc,
        &.{
            // Slice 1
            1, 2, 3,
            4, 5, 6,
            7, 8, 9,

            // Slice 2
            9, 8, 7,
            6, 5, 4,
            3, 2, 1,
        },
        &.{ 3, 3, 2 },
    );

    const res = try fixture.cl_math.gt(&fixture.cl_alloc, vals, 2);
    try std.testing.expectEqualSlices(u32, &.{ 3, 3 }, res.dims.inner);

    const expected: []const f32 = &.{
        0.0, 0.0, 0.0,
        0.0, 0.0, 1.0,
        1.0, 1.0, 1.0,
    };

    const output = try fixture.cl_math.toCpu(fixture.cl_alloc.heap(), &fixture.cl_alloc, res);

    for (expected, output) |ec, o| {
        try std.testing.expectApproxEqAbs(ec, o, 0.001);
    }
}

test "squaredErr" {
    var fixture: ClExecutorFixture = undefined;
    try fixture.initPinned();
    defer fixture.deinit();

    const a = try fixture.cl_math.createTensorUntracked(
        &fixture.cl_alloc,
        &.{ 1.0, 2.0, 2.0 },
        &.{3},
    );

    const b = try fixture.cl_math.createTensorUntracked(
        &fixture.cl_alloc,
        &.{ 3.0, 2.0, 1.0 },
        &.{3},
    );

    const res = try fixture.cl_math.squaredErr(&fixture.cl_alloc, a, b);
    try std.testing.expectEqualSlices(u32, &.{3}, res.dims.inner);

    const expected: []const f32 = &.{
        4.0, 0.0, 1.0,
    };

    const output = try fixture.cl_math.toCpu(fixture.cl_alloc.heap(), &fixture.cl_alloc, res);

    for (expected, output) |ec, o| {
        try std.testing.expectApproxEqAbs(ec, o, 0.001);
    }
}

test "squaredErrGrad" {
    var fixture: ClExecutorFixture = undefined;
    try fixture.initPinned();
    defer fixture.deinit();

    const a = try fixture.cl_math.createTensorUntracked(
        &fixture.cl_alloc,
        &.{ 1.0, 2.0, 2.0 },
        &.{3},
    );

    const b = try fixture.cl_math.createTensorUntracked(
        &fixture.cl_alloc,
        &.{ 3.0, 2.0, 1.0 },
        &.{3},
    );

    const downstream_gradients = try fixture.cl_math.createTensorUntracked(&fixture.cl_alloc, &.{ 4, 5, 6 }, &.{3});

    const a_grad, const b_grad = try fixture.cl_math.squaredErrGrad(&fixture.cl_alloc, downstream_gradients, a, b);

    const a_grad_cpu = try fixture.cl_math.toCpu(fixture.cl_alloc.heap(), &fixture.cl_alloc, a_grad);
    const b_grad_cpu = try fixture.cl_math.toCpu(fixture.cl_alloc.heap(), &fixture.cl_alloc, b_grad);

    const expected_a: []const f32 = &.{
        -16, 0, 12,
    };

    const expected_b: []const f32 = &.{
        16, 0, -12,
    };

    for (expected_a, a_grad_cpu) |ec, ac| {
        try std.testing.expectApproxEqAbs(ec, ac, 0.001);
    }

    for (expected_b, b_grad_cpu) |ec, ac| {
        try std.testing.expectApproxEqAbs(ec, ac, 0.001);
    }
}

fn calcChiSquared(actual_buckets: []const usize, expected_buckets: []const usize) f32 {
    std.debug.assert(actual_buckets.len == expected_buckets.len);

    var chi2: f32 = 0;

    for (actual_buckets, expected_buckets) |actual, expected| {
        const expected_f: f32 = @floatFromInt(expected);
        const actual_f: f32 = @floatFromInt(actual);
        var num = actual_f - expected_f;
        num *= num;
        num /= expected_f;
        chi2 += num;
    }

    return chi2;
}

// Empirically, our rng seems to be a "bad" fit for a uniform distribution
// Graphing a histogram of returned numbers manually, we see a "box" of
// numbers that are relatively flat, but with fairly high variance in Y (1%)
//
// AFAICT, this means that the chi2 stat will indicate that we are not a
// great fit. Which is true. HOWEVER if we plot a completely wrong curve
// (i.e. gaussian shifted into [0,1) range we get a very very different
// chi2 number (like 700k vs 1k)
//
// This indicates to me that this is still a valuable stat, but expecting
// an actual good fit is unreasonable. Testing our values against the zig
// default PRNG gives similar results (with 1M numbers 1104 for zig, 1107
// for us)
//
// Instead of picking a fit number from a chi2 table and probability value,
// just check that we are still within some reasonable value that we
// previously measured :)
const chi2_threshold = 1110.0;

test "rand" {
    var fixture: ClExecutorFixture = undefined;
    try fixture.initPinned();
    defer fixture.deinit();

    const num_nums = 1000000;
    const numbers_gpu = try fixture.cl_math.rand(&fixture.cl_alloc, &.{num_nums}, &fixture.rand_source);
    const numbers_cpu = try fixture.cl_math.toCpu(fixture.cl_alloc.heap(), &fixture.cl_alloc, numbers_gpu);

    const num_buckets = 1000;
    const expected: [num_buckets]usize = @splat(num_nums / num_buckets);
    var actual: [num_buckets]usize = @splat(0);
    for (numbers_cpu) |num| {
        const bucket_idx: usize = @intFromFloat(num * num_buckets);
        actual[bucket_idx] += 1;
    }

    const chi2 = calcChiSquared(&actual, &expected);
    try std.testing.expect(chi2 < chi2_threshold);
}

pub fn normalDistAt(x: f32) f32 {
    //https://en.wikipedia.org/wiki/Normal_distribution
    return std.math.exp(-(x * x) / 2) / std.math.sqrt(2 * std.math.pi);
}

test "gaussian" {
    var fixture: ClExecutorFixture = undefined;
    try fixture.initPinned();
    defer fixture.deinit();

    const num_nums = 1000000;
    const numbers_gpu = try fixture.cl_math.randGaussian(&fixture.cl_alloc, &.{num_nums}, &fixture.rand_source);
    const numbers_cpu = try fixture.cl_math.toCpu(fixture.cl_alloc.heap(), &fixture.cl_alloc, numbers_gpu);

    const num_buckets = 1000;
    var expected: [num_buckets]usize = undefined;
    var expected_sum: usize = 0;

    for (0..num_buckets) |i| {
        // [0, num_buckets] -> [-3, 3]
        // i * 6 / num_buckets - 3
        const i_f: f32 = @floatFromInt(i);
        const x = i_f * 6 / num_buckets - 3;
        expected[i] = @intFromFloat(@round(normalDistAt(x) * 6.0 / num_buckets * num_nums));
        expected_sum += expected[i];
    }

    var actual: [num_buckets]usize = @splat(0);
    for (numbers_cpu) |num| {
        // [-3, 3] -> [0, num_buckets]
        const bucket_idx_f: f32 = (num + 3) / 6 * num_buckets;
        if (bucket_idx_f >= num_buckets or bucket_idx_f < 0) continue;
        const bucket_idx: usize = @intFromFloat(bucket_idx_f);
        actual[bucket_idx] += 1;
    }

    const chi2 = calcChiSquared(&actual, &expected);
    try std.testing.expect(chi2 < chi2_threshold);
}
