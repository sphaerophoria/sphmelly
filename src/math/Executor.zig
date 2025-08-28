const std = @import("std");
const cl = @import("../cl.zig");
const math = @import("../math.zig");

pub const Tensor = math.Tensor(cl.Executor.Buffer);
pub const TensorSlice = math.TensorSlice(cl.Executor.Buffer);
const Executor = @This();

pub const sum_program_content = @embedFile("sum.cl");
pub const matmul_program_content = @embedFile("matmul.cl");
pub const sigmoid_program_content = @embedFile("sigmoid.cl");
pub const relu_program_content = @embedFile("relu.cl");
pub const add_splat_program_content = @embedFile("add_splat.cl");
pub const gt_program_content = @embedFile("gt.cl");
pub const squared_err_program_content = @embedFile("squared_err.cl");
pub const mul_program_content = @embedFile("mul.cl");
pub const rand_program_content = @embedFile("rand.cl");
pub const conv_program_content = @embedFile("conv.cl");

matmul_kernel: cl.Executor.Kernel,
matmul_grad_a_kernel: cl.Executor.Kernel,
matmul_grad_b_kernel: cl.Executor.Kernel,
sigmoid_kernel: cl.Executor.Kernel,
sigmoid_grad_kernel: cl.Executor.Kernel,
relu_kernel: cl.Executor.Kernel,
relu_grad_kernel: cl.Executor.Kernel,
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
masked_conv_kernel: cl.Executor.Kernel,
conv_many_kernel: cl.Executor.Kernel,
conv_many_grad_kernel_pass1_kernel: cl.Executor.Kernel,
conv_many_grad_kernel_pass2_kernel: cl.Executor.Kernel,
conv_many_grad_img_kernel: cl.Executor.Kernel,
conv_many_make_grad_mirrored_kernel_kernel: cl.Executor.Kernel,
transpose_kernel: cl.Executor.Kernel,
executor: *cl.Executor,

pub fn init(cl_alloc: *cl.Alloc, executor: *cl.Executor) !Executor {
    const sum_program = try executor.createProgram(cl_alloc, sum_program_content);
    const add_assign_kernel = try sum_program.createKernel(cl_alloc, "add_assign");

    const matmul_program = try executor.createProgram(cl_alloc, matmul_program_content);
    const matmul_kernel = try matmul_program.createKernel(cl_alloc, "matmul");
    const matmul_grad_a_kernel = try matmul_program.createKernel(cl_alloc, "matmul_grad_a");
    const matmul_grad_b_kernel = try matmul_program.createKernel(cl_alloc, "matmul_grad_b");
    const transpose_kernel = try matmul_program.createKernel(cl_alloc, "transpose");

    const sigmoid_program = try executor.createProgram(cl_alloc, sigmoid_program_content);
    const sigmoid_kernel = try sigmoid_program.createKernel(cl_alloc, "sigmoid");
    const sigmoid_grad_kernel = try sigmoid_program.createKernel(cl_alloc, "sigmoid_grad");

    const relu_program = try executor.createProgram(cl_alloc, relu_program_content);
    const relu_kernel = try relu_program.createKernel(cl_alloc, "relu");
    const relu_grad_kernel = try relu_program.createKernel(cl_alloc, "relu_grad");

    const add_splat_program = try executor.createProgram(cl_alloc, add_splat_program_content);
    const add_splat_kernel = try add_splat_program.createKernel(cl_alloc, "add_splat_outer");
    const add_splat_grad_a_kernel = try add_splat_program.createKernel(cl_alloc, "add_splat_outer_grad_a");

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

    const conv_program = try executor.createProgram(cl_alloc, conv_program_content);
    const masked_conv_kernel = try conv_program.createKernel(cl_alloc, "masked_conv");
    const conv_many_kernel = try conv_program.createKernel(cl_alloc, "conv_many");
    const conv_many_grad_kernel_pass1_kernel = try conv_program.createKernel(cl_alloc, "conv_many_grad_kernel_pass1");
    const conv_many_grad_kernel_pass2_kernel = try conv_program.createKernel(cl_alloc, "conv_many_grad_kernel_pass2");
    const conv_many_grad_img_kernel = try conv_program.createKernel(cl_alloc, "conv_many_grad_img");
    const conv_many_make_grad_mirrored_kernel_kernel = try conv_program.createKernel(cl_alloc, "make_grad_mirrored_kernel");
    return .{
        .matmul_kernel = matmul_kernel,
        .matmul_grad_a_kernel = matmul_grad_a_kernel,
        .matmul_grad_b_kernel = matmul_grad_b_kernel,
        .sigmoid_kernel = sigmoid_kernel,
        .sigmoid_grad_kernel = sigmoid_grad_kernel,
        .relu_kernel = relu_kernel,
        .relu_grad_kernel = relu_grad_kernel,
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
        .masked_conv_kernel = masked_conv_kernel,
        .conv_many_kernel = conv_many_kernel,
        .conv_many_grad_kernel_pass1_kernel = conv_many_grad_kernel_pass1_kernel,
        .conv_many_grad_kernel_pass2_kernel = conv_many_grad_kernel_pass2_kernel,
        .conv_many_grad_img_kernel = conv_many_grad_img_kernel,
        .conv_many_make_grad_mirrored_kernel_kernel = conv_many_make_grad_mirrored_kernel_kernel,
        .transpose_kernel = transpose_kernel,
        .executor = executor,
    };
}

pub fn addAssign(self: Executor, cl_alloc: *cl.Alloc, a: Tensor, b: Tensor) !void {
    if (!a.dims.eql(b.dims)) {
        std.log.err("{any} does not match {any}\n", .{ a.dims, b.dims });
        return error.InvalidDims;
    }

    const n = a.dims.numElems();
    try self.executor.executeKernelUntracked(cl_alloc, self.add_assign_kernel, n, &.{
        .{ .buf = a.buf },
        .{ .buf = b.buf },
        .{ .uint = n },
    });
}

pub fn mulScalar(self: Executor, cl_alloc: *cl.Alloc, a: Tensor, b: f32) !Tensor {
    const ret = try self.createTensorUninitialized(cl_alloc, a.dims);
    const n = a.dims.numElems();

    try self.executor.executeKernelUntracked(cl_alloc, self.mul_scalar_kernel, n, &.{
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
    try self.executor.executeKernelUntracked(cl_alloc, self.rand_kernel, n, &.{
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

pub fn randGaussian(self: Executor, cl_alloc: *cl.Alloc, dims_in: anytype, stddev: f32, source: *math.RandSource) !Tensor {
    const out_dims = try math.TensorDims.init(cl_alloc.heap(), dims_in);

    const ret = try self.executor.createBuffer(cl_alloc, .read_write, out_dims.byteSize());

    const n = out_dims.numElems();
    try self.executor.executeKernelUntracked(cl_alloc, self.gaussian_kernel, n, &.{
        .{ .buf = ret },
        .{ .ulong = source.ctr },
        .{ .uint = source.seed },
        .{ .uint = n },
        .{ .float = stddev },
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
    try self.executor.executeKernelUntracked(cl_alloc, self.gaussian_noise_kernel, n, &.{
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
    const out_dims_slice: []u32 = try cl_alloc.heap().alloc(u32, in.dims.len() - 1);
    var out_dim_idx: usize = 0;
    for (0..in.dims.len()) |i| {
        if (i == dim) continue;
        out_dims_slice[out_dim_idx] = in.dims.get(i);
        out_dim_idx += 1;
    }

    const out_dims = math.TensorDims.initRef(out_dims_slice);

    const ret = try self.executor.createBuffer(cl_alloc, .read_write, out_dims.byteSize());

    const n = out_dims.numElems();
    try self.executor.executeKernelUntracked(cl_alloc, self.gt_kernel, n, &.{
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
    try self.executor.executeKernelUntracked(cl_alloc, self.squared_err_kernel, n, &.{
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
    try self.executor.executeKernelUntracked(cl_alloc, self.squared_err_grad_kernel, n, &.{
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

pub fn maskedConv(self: Executor, cl_alloc: *cl.Alloc, img: Tensor, mask: Tensor, img_kernels: Tensor) !Tensor {
    if (!img.dims.eql(mask.dims)) {
        return error.InvalidDims;
    }

    if (img.dims.len() != 3 or img.dims.get(2) != img_kernels.dims.get(2)) {
        return error.InvalidDims;
    }

    if (img_kernels.dims.len() != 3 or img_kernels.dims.get(0) != img_kernels.dims.get(1) or img_kernels.dims.get(0) % 2 != 1) {
        return error.InvalidDims;
    }

    const ret = try self.createTensorUninitialized(cl_alloc, img.dims);

    const n = img.dims.numElems();
    try self.executor.executeKernelUntracked(cl_alloc, self.masked_conv_kernel, n, &.{
        .{ .buf = img.buf },
        .{ .buf = mask.buf },
        .{ .buf = ret.buf },
        .{ .uint = img.dims.get(0) },
        .{ .uint = img.dims.get(1) },
        .{ .uint = n },
        .{ .buf = img_kernels.buf },
        .{ .uint = img_kernels.dims.get(0) },
    });

    return ret;
}

pub fn Deferred(comptime T: type) type {
    return struct {
        event: cl.Executor.Event,
        val: T,
    };
}

pub fn createTensor(self: Executor, cl_alloc: *cl.Alloc, initial_data: []const f32, dims_in: []const u32) !Deferred(Tensor) {
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

pub fn createTensorFilled(self: Executor, cl_alloc: *cl.Alloc, dims_in: anytype, val: f32) !Tensor {
    const dims = try math.TensorDims.init(cl_alloc.heap(), dims_in);
    const buf = try self.executor.createBuffer(cl_alloc, .read_write, dims.byteSize());
    try self.executor.fillBuffer(buf, val, dims.byteSize());

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

fn validateAddSplatOuterDims(a: Tensor, b: Tensor) !void {
    if (a.dims.len() != b.dims.len() - 1) {
        return error.InvalidDims;
    }

    for (a.dims.inner, b.dims.inner[0 .. b.dims.len() - 1]) |a_dim, b_dim| {
        if (a_dim != b_dim) return error.InvalidDims;
    }
}

// Add a to b, where a matches b's shape, but with one smaller dimension, a
// repeats for each element in b
pub fn addSplatOuter(self: Executor, cl_alloc: *cl.Alloc, a: Tensor, b: Tensor) !Tensor {
    try validateAddSplatOuterDims(a, b);

    const out = try self.executor.createBuffer(cl_alloc, .read_write, b.dims.byteSize());

    const n = b.dims.numElems();
    try self.executor.executeKernelUntracked(cl_alloc, self.add_splat_kernel, n, &.{
        .{ .buf = a.buf },
        .{ .buf = b.buf },
        .{ .uint = a.dims.numElems() },
        .{ .buf = out },
        .{ .uint = n },
    });

    return .{
        .buf = out,
        .dims = try b.dims.clone(cl_alloc.heap()),
    };
}

pub fn addSplatOuterGrad(self: Executor, cl_alloc: *cl.Alloc, downstream_gradients: Tensor, a: Tensor, b: Tensor) ![2]Tensor {
    try validateAddSplatOuterDims(a, b);

    if (!b.dims.eql(downstream_gradients.dims)) {
        return error.InvalidDims;
    }

    const a_grads = try self.createTensorUninitialized(cl_alloc, a.dims);

    const n = a.dims.numElems();
    try self.executor.executeKernelUntracked(cl_alloc, self.add_splat_grad_a_kernel, n, &.{
        .{ .buf = downstream_gradients.buf },
        .{
            .uint = downstream_gradients.dims.numElems(),
        },
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
    const ts = tensor.asSlice();
    const res = try self.sliceToCpuDeferred(alloc, scratch_cl, ts);
    try res.event.wait();
    return res.val;
}

pub fn sliceToCpuDeferred(self: Executor, data_alloc: std.mem.Allocator, event_alloc: *cl.Alloc, tensor: TensorSlice) !Deferred([]f32) {
    const res_cpu = try data_alloc.alloc(f32, tensor.dims.numElems());

    const event = try self.executor.readBuffer(event_alloc, tensor.buf, tensor.elem_offs * @sizeOf(f32), std.mem.sliceAsBytes(res_cpu));

    return .{
        .event = event,
        .val = res_cpu,
    };
}

pub fn transpose(self: Executor, cl_alloc: *cl.Alloc, in: Tensor) !Tensor {
    if (in.dims.len() != 2) return error.Unimplemented;

    const out_dims = try math.TensorDims.init(cl_alloc.heap(), &.{ in.dims.get(1), in.dims.get(0) });
    const out = try self.createTensorUninitialized(cl_alloc, out_dims);

    try self.executor.executeKernelUntracked(cl_alloc, self.transpose_kernel, out.dims.numElems(), &.{
        .{ .buf = in.buf },
        .{ .buf = out.buf },
        .{ .uint = in.dims.get(0) },
        .{ .uint = in.dims.get(1) },
    });

    return out;
}

fn validateMatmulDims(a: Tensor, b: Tensor) !void {
    if (a.dims.len() != 2 or b.dims.len() != 3) {
        return error.InvalidMatMul;
    }

    if (a.dims.get(0) != b.dims.get(1)) {
        std.log.err("{} cannot matmul {}\n", .{ a.dims, b.dims });
        return error.InvalidDims;
    }
}

pub fn matmul(self: Executor, cl_alloc: *cl.Alloc, a: Tensor, b: Tensor) !Tensor {
    try validateMatmulDims(a, b);

    const out_dims = try math.TensorDims.init(cl_alloc.heap(), &.{ b.dims.get(0), a.dims.get(1), b.dims.get(2) });

    const ret = try self.executor.createBuffer(cl_alloc, .read_write, out_dims.byteSize());

    const n = out_dims.numElems();
    try self.executor.executeKernelUntracked(cl_alloc, self.matmul_kernel, n, &.{
        .{ .buf = a.buf },
        .{ .uint = a.dims.get(0) },
        .{ .uint = a.dims.get(1) },
        .{ .buf = b.buf },
        .{ .uint = b.dims.get(0) },
        .{ .uint = b.dims.get(1) },
        .{ .buf = ret },
        .{ .uint = n },
    });

    return .{
        .buf = ret,
        .dims = out_dims,
    };
}

pub fn matmulGrad(self: Executor, cl_alloc: *cl.Alloc, downstream_gradients: Tensor, a: Tensor, b: Tensor) ![2]Tensor {
    try validateMatmulDims(a, b);

    if (downstream_gradients.dims.len() != 3) {
        return error.InvalidDims;
    }

    if (downstream_gradients.dims.get(0) != b.dims.get(0) or
        downstream_gradients.dims.get(1) != a.dims.get(1) or
        downstream_gradients.dims.get(2) != b.dims.get(2))
    {
        return error.InvalidDims;
    }

    const a_grad = try self.createTensorUninitialized(cl_alloc, a.dims);
    const b_grad = try self.createTensorUninitialized(cl_alloc, b.dims);

    const a_n = a_grad.dims.numElems();
    try self.executor.executeKernelUntracked(cl_alloc, self.matmul_grad_a_kernel, a_n, &.{
        .{ .buf = downstream_gradients.buf },
        .{ .uint = downstream_gradients.dims.get(2) },
        .{ .buf = a.buf },
        .{ .uint = a.dims.get(0) },
        .{ .uint = a.dims.get(1) },
        .{ .buf = b.buf },
        .{ .uint = b.dims.get(0) },
        .{ .buf = a_grad.buf },
        .{ .uint = a_n },
    });

    const b_n = b_grad.dims.numElems();
    try self.executor.executeKernelUntracked(cl_alloc, self.matmul_grad_b_kernel, b_n, &.{
        .{ .buf = downstream_gradients.buf },
        .{ .buf = a.buf },
        .{ .uint = a.dims.get(0) },
        .{ .uint = a.dims.get(1) },
        .{ .buf = b.buf },
        .{ .uint = b.dims.get(0) },
        .{ .uint = b.dims.get(1) },
        .{ .buf = b_grad.buf },
        .{ .uint = b_n },
    });

    return .{ a_grad, b_grad };
}

pub fn sigmoid(self: Executor, cl_alloc: *cl.Alloc, in: Tensor) !Tensor {
    const ret = try self.executor.createBuffer(cl_alloc, .read_write, in.dims.byteSize());

    const n = in.dims.numElems();
    try self.executor.executeKernelUntracked(cl_alloc, self.sigmoid_kernel, n, &.{
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
        std.log.err("downstream {any} does not match in {any}\n", .{ downstream_grad.dims, in.dims });
        return error.InvalidDims;
    }

    const ret = try self.executor.createBuffer(cl_alloc, .read_write, in.dims.byteSize());

    const n = in.dims.numElems();
    try self.executor.executeKernelUntracked(cl_alloc, self.sigmoid_grad_kernel, n, &.{
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

pub fn relu(self: Executor, cl_alloc: *cl.Alloc, in: Tensor) !Tensor {
    const ret = try self.executor.createBuffer(cl_alloc, .read_write, in.dims.byteSize());

    const n = in.dims.numElems();
    try self.executor.executeKernelUntracked(cl_alloc, self.relu_kernel, n, &.{
        .{ .buf = in.buf },
        .{ .buf = ret },
        .{ .uint = n },
    });

    return .{
        .buf = ret,
        .dims = try in.dims.clone(cl_alloc.heap()),
    };
}

pub fn reluGrad(self: Executor, cl_alloc: *cl.Alloc, downstream_grad: Tensor, in: Tensor) !Tensor {
    if (!downstream_grad.dims.eql(in.dims)) {
        std.log.err("downstream {any} does not match in {any}\n", .{ downstream_grad.dims, in.dims });
        return error.InvalidDims;
    }

    const ret = try self.executor.createBuffer(cl_alloc, .read_write, in.dims.byteSize());

    const n = in.dims.numElems();
    try self.executor.executeKernelUntracked(cl_alloc, self.relu_grad_kernel, n, &.{
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

pub fn convMany(self: Executor, cl_alloc: *cl.Alloc, in: Tensor, kernel: Tensor) !Tensor {
    // in dims (w, h, in_c, n)
    // kernel dims (kw, kh, in_c, out_c)
    // ret (w, h, out_c, n)

    if (in.dims.len() != 4) {
        return error.InvalidDims;
    }

    if (kernel.dims.len() != 4) {
        return error.InvalidDims;
    }

    if (kernel.dims.get(2) != in.dims.get(2)) {
        return error.InvalidDims;
    }

    const out_dims = try math.TensorDims.init(cl_alloc.heap(), &.{ in.dims.get(0), in.dims.get(1), kernel.dims.get(3), in.dims.get(3) });
    const ret = try self.createTensorUninitialized(cl_alloc, out_dims);

    const n = out_dims.numElems();
    try self.executor.executeKernelUntracked(cl_alloc, self.conv_many_kernel, n, &.{
        .{ .buf = in.buf },
        .{ .buf = kernel.buf },
        .{ .buf = ret.buf },
        .{ .uint = in.dims.get(0) },
        .{ .uint = in.dims.get(1) },
        .{ .uint = in.dims.get(2) },
        .{ .uint = in.dims.get(3) },
        .{ .uint = kernel.dims.get(0) },
        .{ .uint = kernel.dims.get(1) },
        .{ .uint = kernel.dims.get(3) },
    });

    return ret;
}

pub fn convManyGrad(self: Executor, cl_alloc: *cl.Alloc, downstream_gradients: Tensor, img_input: Tensor, kernel_input: Tensor) ![2]Tensor {
    // FIXME: Validate :D

    const a_grad = try self.createTensorUninitialized(cl_alloc, img_input.dims);
    const b_grad = try self.createTensorUninitialized(cl_alloc, kernel_input.dims);

    const grad_kernel = try self.createTensorUninitialized(cl_alloc, &.{
        kernel_input.dims.get(0),
        kernel_input.dims.get(1),
        kernel_input.dims.get(3),
        kernel_input.dims.get(2),
    });
    const kernel_num_elems = kernel_input.dims.numElems();
    try self.executor.executeKernelUntracked(cl_alloc, self.conv_many_make_grad_mirrored_kernel_kernel, kernel_num_elems, &.{
        .{ .buf = kernel_input.buf },
        .{ .buf = grad_kernel.buf },
        .{ .uint = kernel_input.dims.get(0) },
        .{ .uint = kernel_input.dims.get(1) },
        .{ .uint = kernel_input.dims.get(2) },
        .{ .uint = kernel_input.dims.get(3) },
    });

    const a_n = a_grad.dims.numElems();
    try self.executor.executeKernelUntracked(cl_alloc, self.conv_many_grad_img_kernel, a_n, &.{
        .{ .buf = downstream_gradients.buf },
        .{ .buf = grad_kernel.buf },
        .{ .buf = a_grad.buf },
        .{ .uint = img_input.dims.get(0) },
        .{ .uint = img_input.dims.get(1) },
        .{ .uint = img_input.dims.get(2) },
        .{ .uint = img_input.dims.get(3) },
        .{ .uint = kernel_input.dims.get(0) },
        .{ .uint = kernel_input.dims.get(1) },
        .{ .uint = kernel_input.dims.get(3) },
    });

    const b_grad_pass1 = try self.createTensorUninitialized(cl_alloc, &.{
        b_grad.dims.get(0),
        b_grad.dims.get(1),
        b_grad.dims.get(2),
        b_grad.dims.get(3),
        downstream_gradients.dims.get(3),
    });

    const b_pass1_n = b_grad_pass1.dims.numElems();
    try self.executor.executeKernelUntracked(cl_alloc, self.conv_many_grad_kernel_pass1_kernel, b_pass1_n, &.{
        .{ .buf = downstream_gradients.buf },
        .{ .buf = img_input.buf },
        .{ .buf = b_grad_pass1.buf },
        .{ .uint = img_input.dims.get(0) },
        .{ .uint = img_input.dims.get(1) },
        .{ .uint = img_input.dims.get(2) },
        .{ .uint = img_input.dims.get(3) },
        .{ .uint = kernel_input.dims.get(0) },
        .{ .uint = kernel_input.dims.get(1) },
        .{ .uint = kernel_input.dims.get(3) },
    });

    const b_pass2_n = b_grad.dims.numElems();
    try self.executor.executeKernelUntracked(cl_alloc, self.conv_many_grad_kernel_pass2_kernel, b_pass2_n, &.{
        .{ .buf = b_grad_pass1.buf },
        .{ .buf = b_grad.buf },
        .{ .uint = b_grad_pass1.dims.get(0) },
        .{ .uint = b_grad_pass1.dims.get(1) },
        .{ .uint = b_grad_pass1.dims.get(2) },
        .{ .uint = b_grad_pass1.dims.get(3) },
        .{ .uint = b_grad_pass1.dims.get(4) },
    });

    return .{ a_grad, b_grad };
}

const ClExecutorFixture = struct {
    cl_alloc: *cl.Alloc,
    alloc_checkpoint: cl.Alloc.Checkpoint,
    executor: *cl.Executor,
    cl_math: *Executor,
    rand_source: math.RandSource,

    const Global = struct {
        buf: []u8,
        cl_alloc: cl.Alloc,
        executor: cl.Executor,
        cl_math: Executor,
    };

    var global: ?Global = null;

    fn init() !ClExecutorFixture {
        const g = try initGlobal();

        return .{
            .cl_alloc = &g.cl_alloc,
            .alloc_checkpoint = g.cl_alloc.checkpoint(),
            .executor = &g.executor,
            .cl_math = &g.cl_math,
            .rand_source = .{
                .ctr = 0,
                .seed = 0,
            },
        };
    }

    fn deinit(self: *ClExecutorFixture) void {
        self.cl_alloc.reset(self.alloc_checkpoint);
    }

    fn initGlobal() !*Global {
        if (global) |*g| return g;

        global = undefined;
        const g = &global.?;

        g.buf = try std.heap.page_allocator.alloc(u8, 10 * 1024 * 1024);
        errdefer std.heap.page_allocator.free(g.buf);

        try g.cl_alloc.initPinned(g.buf);
        errdefer g.cl_alloc.deinit();

        g.executor = try cl.Executor.init(g.cl_alloc.heap(), .non_profiling);
        errdefer g.executor.deinit();

        g.cl_math = try Executor.init(&g.cl_alloc, &g.executor);
        return g;
    }
};

test "addAssign" {
    var fixture = try ClExecutorFixture.init();
    defer fixture.deinit();

    const a = try fixture.cl_math.createTensorUntracked(fixture.cl_alloc, &.{
        1, 2, 3, 4,
    }, &.{4});

    const b = try fixture.cl_math.createTensorUntracked(fixture.cl_alloc, &.{
        5, 6, 7, 8,
    }, &.{4});

    try fixture.cl_math.addAssign(fixture.cl_alloc, a, b);

    const actual = try fixture.cl_math.toCpu(fixture.cl_alloc.heap(), fixture.cl_alloc, a);
    const expected: []const f32 = &.{ 6, 8, 10, 12 };

    for (expected, actual) |ex, ac| {
        try std.testing.expectApproxEqAbs(ex, ac, 0.0001);
    }
}

test "mulScalar" {
    var fixture = try ClExecutorFixture.init();
    defer fixture.deinit();

    const a = try fixture.cl_math.createTensorUntracked(fixture.cl_alloc, &.{
        1, 2,
        3, 4,
    }, &.{ 2, 2 });

    const res = try fixture.cl_math.mulScalar(fixture.cl_alloc, a, 4);
    const res_cpu = try fixture.cl_math.toCpu(fixture.cl_alloc.heap(), fixture.cl_alloc, res);

    const expected: []const f32 = &.{ 4, 8, 12, 16 };

    for (expected, res_cpu) |ex, ac| {
        try std.testing.expectApproxEqAbs(ex, ac, 0.0001);
    }
}

test "matmul" {
    var fixture = try ClExecutorFixture.init();
    defer fixture.deinit();

    const a = try fixture.cl_math.createTensorUntracked(fixture.cl_alloc, &.{
        1, 2, 3,
        4, 5, 6,
    }, &.{ 3, 2 });

    const b = try fixture.cl_math.createTensorUntracked(fixture.cl_alloc, &.{
        7,  8,  9,  10,
        11, 12, 13, 14,
        15, 16, 17, 18,

        19, 20, 21, 22,
        23, 24, 25, 26,
        27, 28, 29, 30,
    }, &.{ 4, 3, 2 });

    const c = try fixture.cl_math.matmul(fixture.cl_alloc, a, b);

    try std.testing.expectEqual(4, c.dims.get(0));
    try std.testing.expectEqual(2, c.dims.get(1));
    try std.testing.expectEqual(2, c.dims.get(2));

    const expected: []const f32 = &.{
        74,  80,  86,  92,
        173, 188, 203, 218,

        146, 152, 158, 164,
        353, 368, 383, 398,
    };

    var actual: [16]f32 = undefined;
    const finish = try fixture.executor.readBuffer(fixture.cl_alloc, c.buf, 0, std.mem.sliceAsBytes(&actual));
    try finish.wait();

    for (expected, actual) |ex, ac| {
        try std.testing.expectApproxEqAbs(ex, ac, 0.0001);
    }
}

test "matmulGrad" {
    var fixture = try ClExecutorFixture.init();
    defer fixture.deinit();

    const grads = try fixture.cl_math.createTensorUntracked(
        fixture.cl_alloc,
        &.{
            1,  2,  3,
            4,  5,  6,
            7,  8,  9,

            22, 23, 24,
            25, 26, 27,
            28, 29, 30,
        },
        &.{ 3, 3, 2 },
    );

    const a = try fixture.cl_math.createTensorUntracked(
        fixture.cl_alloc,
        &.{
            10, 11,
            12, 13,
            14, 15,
        },
        &.{ 2, 3 },
    );

    const b = try fixture.cl_math.createTensorUntracked(
        fixture.cl_alloc,
        &.{
            16, 17, 18,
            19, 20, 21,

            31, 32, 33,
            34, 35, 36,
        },
        &.{ 3, 2, 2 },
    );

    const a_grads, const b_grads = try fixture.cl_math.matmulGrad(fixture.cl_alloc, grads, a, b);

    const a_expected: []const f32 = &.{
        1 * 16 + 2 * 17 + 3 * 18 + 22 * 31 + 23 * 32 + 24 * 33,
        1 * 19 + 2 * 20 + 3 * 21 + 22 * 34 + 23 * 35 + 24 * 36,

        4 * 16 + 5 * 17 + 6 * 18 + 25 * 31 + 26 * 32 + 27 * 33,
        4 * 19 + 5 * 20 + 6 * 21 + 25 * 34 + 26 * 35 + 27 * 36,

        7 * 16 + 8 * 17 + 9 * 18 + 28 * 31 + 29 * 32 + 30 * 33,
        7 * 19 + 8 * 20 + 9 * 21 + 28 * 34 + 29 * 35 + 30 * 36,
    };

    const b_expected: []const f32 = &.{
        1 * 10 + 4 * 12 + 7 * 14,
        2 * 10 + 5 * 12 + 8 * 14,
        3 * 10 + 6 * 12 + 9 * 14,

        1 * 11 + 4 * 13 + 7 * 15,
        2 * 11 + 5 * 13 + 8 * 15,
        3 * 11 + 6 * 13 + 9 * 15,

        22 * 10 + 25 * 12 + 28 * 14,
        23 * 10 + 26 * 12 + 29 * 14,
        24 * 10 + 27 * 12 + 30 * 14,

        22 * 11 + 25 * 13 + 28 * 15,
        23 * 11 + 26 * 13 + 29 * 15,
        24 * 11 + 27 * 13 + 30 * 15,
    };

    const a_actual = try fixture.cl_math.toCpu(fixture.cl_alloc.heap(), fixture.cl_alloc, a_grads);

    for (a_expected, a_actual) |ex, ac| {
        try std.testing.expectApproxEqAbs(ex, ac, 0.001);
    }

    const b_actual = try fixture.cl_math.toCpu(fixture.cl_alloc.heap(), fixture.cl_alloc, b_grads);

    for (b_expected, b_actual) |ex, ac| {
        try std.testing.expectApproxEqAbs(ex, ac, 0.001);
    }
}

test "sigmoid" {
    var fixture = try ClExecutorFixture.init();
    defer fixture.deinit();

    const input: []const f32 = &.{
        -10.0,
        -5.0,
        0.0,
        5.0,
        10.0,
    };

    const in_gpu = try fixture.cl_math.createTensorUntracked(fixture.cl_alloc, input, &.{input.len});
    const out_gpu = try fixture.cl_math.sigmoid(fixture.cl_alloc, in_gpu);

    var actual: [5]f32 = undefined;
    const final = try fixture.executor.readBuffer(fixture.cl_alloc, out_gpu.buf, 0, std.mem.sliceAsBytes(&actual));
    try final.wait();

    const expected: []const f32 = &.{ 4.53978687e-05, 6.69285092e-03, 5.00000000e-01, 9.93307149e-01, 9.99954602e-01 };

    for (expected, actual) |ex, ac| {
        try std.testing.expectApproxEqAbs(ex, ac, 0.0001);
    }
}

test "sigmoidGrad" {
    var fixture = try ClExecutorFixture.init();
    defer fixture.deinit();

    const downstream_grads = try fixture.cl_math.createTensorUntracked(
        fixture.cl_alloc,
        &.{
            1, 2, 3, 4, 5,
        },
        &.{5},
    );

    const inputs = try fixture.cl_math.createTensorUntracked(
        fixture.cl_alloc,
        &.{ -10, -5, 0, 5, 10 },
        &.{5},
    );

    const gradients = try fixture.cl_math.sigmoidGrad(
        fixture.cl_alloc,
        downstream_grads,
        inputs,
    );

    const actual = try fixture.cl_math.toCpu(fixture.cl_alloc.heap(), fixture.cl_alloc, gradients);
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

test "relu" {
    var fixture = try ClExecutorFixture.init();
    defer fixture.deinit();

    const in_cpu = &.{
        -2,  0.1, -0.1,
        0.5, 3,   0.0,
    };
    const in_gpu = try fixture.cl_math.createTensorUntracked(fixture.cl_alloc, in_cpu, &.{ 3, 2 });

    const out_gpu = try fixture.cl_math.relu(fixture.cl_alloc, in_gpu);

    try std.testing.expect(in_gpu.dims.eql(out_gpu.dims));

    const out_cpu = try fixture.cl_math.toCpu(fixture.cl_alloc.heap(), fixture.cl_alloc, out_gpu);

    const expected: []const f32 = &.{
        0,   0.1, 0,
        0.5, 3,   0.0,
    };

    for (expected, out_cpu) |ex, ac| {
        try std.testing.expectApproxEqAbs(ex, ac, 0.0001);
    }
}

test "reluGrad" {
    var fixture = try ClExecutorFixture.init();
    defer fixture.deinit();

    const in_cpu = &.{
        -2,  0.1, -0.1,
        0.5, 3,   0.0,
    };

    const downstream_grads_cpu = &.{
        1, 2, 3,
        4, 5, 6,
    };

    const in_gpu = try fixture.cl_math.createTensorUntracked(fixture.cl_alloc, in_cpu, &.{ 3, 2 });
    const downstream_grads_gpu = try fixture.cl_math.createTensorUntracked(fixture.cl_alloc, downstream_grads_cpu, &.{ 3, 2 });

    const out_gpu = try fixture.cl_math.reluGrad(fixture.cl_alloc, downstream_grads_gpu, in_gpu);

    try std.testing.expect(in_gpu.dims.eql(out_gpu.dims));

    const out_cpu = try fixture.cl_math.toCpu(fixture.cl_alloc.heap(), fixture.cl_alloc, out_gpu);

    const expected: []const f32 = &.{
        0, 2, 0,
        4, 5, 0,
    };

    for (expected, out_cpu) |ex, ac| {
        try std.testing.expectApproxEqAbs(ex, ac, 0.0001);
    }
}

test "addSplatOuter" {
    var fixture = try ClExecutorFixture.init();
    defer fixture.deinit();

    const a = try fixture.cl_math.createTensorUntracked(fixture.cl_alloc, &.{ 1, 2, 3, 4 }, &.{4});
    const b = try fixture.cl_math.createTensorUntracked(fixture.cl_alloc, &.{
        5,  6,  7,  8,
        9,  10, 11, 12,
        13, 14, 15, 16,
        17, 18, 19, 20,
    }, &.{ 4, 4 });

    const expected: []const f32 = &.{
        6,  8,  10, 12,
        10, 12, 14, 16,
        14, 16, 18, 20,
        18, 20, 22, 24,
    };

    const actual_gpu = try fixture.cl_math.addSplatOuter(fixture.cl_alloc, a, b);
    var actual: [16]f32 = undefined;
    const final = try fixture.executor.readBuffer(fixture.cl_alloc, actual_gpu.buf, 0, std.mem.sliceAsBytes(&actual));
    try final.wait();

    for (expected, actual) |ex, ac| {
        try std.testing.expectApproxEqAbs(ex, ac, 0.0001);
    }
}

test "addSplatOuterGrad" {
    var fixture = try ClExecutorFixture.init();
    defer fixture.deinit();

    const a = try fixture.cl_math.createTensorUntracked(fixture.cl_alloc, &.{ 1, 2, 3, 4 }, &.{4});
    const b = try fixture.cl_math.createTensorUntracked(fixture.cl_alloc, &.{
        5,  6,  7,  8,
        9,  10, 11, 12,
        13, 14, 15, 16,
        17, 18, 19, 20,
    }, &.{ 4, 4 });

    const downstream_gradients = try fixture.cl_math.createTensorUntracked(fixture.cl_alloc, &.{
        21, 22, 23, 24,
        25, 26, 27, 28,
        29, 30, 31, 32,
        33, 34, 35, 36,
    }, &.{ 4, 4 });

    const expected_a: []const f32 = &.{
        21 + 25 + 29 + 33,
        22 + 26 + 30 + 34,
        23 + 27 + 31 + 35,
        24 + 28 + 32 + 36,
    };

    const expected_b: []const f32 = &.{
        21, 22, 23, 24,
        25, 26, 27, 28,
        29, 30, 31, 32,
        33, 34, 35, 36,
    };

    const a_grad, const b_grad = try fixture.cl_math.addSplatOuterGrad(fixture.cl_alloc, downstream_gradients, a, b);

    const a_grad_cpu = try fixture.cl_math.toCpu(fixture.cl_alloc.heap(), fixture.cl_alloc, a_grad);
    const b_grad_cpu = try fixture.cl_math.toCpu(fixture.cl_alloc.heap(), fixture.cl_alloc, b_grad);

    for (expected_a, a_grad_cpu) |ex, ac| {
        try std.testing.expectApproxEqAbs(ex, ac, 0.0001);
    }

    for (expected_b, b_grad_cpu) |ex, ac| {
        try std.testing.expectApproxEqAbs(ex, ac, 0.0001);
    }
}

test "gt" {
    var fixture = try ClExecutorFixture.init();
    defer fixture.deinit();

    const vals = try fixture.cl_math.createTensorUntracked(
        fixture.cl_alloc,
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

    const res = try fixture.cl_math.gt(fixture.cl_alloc, vals, 2);
    try std.testing.expectEqualSlices(u32, &.{ 3, 3 }, res.dims.inner);

    const expected: []const f32 = &.{
        0.0, 0.0, 0.0,
        0.0, 0.0, 1.0,
        1.0, 1.0, 1.0,
    };

    const output = try fixture.cl_math.toCpu(fixture.cl_alloc.heap(), fixture.cl_alloc, res);

    for (expected, output) |ec, o| {
        try std.testing.expectApproxEqAbs(ec, o, 0.001);
    }
}

test "squaredErr" {
    var fixture = try ClExecutorFixture.init();
    defer fixture.deinit();

    const a = try fixture.cl_math.createTensorUntracked(
        fixture.cl_alloc,
        &.{ 1.0, 2.0, 2.0 },
        &.{3},
    );

    const b = try fixture.cl_math.createTensorUntracked(
        fixture.cl_alloc,
        &.{ 3.0, 2.0, 1.0 },
        &.{3},
    );

    const res = try fixture.cl_math.squaredErr(fixture.cl_alloc, a, b);
    try std.testing.expectEqualSlices(u32, &.{3}, res.dims.inner);

    const expected: []const f32 = &.{
        4.0, 0.0, 1.0,
    };

    const output = try fixture.cl_math.toCpu(fixture.cl_alloc.heap(), fixture.cl_alloc, res);

    for (expected, output) |ec, o| {
        try std.testing.expectApproxEqAbs(ec, o, 0.001);
    }
}

test "squaredErrGrad" {
    var fixture = try ClExecutorFixture.init();
    defer fixture.deinit();

    const a = try fixture.cl_math.createTensorUntracked(
        fixture.cl_alloc,
        &.{ 1.0, 2.0, 2.0 },
        &.{3},
    );

    const b = try fixture.cl_math.createTensorUntracked(
        fixture.cl_alloc,
        &.{ 3.0, 2.0, 1.0 },
        &.{3},
    );

    const downstream_gradients = try fixture.cl_math.createTensorUntracked(fixture.cl_alloc, &.{ 4, 5, 6 }, &.{3});

    const a_grad, const b_grad = try fixture.cl_math.squaredErrGrad(fixture.cl_alloc, downstream_gradients, a, b);

    const a_grad_cpu = try fixture.cl_math.toCpu(fixture.cl_alloc.heap(), fixture.cl_alloc, a_grad);
    const b_grad_cpu = try fixture.cl_math.toCpu(fixture.cl_alloc.heap(), fixture.cl_alloc, b_grad);

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
    var fixture = try ClExecutorFixture.init();
    defer fixture.deinit();

    const num_nums = 1000000;
    const numbers_gpu = try fixture.cl_math.rand(fixture.cl_alloc, &.{num_nums}, &fixture.rand_source);
    const numbers_cpu = try fixture.cl_math.toCpu(fixture.cl_alloc.heap(), fixture.cl_alloc, numbers_gpu);

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
    var fixture = try ClExecutorFixture.init();
    defer fixture.deinit();

    const num_nums = 1000000;
    const numbers_gpu = try fixture.cl_math.randGaussian(fixture.cl_alloc, &.{num_nums}, 1.0, &fixture.rand_source);
    const numbers_cpu = try fixture.cl_math.toCpu(fixture.cl_alloc.heap(), fixture.cl_alloc, numbers_gpu);

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

test "convMany" {
    var fixture = try ClExecutorFixture.init();
    defer fixture.deinit();

    // (5, 5, 2, 1)
    const in_img: []const f32 = &.{
        // c1
        0, 0, 1, 1, 1,
        0, 0, 1, 1, 1,
        1, 1, 1, 1, 1,
        1, 1, 1, 1, 1,
        1, 1, 1, 1, 1,

        // c2
        0, 0, 0, 1, 1,
        0, 0, 0, 1, 1,
        1, 1, 1, 1, 1,
        0, 0, 0, 1, 1,
        0, 0, 0, 1, 1,
    };

    // (3, 3, 2, 2)
    // Sum horizontal and vertical gradients
    const in_kernel: []const f32 = &.{
        // o1, c1
        0,  -1, 0,
        0,  0,  0,
        0,  1,  0,

        // o1, c2
        0,  0,  0,
        -1, 0,  1,
        0,  0,  0,

        //// o2, c1
        0,  0,  0,
        -1, 0,  1,
        0,  0,  0,

        //// o2, c2
        0,  -1, 0,
        0,  0,  0,
        0,  1,  0,
    };

    const expected_output: []const f32 = &.{
        0,  0,  1,  1, 0,
        1,  1,  1,  1, 0,
        1,  1,  0,  0, 0,
        0,  0,  1,  1, 0,
        0,  0,  1,  1, 0,

        0,  1,  1,  0, 0,
        1,  2,  2,  0, 0,
        0,  0,  0,  0, 0,
        -1, -1, -1, 0, 0,
        0,  0,  0,  0, 0,
    };

    const img_gpu = try fixture.cl_math.createTensorUntracked(fixture.cl_alloc, in_img, &.{ 5, 5, 2, 1 });
    const kernel_gpu = try fixture.cl_math.createTensorUntracked(fixture.cl_alloc, in_kernel, &.{ 3, 3, 2, 2 });

    const ret = try fixture.cl_math.convMany(fixture.cl_alloc, img_gpu, kernel_gpu);
    try std.testing.expectEqualSlices(u32, ret.dims.inner, &.{ 5, 5, 2, 1 });

    const ret_cpu = try fixture.cl_math.toCpu(fixture.cl_alloc.heap(), fixture.cl_alloc, ret);

    for (ret_cpu, expected_output) |ac, ex| {
        try std.testing.expectApproxEqAbs(ex, ac, 0.0001);
    }
}

const JsonTensor = struct {
    shape: []const u32,
    data: []const f32,

    fn toTensor(self: JsonTensor, alloc: *cl.Alloc, executor: *Executor) !Tensor {
        return try executor.createTensorUntracked(alloc, self.data, self.shape);
    }
};

const ConvDefinition = struct {
    name: []const u8,
    img: JsonTensor,
    kernel: JsonTensor,
    downstream_grad: JsonTensor,
    kernel_grad: JsonTensor,
    img_grad: JsonTensor,
    output: JsonTensor,
};

test "convMany2" {
    var fixture = try ClExecutorFixture.init();
    defer fixture.deinit();

    const test_data = try std.json.parseFromSliceLeaky([]ConvDefinition, fixture.cl_alloc.heap(), @embedFile("conv_test_data"), .{ .ignore_unknown_fields = true });

    for (test_data) |test_elem| {
        const img = try test_elem.img.toTensor(fixture.cl_alloc, fixture.cl_math);
        const kernel = try test_elem.kernel.toTensor(fixture.cl_alloc, fixture.cl_math);

        const output = try fixture.cl_math.convMany(fixture.cl_alloc, img, kernel);
        try std.testing.expect(output.dims.eql(try .init(fixture.cl_alloc.heap(), test_elem.output.shape)));

        const downstream_grad = try test_elem.downstream_grad.toTensor(fixture.cl_alloc, fixture.cl_math);

        const img_grads, const kernel_grads = try fixture.cl_math.convManyGrad(fixture.cl_alloc, downstream_grad, img, kernel);

        try std.testing.expect(img_grads.dims.eql(try .init(fixture.cl_alloc.heap(), test_elem.img_grad.shape)));
        try std.testing.expect(kernel_grads.dims.eql(try .init(fixture.cl_alloc.heap(), test_elem.kernel_grad.shape)));

        const output_cpu = try fixture.cl_math.toCpu(fixture.cl_alloc.heap(), fixture.cl_alloc, output);
        const kg_cpu = try fixture.cl_math.toCpu(fixture.cl_alloc.heap(), fixture.cl_alloc, kernel_grads);
        const ig_cpu = try fixture.cl_math.toCpu(fixture.cl_alloc.heap(), fixture.cl_alloc, img_grads);

        for (output_cpu, test_elem.output.data) |ac, ex| {
            try std.testing.expectApproxEqAbs(ex, ac, 0.01);
        }

        for (kg_cpu, test_elem.kernel_grad.data) |ac, ex| {
            try std.testing.expectApproxEqAbs(ex, ac, 0.01);
        }

        for (ig_cpu, test_elem.img_grad.data) |ac, ex| {
            try std.testing.expectApproxEqAbs(ex, ac, 0.01);
        }
    }
}

test "transpose" {
    var fixture = try ClExecutorFixture.init();
    defer fixture.deinit();

    const in_cpu = &.{
        1, 2, 3,
        4, 5, 6,
    };

    const in_gpu = try fixture.cl_math.createTensorUntracked(fixture.cl_alloc, in_cpu, &.{ 3, 2 });
    const out_gpu = try fixture.cl_math.transpose(fixture.cl_alloc, in_gpu);
    try std.testing.expectEqual(out_gpu.dims.get(0), 2);
    try std.testing.expectEqual(out_gpu.dims.get(1), 3);

    const expected: []const f32 = &.{
        1, 4,
        2, 5,
        3, 6,
    };

    const out_cpu = try fixture.cl_math.toCpu(fixture.cl_alloc.heap(), fixture.cl_alloc, out_gpu);

    for (expected, out_cpu) |ex, ac| {
        try std.testing.expectApproxEqAbs(ex, ac, 0.0001);
    }
}
