const std = @import("std");
const cl = @import("cl.zig");
const math = @import("math.zig");

math_executor: math.Executor,
module_gen_kernel: cl.Executor.Kernel,
barcode_gen_kernel: cl.Executor.Kernel,
sample_params_kernel: cl.Executor.Kernel,
generate_blur_kernels_kernel: cl.Executor.Kernel,

const barcode_gen_program_source = math.Executor.rand_program_content ++ @embedFile("BarcodeGen/generate.cl");

const BarcodeGen = @This();

pub const RandomizationParams = struct {
    x_offs_range: [2]f32,
    y_offs_range: [2]f32,
    x_scale_range: [2]f32,
    aspect_range: [2]f32,
    min_contrast: f32,
    perlin_grid_size_range: [2]u32,
    x_noise_multiplier_range: [2]f32,
    y_noise_multiplier_range: [2]f32,
    background_color_range: [2]f32,
    blur_stddev_range: [2]f32,
};

pub fn init(cl_alloc: *cl.Alloc, math_executor: math.Executor) !BarcodeGen {
    const program = try math_executor.executor.createProgram(cl_alloc, barcode_gen_program_source);
    const barcode_gen_kernel = try program.createKernel(cl_alloc, "generate_barcode");
    const module_gen_kernel = try program.createKernel(cl_alloc, "generate_module_patterns");
    const sample_params_kernel = try program.createKernel(cl_alloc, "sample_barcode_params");
    const generate_blur_kernels_kernel = try program.createKernel(cl_alloc, "generate_blur_kernels");
    return .{
        .math_executor = math_executor,
        .barcode_gen_kernel = barcode_gen_kernel,
        .module_gen_kernel = module_gen_kernel,
        .sample_params_kernel = sample_params_kernel,
        .generate_blur_kernels_kernel = generate_blur_kernels_kernel,
    };
}

const Bars = struct {
    imgs: math.Executor.Tensor,
    masks: math.Executor.Tensor,
    orientations: math.Executor.Tensor,
};

pub fn makeBars(self: BarcodeGen, cl_alloc: *cl.Alloc, rand_params: RandomizationParams, dims: anytype, rand_source: *math.RandSource) !Bars {
    const out_dims = try math.TensorDims.init(cl_alloc.heap(), dims);
    if (out_dims.len() != 3) {
        return error.InvalidDims;
    }

    const num_barcodes = out_dims.get(2);

    const sample_buf = try self.makeSampleBuf(cl_alloc, rand_source, num_barcodes);
    const blur_kernels = try self.makeBlurKernels(cl_alloc, 5, num_barcodes, rand_source, rand_params);
    const instanced = try self.instanceRandParams(cl_alloc, sample_buf, rand_source, rand_params, out_dims);

    const pass1_out = try self.runFirstPass(cl_alloc, instanced.params_buf, out_dims);
    const pass2_out = try self.math_executor.maskedConv(cl_alloc, pass1_out.imgs, pass1_out.masks, blur_kernels);

    return .{
        .imgs = pass2_out,
        .masks = pass1_out.masks,
        .orientations = instanced.orientations,
    };
}

fn makeSampleBuf(self: BarcodeGen, cl_alloc: *cl.Alloc, rand_source: *math.RandSource, num_barcodes: u32) !math.Executor.Tensor {
    const modules_per_pattern: u32 = @intCast(calcPatternWidth(12));
    const num_modules = modules_per_pattern * num_barcodes;
    const sample_buf = try self.math_executor.createTensorUninitialized(cl_alloc, &.{num_modules});
    try self.math_executor.executor.executeKernelUntracked(self.module_gen_kernel, num_modules, &.{
        .{ .buf = sample_buf.buf },
        .{ .uint = num_barcodes },
        .{ .uint = rand_source.seed },
        .{ .ulong = rand_source.ctr },
    });

    rand_source.ctr += num_modules;

    return sample_buf;
}

const RandParams = struct {
    params_buf: math.Executor.Tensor,
    orientations: math.Executor.Tensor,
};

fn instanceRandParams(
    self: BarcodeGen,
    cl_alloc: *cl.Alloc,
    sample_buf: math.Executor.Tensor,
    rand_source: *math.RandSource,
    rand_params: RandomizationParams,
    dims: math.TensorDims,
) !RandParams {
    const num_barcodes = dims.get(2);
    const params_struct_size = 60;
    const total_params_buf_size = params_struct_size * num_barcodes;
    const params_buf = try self.math_executor.createTensorUninitialized(cl_alloc, &.{total_params_buf_size});

    // x,y per barcode
    const orientations = try self.math_executor.createTensorUninitialized(cl_alloc, &.{ 2, num_barcodes });

    // This fn call may look like a disaster, but it seems better than trying
    // to coordinate struct layout between zig on host and C on GPU
    try self.math_executor.executor.executeKernelUntracked(self.sample_params_kernel, num_barcodes, &.{
        // ret,
        .{ .buf = params_buf.buf },
        // sample_buf_space,
        .{ .buf = sample_buf.buf },
        // orientations_out
        .{ .buf = orientations.buf },
        // n,
        .{ .uint = num_barcodes },
        // img_width,
        .{ .uint = dims.get(0) },
        // img_height,
        .{ .uint = dims.get(1) },
        // min_x_offs,
        .{ .float = rand_params.x_offs_range[0] },
        // max_x_offs,
        .{ .float = rand_params.x_offs_range[1] },
        // min_y_offs,
        .{ .float = rand_params.y_offs_range[0] },
        // max_y_offs,
        .{ .float = rand_params.y_offs_range[1] },
        // min_x_scale,
        .{ .float = rand_params.x_scale_range[0] },
        // max_x_scale,
        .{ .float = rand_params.x_scale_range[1] },
        // min_aspect,
        .{ .float = rand_params.aspect_range[0] },
        // max_aspect,
        .{ .float = rand_params.aspect_range[1] },
        // min_contrast,
        .{ .float = rand_params.min_contrast },
        // min_perlin_grid_size,
        .{ .uint = rand_params.perlin_grid_size_range[0] },
        // max_perlin_grid_size,
        .{ .uint = rand_params.perlin_grid_size_range[1] },
        // min_x_noise_multiplier,
        .{ .float = rand_params.x_noise_multiplier_range[0] },
        // max_x_noise_multiplier,
        .{ .float = rand_params.x_noise_multiplier_range[1] },
        // min_y_noise_multiplier,
        .{ .float = rand_params.y_noise_multiplier_range[0] },
        // max_y_noise_multiplier,
        .{ .float = rand_params.y_noise_multiplier_range[1] },
        // min_background_color,
        .{ .float = rand_params.background_color_range[0] },
        // max_background_color,
        .{ .float = rand_params.background_color_range[1] },
        // seed,
        .{ .uint = rand_source.seed },
        // ctr_start
        .{ .ulong = rand_source.ctr },
    });

    rand_source.ctr += num_barcodes;
    return .{
        .params_buf = params_buf,
        .orientations = orientations,
    };
}

const Pass1Outputs = struct {
    masks: math.Executor.Tensor,
    imgs: math.Executor.Tensor,
};

fn runFirstPass(self: BarcodeGen, cl_alloc: *cl.Alloc, params_buf: math.Executor.Tensor, dims: math.TensorDims) !Pass1Outputs {
    const masks = try self.math_executor.createTensorFilled(cl_alloc, dims, 0.0);
    const imgs = try self.math_executor.createTensorUninitialized(cl_alloc, dims);

    const n = dims.numElems();
    try self.math_executor.executor.executeKernelUntracked(self.barcode_gen_kernel, n, &.{
        .{ .buf = params_buf.buf },
        .{ .buf = imgs.buf },
        .{ .buf = masks.buf },
        .{ .uint = dims.get(0) },
        .{ .uint = dims.get(1) },
        .{ .uint = dims.get(2) },
    });

    return .{
        .masks = masks,
        .imgs = imgs,
    };
}

fn makeBlurKernels(self: BarcodeGen, cl_alloc: *cl.Alloc, kernel_width: u32, num_barcodes: u32, rand_source: *math.RandSource, params: RandomizationParams) !math.Executor.Tensor {
    const kernels = try self.math_executor.createTensorUninitialized(cl_alloc, &.{ kernel_width, kernel_width, num_barcodes });
    const num_elements = kernels.dims.numElems();
    try self.math_executor.executor.executeKernelUntracked(self.generate_blur_kernels_kernel, num_elements, &.{
        .{ .buf = kernels.buf },
        .{ .uint = kernel_width },
        .{ .uint = num_elements },
        .{ .float = params.blur_stddev_range[0] },
        .{ .float = params.blur_stddev_range[1] },
        .{ .uint = rand_source.seed },
        .{ .ulong = rand_source.ctr },
    });

    rand_source.ctr += num_barcodes;

    return kernels;
}

const barcode_constants = struct {
    const start_end_width = 3;
    const middle_width = 5;
    const extra_width = start_end_width * 2 + middle_width;
    const quiet_zone_space = 10;
};

fn calcPatternWidth(len: usize) usize {
    return len * 7 + barcode_constants.extra_width + barcode_constants.quiet_zone_space;
}
