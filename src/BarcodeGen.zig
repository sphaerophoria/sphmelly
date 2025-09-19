const sphtud = @import("sphtud");
const std = @import("std");
const cl = @import("cl.zig");
const math = @import("math.zig");
const stbi = @cImport({
    @cInclude("stb_image.h");
});

math_executor: math.Executor,
module_gen_kernel: cl.Executor.Kernel,
barcode_gen_kernel: cl.Executor.Kernel,
sample_params_kernel: cl.Executor.Kernel,
generate_blur_kernels_kernel: cl.Executor.Kernel,
heal_orientations_kernel: cl.Executor.Kernel,
box_prediction_to_box_kernel: cl.Executor.Kernel,
flip_boxes_kernel: cl.Executor.Kernel,
backgrounds: math.Executor.Tensor,
background_size: u32,

const barcode_gen_program_source = math.Executor.downsample_program_content ++ math.Executor.rand_program_content ++ math.Executor.iou_program_content ++ @embedFile("BarcodeGen/generate.cl");

const BarcodeGen = @This();

pub const RandomizationParams = struct {
    x_offs_range: [2]f32,
    y_offs_range: [2]f32,
    x_scale_range: [2]f32,
    rot_range: [2]f32,
    aspect_range: [2]f32,
    min_contrast: f32,
    perlin_grid_size_range: [2]u32,
    x_noise_multiplier_range: [2]f32,
    y_noise_multiplier_range: [2]f32,
    background_color_range: [2]f32,
    blur_stddev_range: [2]f32,
    no_code_prob: f32,
};

pub fn init(scratch: sphtud.alloc.LinearAllocator, cl_alloc: *cl.Alloc, math_executor: math.Executor, background_image_dir: []const u8, img_width: u32) !BarcodeGen {
    const program = try math_executor.executor.createProgram(cl_alloc, barcode_gen_program_source);
    const barcode_gen_kernel = try program.createKernel(cl_alloc, "generate_barcode");
    const module_gen_kernel = try program.createKernel(cl_alloc, "generate_module_patterns");
    const sample_params_kernel = try program.createKernel(cl_alloc, "sample_barcode_params");
    const generate_blur_kernels_kernel = try program.createKernel(cl_alloc, "generate_blur_kernels");
    const heal_orientations_kernel = try program.createKernel(cl_alloc, "heal_orientations");
    const box_prediction_to_box_kernel = try program.createKernel(cl_alloc, "box_prediction_to_box");
    const flip_boxes_kernel = try program.createKernel(cl_alloc, "flip_boxes");

    const backgrounds = try makeBackgroundImgBuf(scratch, cl_alloc, math_executor, background_image_dir, img_width);

    return .{
        .math_executor = math_executor,
        .barcode_gen_kernel = barcode_gen_kernel,
        .module_gen_kernel = module_gen_kernel,
        .sample_params_kernel = sample_params_kernel,
        .generate_blur_kernels_kernel = generate_blur_kernels_kernel,
        .heal_orientations_kernel = heal_orientations_kernel,
        .box_prediction_to_box_kernel = box_prediction_to_box_kernel,
        .flip_boxes_kernel = flip_boxes_kernel,
        .backgrounds = backgrounds,
        .background_size = img_width,
    };
}

pub const Bars = struct {
    imgs: math.Executor.Tensor,
    masks: math.Executor.Tensor,
    box_labels: math.Executor.Tensor,
    bars: math.Executor.Tensor,
};

pub const ConfidenceMetric = enum(u8) {
    none = 0,
    iou = 1,
    rotation_err = 2,
    corner_dist = 3,
};

pub const MakeBarsParams = struct {
    cl_alloc: *cl.Alloc,
    rand_params: RandomizationParams,
    enable_backgrounds: bool,
    num_images: u32,
    label_in_frame: bool,
    confidence_metric: ConfidenceMetric,
    rand_source: *math.RandSource,
};
pub fn makeBars(self: BarcodeGen, params: MakeBarsParams) !Bars {
    const out_dims = try math.TensorDims.init(params.cl_alloc.heap(), &.{ self.background_size, self.background_size, params.num_images });

    const num_barcodes = out_dims.get(2);

    const sample_buf = try self.makeSampleBuf(params.cl_alloc, params.rand_source, num_barcodes);
    const blur_kernels = try self.makeBlurKernels(params.cl_alloc, 5, num_barcodes, params.rand_source, params.rand_params);
    const instanced = try self.instanceRandParams(params.cl_alloc, sample_buf.sample_buf, params.rand_source, params.rand_params, out_dims, params.label_in_frame, params.confidence_metric);

    const pass1_out = try self.runFirstPass(params.cl_alloc, instanced.params_buf, out_dims, params.enable_backgrounds);
    const pass2_out = try self.math_executor.maskedConv(params.cl_alloc, pass1_out.imgs, pass1_out.masks, blur_kernels);

    return .{
        .imgs = pass2_out,
        .masks = pass1_out.masks,
        .box_labels = instanced.box_labels,
        .bars = sample_buf.no_quiet,
    };
}

pub fn healBboxLabels(self: BarcodeGen, cl_alloc: *cl.Alloc, labels: math.Executor.Tensor, predicted: math.Executor.Tensor, confidence_metric: ConfidenceMetric, disable_bbox_loss_if_out_of_frame: bool) !void {
    if (!predicted.dims.eql(labels.dims)) {
        return error.InvalidDims;
    }

    const n = labels.dims.get(labels.dims.len() - 1);
    try self.math_executor.executor.executeKernelUntracked(cl_alloc, self.heal_orientations_kernel, n, &.{
        .{ .buf = labels.buf },
        .{ .buf = predicted.buf },
        .{ .uint = labels.dims.get(0) },
        .{ .uint = @intFromEnum(confidence_metric) },
        .{ .uint = @intFromBool(disable_bbox_loss_if_out_of_frame) },
        .{ .uint = n },
    });
}

const SampleBuf = struct {
    sample_buf: math.Executor.Tensor,
    no_quiet: math.Executor.Tensor,
};

fn makeSampleBuf(self: BarcodeGen, cl_alloc: *cl.Alloc, rand_source: *math.RandSource, num_barcodes: u32) !SampleBuf {
    const modules_per_pattern: u32 = calcPatternWidth(12);
    const num_modules = modules_per_pattern * num_barcodes;
    const sample_buf = try self.math_executor.createTensorUninitialized(cl_alloc, &.{num_modules});
    const no_quiet_buf = try self.math_executor.createTensorUninitialized(cl_alloc, &.{ calcPatternWidthNoQuiet(12), num_barcodes });
    try self.math_executor.executor.executeKernelUntracked(cl_alloc, self.module_gen_kernel, num_modules, &.{
        .{ .buf = sample_buf.buf },
        .{ .buf = no_quiet_buf.buf },
        .{ .uint = num_barcodes },
        .{ .uint = rand_source.seed },
        .{ .ulong = rand_source.ctr },
    });

    rand_source.ctr += num_modules;

    return .{
        .sample_buf = sample_buf,
        .no_quiet = no_quiet_buf,
    };
}

const RandParams = struct {
    params_buf: math.Executor.Tensor,
    box_labels: math.Executor.Tensor,
};

fn instanceRandParams(
    self: BarcodeGen,
    cl_alloc: *cl.Alloc,
    sample_buf: math.Executor.Tensor,
    rand_source: *math.RandSource,
    rand_params: RandomizationParams,
    dims: math.TensorDims,
    label_in_frame: bool,
    confidence_metric: ConfidenceMetric,
) !RandParams {
    const num_barcodes = dims.get(2);
    const params_struct_size = 72;
    const total_params_buf_size = params_struct_size * num_barcodes;
    const params_buf = try self.math_executor.createTensorUninitialized(cl_alloc, &.{total_params_buf_size});

    // x,y,sqrt(w),sqrt(h),rx,ry,fully_in_frame per barcode
    const box_label_size: u32 = @as(u32, 6) + @intFromBool(label_in_frame) + @intFromBool(confidence_metric != .none);
    const box_labels = try self.math_executor.createTensorUninitialized(cl_alloc, &.{ box_label_size, num_barcodes });

    // This fn call may look like a disaster, but it seems better than trying
    // to coordinate struct layout between zig on host and C on GPU
    try self.math_executor.executor.executeKernelUntracked(cl_alloc, self.sample_params_kernel, num_barcodes, &.{
        // ret,
        .{ .buf = params_buf.buf },
        // expected params size
        .{ .uint = params_struct_size },
        // sample_buf_space,
        .{ .buf = sample_buf.buf },
        // bbox_out
        .{ .buf = box_labels.buf },
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
        // min_rot
        .{ .float = rand_params.rot_range[0] },
        // max_rot
        .{ .float = rand_params.rot_range[1] },
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
        // no_code_prob,
        .{ .float = rand_params.no_code_prob },
        // num_images
        .{ .uint = self.backgrounds.dims.get(2) },
        // seed,
        .{ .uint = rand_source.seed },
        // label_in_frame,
        .{ .uint = @intFromBool(label_in_frame) },
        // confidence_metric,
        .{ .uint = @intFromEnum(confidence_metric) },
        // ctr_start
        .{ .ulong = rand_source.ctr },
    });

    rand_source.ctr += num_barcodes;
    return .{
        .params_buf = params_buf,
        .box_labels = box_labels,
    };
}

const Pass1Outputs = struct {
    masks: math.Executor.Tensor,
    imgs: math.Executor.Tensor,
};

fn runFirstPass(self: BarcodeGen, cl_alloc: *cl.Alloc, params_buf: math.Executor.Tensor, dims: math.TensorDims, enable_backgrounds: bool) !Pass1Outputs {
    const masks = try self.math_executor.createTensorFilled(cl_alloc, dims, 0.0);
    const imgs = try self.math_executor.createTensorUninitialized(cl_alloc, dims);

    const n = dims.numElems();
    try self.math_executor.executor.executeKernelUntracked(cl_alloc, self.barcode_gen_kernel, n, &.{
        .{ .buf = params_buf.buf },
        .{ .buf = imgs.buf },
        .{ .buf = masks.buf },
        .{ .buf = self.backgrounds.buf },
        .{ .uint = dims.get(0) },
        .{ .uint = dims.get(1) },
        .{ .uint = @intFromBool(enable_backgrounds) },
        .{ .uint = dims.get(2) },
    });

    return .{
        .masks = masks,
        .imgs = imgs,
    };
}

pub fn boxPredictionToBox(self: BarcodeGen, cl_alloc: *cl.Alloc, boxes: math.Executor.Tensor, dilation: f32) !math.Executor.Tensor {
    if (boxes.dims.len() != 2) return error.InvalidDims;

    const ret = try self.math_executor.createTensorUninitialized(cl_alloc, &.{ 5, boxes.dims.get(1) });
    const n = ret.dims.get(1);
    try self.math_executor.executor.executeKernelUntracked(cl_alloc, self.box_prediction_to_box_kernel, n, &.{
        .{ .buf = boxes.buf },
        .{ .buf = ret.buf },
        .{ .float = dilation },
        .{ .uint = boxes.dims.get(0) },
        .{ .uint = n },
    });

    return ret;
}

pub fn flipBoxes(self: BarcodeGen, cl_alloc: *cl.Alloc, boxes: math.Executor.Tensor) !math.Executor.Tensor {
    if (boxes.dims.len() != 2) return error.InvalidDims;
    if (boxes.dims.get(0) != 6) return error.InvalidDims;

    const ret = try self.math_executor.createTensorUninitialized(cl_alloc, boxes.dims);
    const n = ret.dims.get(1);
    try self.math_executor.executor.executeKernelUntracked(cl_alloc, self.flip_boxes_kernel, n, &.{
        .{ .buf = boxes.buf },
        .{ .buf = ret.buf },
        .{ .uint = boxes.dims.get(0) },
        .{ .uint = n },
    });

    return ret;
}

fn makeBlurKernels(self: BarcodeGen, cl_alloc: *cl.Alloc, kernel_width: u32, num_barcodes: u32, rand_source: *math.RandSource, params: RandomizationParams) !math.Executor.Tensor {
    const kernels = try self.math_executor.createTensorUninitialized(cl_alloc, &.{ kernel_width, kernel_width, num_barcodes });
    const num_elements = kernels.dims.numElems();
    try self.math_executor.executor.executeKernelUntracked(cl_alloc, self.generate_blur_kernels_kernel, num_elements, &.{
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

fn calcPatternWidth(len: usize) u32 {
    return @as(u32, @intCast(len)) * 7 + barcode_constants.extra_width + barcode_constants.quiet_zone_space;
}

fn calcPatternWidthNoQuiet(len: usize) u32 {
    return @as(u32, @intCast(len)) * 7 + barcode_constants.extra_width;
}

fn loadImgScaledCpu(img_path: [:0]const u8, out_width: usize, out: []f32) !void {
    var in_width: i32 = 0;
    var in_height: i32 = 0;
    const data_ptr = stbi.stbi_load(img_path, &in_width, &in_height, null, 1);
    if (data_ptr == null) return error.OpenImage;

    defer stbi.stbi_image_free(data_ptr);

    const in_width_u: usize = @intCast(in_width);
    const in_width_f: f32 = @floatFromInt(in_width);
    const in_height_f: f32 = @floatFromInt(in_height);
    const out_width_f: f32 = @floatFromInt(out_width);
    const out_height_f: f32 = @floatFromInt(out.len / out_width);

    for (out, 0..) |*out_v, i| {
        const out_x: f32 = @floatFromInt(i % out_width);
        const out_y: f32 = @floatFromInt(i / out_width);

        const in_x: usize = @intFromFloat(out_x / out_width_f * in_width_f);
        const in_y: usize = @intFromFloat(out_y / out_height_f * in_height_f);

        const val_u8 = data_ptr[in_y * in_width_u + in_x];
        out_v.* = @as(f32, @floatFromInt(val_u8)) / 255.0;
    }
}

fn makeBackgroundImgBuf(scratch: sphtud.alloc.LinearAllocator, cl_alloc: *cl.Alloc, math_executor: math.Executor, background_image_dir: []const u8, out_img_width: u32) !math.Executor.Tensor {
    var dir = try std.fs.cwd().openDir(background_image_dir, .{ .iterate = true });
    defer dir.close();

    const cp = scratch.checkpoint();
    defer scratch.restore(cp);

    var it = dir.iterate();

    var image_paths = try sphtud.util.RuntimeSegmentedList([:0]const u8).init(
        scratch.allocator(),
        scratch.allocator(),
        100,
        10000,
    );

    while (try it.next()) |entry| {
        const cwd_rel_path = try std.fmt.allocPrintSentinel(scratch.allocator(), "{s}/{s}", .{ background_image_dir, entry.name }, 0);
        try image_paths.append(cwd_rel_path);
    }

    const backgrounds = try math_executor.createTensorFilled(
        cl_alloc,
        &.{ out_img_width, out_img_width, @as(u32, @intCast(image_paths.len)) },
        std.math.nan(f32),
    );

    const out_img_num_elems = out_img_width * out_img_width;
    const out_img_size_bytes = out_img_num_elems * @sizeOf(f32);

    var path_it = image_paths.iter();
    const cpu_f32: []f32 = try scratch.allocator().alloc(f32, out_img_num_elems);

    const cl_alloc_cp = cl_alloc.checkpoint();
    defer cl_alloc.reset(cl_alloc_cp);

    var img_offset: usize = 0;
    while (path_it.next()) |path| {
        defer img_offset += out_img_size_bytes;

        try loadImgScaledCpu(path.*, out_img_width, cpu_f32);

        const res = try math_executor.executor.writeBuffer(cl_alloc, backgrounds.buf, img_offset, std.mem.sliceAsBytes(cpu_f32));
        try res.wait();
    }

    return backgrounds;
}
