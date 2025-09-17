const std = @import("std");
const cl = @import("cl.zig");
const math = @import("math.zig");
const nn = @import("nn.zig");
const nn_checkpoint = @import("nn/checkpoint.zig");
const Config = @import("Config.zig");
const BarcodeGen = @import("BarcodeGen.zig");
const training_stats = @import("training_stats.zig");

pub fn logBboxHeaders(logger: anytype, val_stats: training_stats.BboxValidationData) !void {
    inline for (std.meta.fields(@TypeOf(val_stats))) |field| {
        inline for (std.meta.fields(training_stats.SegmentedStats)) |stat_field| {
            try logger.print("{s} {s},", .{ field.name, stat_field.name });
        }
    }
    try logger.print("\n", .{});
}

pub fn logBboxVal(logger: anytype, val_stats: training_stats.BboxValidationData) !void {
    inline for (std.meta.fields(@TypeOf(val_stats))) |field| {
        const segmented_stats: ?training_stats.SegmentedStats = @field(val_stats, field.name);
        if (segmented_stats) |s| {
            inline for (std.meta.fields(training_stats.SegmentedStats)) |stat_field| {
                const val: f32 = @field(s, stat_field.name);
                try logger.print("{d},", .{val});
            }
        } else {
            inline for (std.meta.fields(training_stats.SegmentedStats)) |_| {
                try logger.print("0,", .{});
            }
        }
    }
    try logger.print("\n", .{});
}

pub fn main() !void {
    const cl_alloc_buf = try std.heap.page_allocator.alloc(u8, 100 * 1024 * 1024);
    defer std.heap.page_allocator.free(cl_alloc_buf);

    var cl_alloc: cl.Alloc = undefined;
    try cl_alloc.initPinned(cl_alloc_buf);
    defer cl_alloc.deinit();

    const args = try std.process.argsAlloc(cl_alloc.heap());

    const config_path = args[1];
    const checkpoint_path = args[2];
    const background_dir = args[3];
    const out_path = args[4];

    const profiling_mode = cl.Executor.ProfilingMode.non_profiling;

    var cl_executor = try cl.Executor.init(cl_alloc.heap(), profiling_mode);
    defer cl_executor.deinit();

    var rand_source = math.RandSource{
        .seed = 0,
        .ctr = 0,
    };

    var math_executor = try math.Executor.init(&cl_alloc, &cl_executor);

    const initializers = nn.makeInitializers(&math_executor, &rand_source);

    const config = try Config.parse(cl_alloc.heap(), config_path);
    const checkpoint = try nn_checkpoint.loadCheckpoint(&cl_alloc, &math_executor, checkpoint_path);
    const layers = try nn.modelFromConfig(&cl_alloc, &math_executor, &initializers, config.network.layers, checkpoint);

    // FIXME: backLinear hack
    var barcode_gen = try BarcodeGen.init(
        cl_alloc.buf_alloc.backLinear(),
        &cl_alloc,
        math_executor,
        background_dir,
        config.data.img_size,
    );

    const bars = try barcode_gen.makeBars(.{
        .cl_alloc = &cl_alloc,
        .enable_backgrounds = config.data.enable_backgrounds,
        .num_images = config.val_size,
        .rand_params = config.data.val_rand_params,
        .label_in_frame = config.data.label_in_frame,
        .rand_source = &rand_source,
        .confidence_metric = config.data.confidence_metric,
    });

    const results = try nn.runLayersUntraced(
        &cl_alloc,
        try math_executor.reshape(&cl_alloc, bars.imgs, &.{ config.data.img_size, config.data.img_size, 1, config.val_size }),
        layers,
        &math_executor,
    );

    try barcode_gen.healBboxLabels(&cl_alloc, bars.box_labels, results, config.data.confidence_metric, config.disable_bbox_loss_if_out_of_frame);

    const data = try training_stats.calcBboxValidationData(&cl_alloc, .{
        .in_frame = config.data.label_in_frame,
        .confidence = config.data.confidence_metric != .none,
    }, &barcode_gen, math_executor, results, bars.box_labels);

    const out_f = try std.fs.cwd().createFile(out_path, .{});

    const writer = out_f.writer();

    try writer.writeAll("name,");
    try logBboxHeaders(writer, data);
    try writer.print("{s},", .{config_path});
    try logBboxVal(writer, data);
}
