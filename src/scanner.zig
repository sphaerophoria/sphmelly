const std = @import("std");
const sphtud = @import("sphtud");
const BarcodeGen = @import("BarcodeGen.zig");
const stbi = @cImport({
    @cInclude("stb_image.h");
});
const sphalloc = sphtud.alloc;
const sphrender = sphtud.render;
const gl = sphrender.gl;
const sphwindow = sphtud.window;
const gui = sphtud.ui;
const cl = @import("cl.zig");
const math = @import("math.zig");
const tsv = @import("training_sample_view.zig");

const ImagePixelPos = tsv.ImagePixelPos;
const GlImage = tsv.GlImage;
const CpuImage = tsv.CpuImage;
const Config = @import("Config.zig");

const GuiAction = union(enum) {
    update_displayed_img: u32,

    fn genUpdateDisplayedImg(val: u32) GuiAction {
        return .{ .update_displayed_img = val };
    }
};

const Args = struct {
    background_dir: []const u8,
    stage1_config: []const u8,
    stage2_config: []const u8,

    const Switch = enum {
        @"--background-dir",
        @"--stage1-config",
        @"--stage2-config",
    };

    fn parse(alloc: std.mem.Allocator) !Args {
        var it = try std.process.argsWithAllocator(alloc);

        const process_name = it.next() orelse "scanner";

        var background_dir: ?[]const u8 = null;
        var stage1_config: ?[]const u8 = null;
        var stage2_config: ?[]const u8 = null;

        while (it.next()) |arg| {
            const s = std.meta.stringToEnum(Switch, arg) orelse {
                std.log.err("{s} is not a valid argument", .{arg});
                help(process_name);
            };

            switch (s) {
                .@"--background-dir" => background_dir = it.next() orelse {
                    std.log.err("Missing background dir arg", .{});
                    help(process_name);
                },
                .@"--stage1-config" => stage1_config = it.next() orelse {
                    std.log.err("Missing stage1 config arg", .{});
                    help(process_name);
                },
                .@"--stage2-config" => stage2_config = it.next() orelse {
                    std.log.err("Missing stage2 config arg", .{});
                    help(process_name);
                },
            }
        }

        return .{
            .background_dir = background_dir orelse {
                std.log.err("background dir not provided", .{});
                help(process_name);
            },
            .stage1_config = stage1_config orelse {
                std.log.err("stage1_config not provided", .{});
                help(process_name);
            },
            .stage2_config = stage2_config orelse {
                std.log.err("stage2_config not provided", .{});
                help(process_name);
            },
        };
    }

    fn help(process_name: []const u8) noreturn {
        const stdout = std.io.getStdOut();

        stdout.writer().print(
            \\USAGE: {s} [ARGS]
            \\
            \\Required args:
            \\--background-dir: Where to load image backgrounds from
            \\--stage1-config: Data configuration
            \\--stage2-config: Data configuration
            \\
        , .{process_name}) catch {};

        std.process.exit(1);
    }
};

fn asf32(in: anytype) f32 {
    return @floatFromInt(in);
}

fn tensorToImgView(scratch: std.mem.Allocator, cl_alloc: *cl.Alloc, slice: math.Executor.TensorSlice, math_executor: math.Executor, img_view: *tsv.ImageView(GuiAction)) !void {
    const img = try math_executor.sliceToCpuDeferred(scratch, cl_alloc, slice);
    try img.event.wait();

    const cpu_tensor = math.Tensor([]f32){
        .buf = img.val,
        .dims = slice.dims,
    };
    const rgba = try tsv.greyTensorToRgbaCpu(scratch, cpu_tensor);

    //const f = try std.fs.cwd().createFile("test.ppm", .{});
    //defer f.close();

    //try f.writer().print(
    //    \\P6
    //    \\1024 1024
    //    \\255
    //    \\
    //, .{});
    //for (0..rgba.calcHeight()) |y| {
    //    for (0..rgba.width) |x| {
    //        for (0..3) |c| {
    //            try f.writer().writeByte(rgba.data[c + x * 4 + y * rgba.width * 4]);
    //        }
    //    }
    //}

    //std.process.exit(1);
    try img_view.setImg(rgba);
}

pub fn main() !void {
    var allocators: sphrender.AppAllocators(100) = undefined;
    try allocators.initPinned(50 * 1024 * 1024);

    var window: sphwindow.Window = undefined;
    try window.initPinned("sphui demo", 800, 600);

    var cl_alloc: cl.Alloc = undefined;
    try cl_alloc.initPinned(try allocators.root.arena().alloc(u8, 1 * 1024 * 1024));
    defer cl_alloc.deinit();

    var cl_executor = try cl.Executor.init(cl_alloc.heap(), .non_profiling);
    defer cl_executor.deinit();

    const args = try Args.parse(allocators.root.arena());

    const high_res_resolution = 1024;

    var config = blk: {
        const f = try std.fs.cwd().openFile(args.stage1_config, .{});
        var json_reader = std.json.reader(allocators.root.arena(), f.reader());
        break :blk try std.json.parseFromTokenSourceLeaky(Config, allocators.root.arena(), &json_reader, .{ .ignore_unknown_fields = true });
    };

    const math_executor = try math.Executor.init(&cl_alloc, &cl_executor);

    const rand_params = &config.data.rand_params;
    rand_params.x_offs_range[0] = rand_params.x_offs_range[0] * high_res_resolution / asf32(config.data.img_size);
    rand_params.x_offs_range[1] = rand_params.x_offs_range[1] * high_res_resolution / asf32(config.data.img_size);
    rand_params.y_offs_range[0] = rand_params.y_offs_range[0] * high_res_resolution / asf32(config.data.img_size);
    rand_params.y_offs_range[1] = rand_params.y_offs_range[1] * high_res_resolution / asf32(config.data.img_size);

    var barcode_gen = try BarcodeGen.init(allocators.scratch.linear(), &cl_alloc, math_executor, args.background_dir, high_res_resolution);

    gl.glEnable(gl.GL_SCISSOR_TEST);
    gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA);
    gl.glEnable(gl.GL_BLEND);

    const gui_alloc = try allocators.root_render.makeSubAlloc("gui");

    const gui_state = try gui.widget_factory.widgetState(
        GuiAction,
        gui_alloc,
        &allocators.scratch,
        &allocators.scratch_gl,
    );

    var displayed_img_id: u32 = 0;
    const widget_factory = gui_state.factory(gui_alloc);
    const layout = try widget_factory.makeLayout();
    try layout.pushWidget(try widget_factory.makeLabel("Image id"));
    try layout.pushWidget(try widget_factory.makeDrag(u32, &displayed_img_id, &GuiAction.genUpdateDisplayedImg, 1, 10));

    var rand_source = math.RandSource{
        .seed = 0,
        .ctr = 0,
    };

    var image_view = tsv.ImageView(GuiAction){
        .alloc = (try gui_alloc.makeSubAlloc("image_view")).gl,
        .image = undefined,
        .image_renderer = &gui_state.image_renderer,
        .onReqStat = null,
    };
    var resampled_image_view = tsv.ImageView(GuiAction){
        .alloc = (try gui_alloc.makeSubAlloc("image_view")).gl,
        .image = undefined,
        .image_renderer = &gui_state.image_renderer,
        .onReqStat = null,
    };
    try layout.pushWidget(try widget_factory.makeBox(image_view.asWidget(), .{ .width = 0, .height = 512 }, .fill_width));
    try layout.pushWidget(try widget_factory.makeBox(resampled_image_view.asWidget(), .{ .width = 0, .height = 512 }, .fill_width));
    var runner = try widget_factory.makeRunner(try widget_factory.makeScrollView(layout.asWidget()));

    const bars = try barcode_gen.makeBars(&cl_alloc, config.data.rand_params, config.data.enable_backgrounds, config.data.batch_size, &rand_source);

    const resampled = try math_executor.downsample(
        &cl_alloc,
        try math_executor.reshape(&cl_alloc, bars.imgs, &.{ bars.imgs.dims.get(0), bars.imgs.dims.get(1), 1, bars.imgs.dims.get(2) }),
        config.data.img_size,
    );

    try tensorToImgView(
        allocators.scratch.allocator(),
        &cl_alloc,
        try bars.imgs.indexOuter(displayed_img_id),
        math_executor,
        &image_view,
    );

    try tensorToImgView(
        allocators.scratch.allocator(),
        &cl_alloc,
        try resampled.indexOuter(displayed_img_id),
        math_executor,
        &resampled_image_view,
    );

    while (!window.closed()) {
        allocators.resetScratch();
        const width, const height = window.getWindowSize();

        gl.glViewport(0, 0, @intCast(width), @intCast(height));
        gl.glScissor(0, 0, @intCast(width), @intCast(height));

        gl.glClear(gl.GL_COLOR_BUFFER_BIT);

        const response = try runner.step(1.0, .{
            .width = @intCast(width),
            .height = @intCast(height),
        }, &window.queue);

        if (response.action) |a| switch (a) {
            .update_displayed_img => |val| {
                displayed_img_id = std.math.clamp(val, 0, bars.imgs.dims.get(bars.imgs.dims.len() - 1) - 1);

                {
                    const bar_slice = try bars.imgs.indexOuter(displayed_img_id);
                    const img = try math_executor.sliceToCpuDeferred(allocators.scratch.allocator(), &cl_alloc, bar_slice);
                    try img.event.wait();

                    const cpu_tensor = math.Tensor([]f32){
                        .buf = img.val,
                        .dims = bar_slice.dims,
                    };
                    const rgba = try tsv.greyTensorToRgbaCpu(allocators.scratch.allocator(), cpu_tensor);
                    try image_view.setImg(rgba);
                }
            },
        };
        window.swapBuffers();
    }
}
