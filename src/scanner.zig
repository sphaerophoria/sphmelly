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
const nn = @import("nn.zig");
const nn_checkpoint = @import("nn/checkpoint.zig");

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
    stage1_checkpoint: []const u8,
    stage2_config: []const u8,
    stage2_checkpoint: []const u8,

    const Switch = enum {
        @"--background-dir",
        @"--stage1-config",
        @"--stage1-checkpoint",
        @"--stage2-config",
        @"--stage2-checkpoint",
    };

    fn parse(alloc: std.mem.Allocator) !Args {
        var it = try std.process.argsWithAllocator(alloc);

        const process_name = it.next() orelse "scanner";

        var background_dir: ?[]const u8 = null;
        var stage1_config: ?[]const u8 = null;
        var stage1_checkpoint: ?[]const u8 = null;
        var stage2_config: ?[]const u8 = null;
        var stage2_checkpoint: ?[]const u8 = null;

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
                .@"--stage1-checkpoint" => stage1_checkpoint = it.next() orelse {
                    std.log.err("Missing stage1 checkpoint arg", .{});
                    help(process_name);
                },
                .@"--stage2-config" => stage2_config = it.next() orelse {
                    std.log.err("Missing stage2 config arg", .{});
                    help(process_name);
                },
                .@"--stage2-checkpoint" => stage2_checkpoint = it.next() orelse {
                    std.log.err("Missing stage2 checkpoint arg", .{});
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
            .stage1_checkpoint = stage1_checkpoint orelse {
                std.log.err("stage1_checkpoint not provided", .{});
                help(process_name);
            },
            .stage2_config = stage2_config orelse {
                std.log.err("stage2_config not provided", .{});
                help(process_name);
            },
            .stage2_checkpoint = stage2_checkpoint orelse {
                std.log.err("stage2_checkpoint not provided", .{});
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
            \\--stage1-config
            \\--stage1-checkpoint
            \\--stage2-config
            \\--stage2-checkpoint
            \\
        , .{process_name}) catch {};

        std.process.exit(1);
    }
};

fn asf32(in: anytype) f32 {
    return @floatFromInt(in);
}

const ImageUpdater = struct {
    scratch: std.mem.Allocator,
    scratch_gl: *sphrender.GlAlloc,
    cl_alloc: *cl.Alloc,
    math_executor: math.Executor,
    widgets: *Widgets,
    bars: BarcodeGen.Bars,
    resampled: math.Executor.Tensor,
    extracted: math.Executor.Tensor,
    extracted_flipped: math.Executor.Tensor,
    boxes: []const f32,

    fn update(self: ImageUpdater, displayed_img_id: u32) !void {
        const box = self.boxes[6 * displayed_img_id ..][0..6];

        try self.tensorToImgView(
            try self.bars.imgs.indexOuter(displayed_img_id),
            box,
            self.widgets.input_image,
        );
        try self.tensorToImgView(
            try self.resampled.indexOuter(displayed_img_id),
            box,
            self.widgets.low_res_image,
        );
        try self.tensorToImgView(
            try self.extracted.indexOuter(displayed_img_id),
            null,
            self.widgets.extracted_image,
        );
        try self.tensorToImgView(
            try self.extracted_flipped.indexOuter(displayed_img_id),
            null,
            self.widgets.extracted_flipped_image,
        );
    }

    fn tensorToImgView(self: ImageUpdater, slice: math.Executor.TensorSlice, box: ?[]const f32, img_view: *tsv.ImageView(GuiAction)) !void {
        const img = try self.math_executor.sliceToCpuDeferred(self.scratch, self.cl_alloc, slice);
        try img.event.wait();

        const cpu_tensor = math.Tensor([]f32){
            .buf = img.val,
            .dims = slice.dims,
        };
        const rgba = try tsv.greyTensorToRgbaCpu(self.scratch, cpu_tensor);

        try img_view.setImg(rgba);

        const render_ctx = try tsv.ImageRenderContext.init(img_view.image);
        defer render_ctx.reset();

        const gl_cp = self.scratch_gl.checkpoint();
        defer self.scratch_gl.restore(gl_cp);

        if (box) |b| {
            const bbox_source = try tsv.makeBBoxGLBuffer(self.scratch_gl, b, self.widgets.solid_color_renderer);
            self.widgets.solid_color_renderer.renderLineStrip(bbox_source, .{
                .color = .{ 1, 0, 0 },
                .transform = sphtud.math.Transform.identity.inner,
            });
        }
    }
};

fn makeImageView(gui_alloc: gui.GuiAlloc, gui_state: *gui.widget_factory.WidgetState(GuiAction)) !*tsv.ImageView(GuiAction) {
    const ret = try gui_alloc.heap.arena().create(tsv.ImageView(GuiAction));
    ret.* = tsv.ImageView(GuiAction){
        .alloc = try gui_alloc.gl.makeSubAlloc(gui_alloc.heap),
        .image = .empty,
        .image_renderer = &gui_state.image_renderer,
        .onReqStat = null,
    };
    return ret;
}

const Widgets = struct {
    input_image: *tsv.ImageView(GuiAction),
    low_res_image: *tsv.ImageView(GuiAction),
    extracted_image: *tsv.ImageView(GuiAction),
    // FIXME: Flipped shouldn't be needed if our model is better
    extracted_flipped_image: *tsv.ImageView(GuiAction),
    solid_color_renderer: *sphrender.xyt_program.SolidColorProgram,
    root: gui.runner.Runner(GuiAction),
};

fn buildUi(allocators: anytype, gui_alloc: gui.GuiAlloc, displayed_img_id: *u32) !Widgets {
    const gui_state = try gui.widget_factory.widgetState(
        GuiAction,
        gui_alloc,
        &allocators.scratch,
        &allocators.scratch_gl,
    );

    const widget_factory = gui_state.factory(gui_alloc);
    const layout = try widget_factory.makeLayout();
    try layout.pushWidget(try widget_factory.makeLabel("Image id"));
    try layout.pushWidget(try widget_factory.makeDrag(u32, displayed_img_id, &GuiAction.genUpdateDisplayedImg, 1, 10));

    const image_layout = try widget_factory.makeLayout();
    try layout.pushWidget(try widget_factory.makeScrollView(image_layout.asWidget()));

    const input_image = try makeImageView(gui_alloc, gui_state);
    const low_res_image = try makeImageView(gui_alloc, gui_state);
    const extracted_image = try makeImageView(gui_alloc, gui_state);
    const extracted_flipped_image = try makeImageView(gui_alloc, gui_state);

    const image_widgets: []const *tsv.ImageView(GuiAction) = &.{ input_image, low_res_image, extracted_image, extracted_flipped_image };
    for (image_widgets) |view| {
        try image_layout.pushWidget(
            try widget_factory.makeBox(
                view.asWidget(),
                .{ .width = 0, .height = 512 },
                .fill_width,
            ),
        );
    }

    return .{
        .input_image = input_image,
        .low_res_image = low_res_image,
        .extracted_image = extracted_image,
        // FIXME: Flipped shouldn't be needed if our model is better
        .extracted_flipped_image = extracted_flipped_image,
        .solid_color_renderer = &gui_state.solid_color_renderer,
        .root = try widget_factory.makeRunner(layout.asWidget()),
    };
}

pub fn main() !void {
    var allocators: sphrender.AppAllocators(100) = undefined;
    try allocators.initPinned(50 * 1024 * 1024);

    var window: sphwindow.Window = undefined;
    try window.initPinned("sphui demo", 800, 600);

    var cl_alloc: cl.Alloc = undefined;
    try cl_alloc.initPinned(try allocators.root.arena().alloc(u8, 500 * 1024 * 1024));
    defer cl_alloc.deinit();

    var cl_executor = try cl.Executor.init(cl_alloc.heap(), .non_profiling);
    defer cl_executor.deinit();

    const args = try Args.parse(allocators.root.arena());

    const high_res_resolution = 1024;

    var stage1_config = blk: {
        const f = try std.fs.cwd().openFile(args.stage1_config, .{});
        var json_reader = std.json.reader(allocators.root.arena(), f.reader());
        break :blk try std.json.parseFromTokenSourceLeaky(Config, allocators.root.arena(), &json_reader, .{ .ignore_unknown_fields = true });
    };

    const stage2_config = blk: {
        const f = try std.fs.cwd().openFile(args.stage2_config, .{});
        var json_reader = std.json.reader(allocators.root.arena(), f.reader());
        break :blk try std.json.parseFromTokenSourceLeaky(Config, allocators.root.arena(), &json_reader, .{ .ignore_unknown_fields = true });
    };

    var math_executor = try math.Executor.init(&cl_alloc, &cl_executor);

    const rand_params = &stage1_config.data.rand_params;
    rand_params.x_offs_range[0] = rand_params.x_offs_range[0] * high_res_resolution / asf32(stage1_config.data.img_size);
    rand_params.x_offs_range[1] = rand_params.x_offs_range[1] * high_res_resolution / asf32(stage1_config.data.img_size);
    rand_params.y_offs_range[0] = rand_params.y_offs_range[0] * high_res_resolution / asf32(stage1_config.data.img_size);
    rand_params.y_offs_range[1] = rand_params.y_offs_range[1] * high_res_resolution / asf32(stage1_config.data.img_size);

    var barcode_gen = try BarcodeGen.init(allocators.scratch.linear(), &cl_alloc, math_executor, args.background_dir, high_res_resolution);

    gl.glEnable(gl.GL_SCISSOR_TEST);
    gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA);
    gl.glEnable(gl.GL_BLEND);

    var displayed_img_id: u32 = 0;
    const gui_alloc = try allocators.root_render.makeSubAlloc("gui");
    var widgets = try buildUi(&allocators, gui_alloc, &displayed_img_id);

    var rand_source = math.RandSource{
        .seed = 0,
        .ctr = 0,
    };

    var bars = try barcode_gen.makeBars(&cl_alloc, stage1_config.data.rand_params, stage1_config.data.enable_backgrounds, stage1_config.data.batch_size, &rand_source);

    const reshaped_bars = try math_executor.reshape(&cl_alloc, bars.imgs, &.{ bars.imgs.dims.get(0), bars.imgs.dims.get(1), 1, bars.imgs.dims.get(2) });

    const resampled = try math_executor.downsample(
        &cl_alloc,
        reshaped_bars,
        stage1_config.data.img_size,
    );

    const initializers = nn.makeInitializers(&math_executor, &rand_source);

    const stage1_checkpoint = try nn_checkpoint.loadCheckpoint(&cl_alloc, &math_executor, args.stage1_checkpoint);
    const stage1_layers = try nn.modelFromConfig(&cl_alloc, &math_executor, &initializers, stage1_config.network.layers, stage1_checkpoint);

    const stage2_checkpoint = try nn_checkpoint.loadCheckpoint(&cl_alloc, &math_executor, args.stage2_checkpoint);
    const stage2_layers = try nn.modelFromConfig(&cl_alloc, &math_executor, &initializers, stage2_config.network.layers, stage2_checkpoint);

    const box_predictions = try nn.runLayersUntraced(&cl_alloc, resampled, stage1_layers, &math_executor);
    const flipped_box_predictions = try barcode_gen.flipBoxes(&cl_alloc, box_predictions);
    const boxes_cpu = try math_executor.toCpu(cl_alloc.heap(), &cl_alloc, box_predictions);

    const boxes = try barcode_gen.boxPredictionToBox(&cl_alloc, box_predictions);
    const flipped_boxes = try barcode_gen.boxPredictionToBox(&cl_alloc, flipped_box_predictions);

    const extracted = try math_executor.downsampleBox(
        &cl_alloc,
        reshaped_bars,
        boxes,
        stage2_config.data.img_size,
    );

    const extracted_flipped = try math_executor.downsampleBox(
        &cl_alloc,
        reshaped_bars,
        flipped_boxes,
        stage2_config.data.img_size,
    );

    const predicted_codes = try nn.runLayersUntraced(&cl_alloc, extracted, stage2_layers, &math_executor);
    const predicted_codes_cpu = try math_executor.toCpu(cl_alloc.heap(), &cl_alloc, predicted_codes);

    const predicted_codes_flipped = try nn.runLayersUntraced(&cl_alloc, extracted_flipped, stage2_layers, &math_executor);
    const predicted_codes_flipped_cpu = try math_executor.toCpu(cl_alloc.heap(), &cl_alloc, predicted_codes_flipped);

    for (predicted_codes_cpu[0..95]) |p| {
        const p_true: u32 = if (p > 0) 1 else 0;
        std.debug.print("{d} ", .{p_true});
    }
    std.debug.print("\n", .{});

    for (predicted_codes_flipped_cpu[0..95]) |p| {
        const p_true: u32 = if (p > 0) 1 else 0;
        std.debug.print("{d} ", .{p_true});
    }
    std.debug.print("\n", .{});

    const updater = ImageUpdater{
        .scratch = allocators.scratch.allocator(),
        .scratch_gl = &allocators.scratch_gl,
        .cl_alloc = &cl_alloc,
        .math_executor = math_executor,
        .widgets = &widgets,
        .bars = bars,
        .resampled = resampled,
        .extracted = extracted,
        .extracted_flipped = extracted_flipped,
        .boxes = boxes_cpu,
    };

    try updater.update(displayed_img_id);

    while (!window.closed()) {
        allocators.resetScratch();
        const width, const height = window.getWindowSize();

        gl.glViewport(0, 0, @intCast(width), @intCast(height));
        gl.glScissor(0, 0, @intCast(width), @intCast(height));

        gl.glClear(gl.GL_COLOR_BUFFER_BIT);

        const response = try widgets.root.step(1.0, .{
            .width = @intCast(width),
            .height = @intCast(height),
        }, &window.queue);

        if (response.action) |a| switch (a) {
            .update_displayed_img => |val| {
                const last_dim = bars.imgs.dims.len() - 1;
                const num_images = bars.imgs.dims.get(last_dim);
                const last_idx = num_images - 1;
                displayed_img_id = std.math.clamp(val, 0, last_idx);

                try updater.update(displayed_img_id);
            },
        };
        window.swapBuffers();
    }
}
