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
const bar_comparison_widget = @import("bar_comparison_widget.zig");

const ImagePixelPos = tsv.ImagePixelPos;
const GlImage = tsv.GlImage;
const CpuImage = tsv.CpuImage;
const Config = @import("Config.zig");

const GuiAction = union(enum) {
    inspect_img: usize,
    go_overview,
    inspect_prev,
    inspect_next,
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
        var stdout_buf: [4096]u8 = undefined;
        var stdout = std.fs.File.stdout().writer(&stdout_buf);

        stdout.interface.print(
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
        stdout.interface.flush() catch {};

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
    predicted_bars: math.Executor.Tensor,
    predicted_bars_flipped: math.Executor.Tensor,
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

        const expected_cpu = try self.extractCpuTensor(try self.bars.bars.indexOuter(displayed_img_id));
        try self.widgets.comparison.renderBarComparison(
            try self.extractCpuTensor(try self.predicted_bars.indexOuter(displayed_img_id)),
            expected_cpu,
        );

        try self.widgets.flipped_comparison.renderBarComparison(
            try self.extractCpuTensor(try self.predicted_bars_flipped.indexOuter(displayed_img_id)),
            expected_cpu,
        );
    }

    fn extractCpuTensor(self: ImageUpdater, slice: math.Executor.TensorSlice) !math.Tensor([]f32) {
        const deferred = try self.math_executor.sliceToCpuDeferred(self.scratch, self.cl_alloc, slice);
        try deferred.event.wait();
        return .{
            .buf = deferred.val,
            .dims = slice.dims,
        };
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
    comparison: bar_comparison_widget.ComparisonView(GuiAction),
    flipped_comparison: bar_comparison_widget.ComparisonView(GuiAction),
    root: gui.runner.Runner(GuiAction),
};

const DisplayMode = enum {
    overview,
    inspection,
};

const AppViewRetriever = struct {
    display_mode: *DisplayMode,

    pub fn get(self: AppViewRetriever) usize {
        return @intFromEnum(self.display_mode.*);
    }
};

const InfoImgIdLabel = struct {
    displayed_img_info: *usize,
    infos: []const ImgInfo,

    pub fn getText(self: *InfoImgIdLabel) []const u8 {
        return self.infos[self.displayed_img_info.*].name;
    }
};

const InfoImgCorrectLabel = struct {
    displayed_img_info: *usize,
    infos: []const ImgInfo,

    pub fn getText(self: *InfoImgCorrectLabel) []const u8 {
        return self.infos[self.displayed_img_info.*].correct_label;
    }
};

fn boxImageView(widget_factory: gui.widget_factory.WidgetFactory(GuiAction), image_view: *tsv.ImageView(GuiAction)) !gui.Widget(GuiAction) {
    return try widget_factory.makeBox(
        image_view.asWidget(),
        .{ .width = 0, .height = 512 },
        .fill_width,
    );
}
fn pushImageView(
    image_layout: *gui.layout.Layout(GuiAction),
    widget_factory: gui.widget_factory.WidgetFactory(GuiAction),
    name: []const u8,
    view: *tsv.ImageView(GuiAction),
) !void {
    try image_layout.pushWidget(
        try widget_factory.makeLabel(name),
    );
    try image_layout.pushWidget(
        try boxImageView(widget_factory, view),
    );
}

fn buildUi(
    allocators: anytype,
    gui_alloc: gui.GuiAlloc,
    displayed_img_info: *usize,
    display_mode: *DisplayMode,
    infos: []const ImgInfo,
) !Widgets {
    const gui_state = try gui.widget_factory.widgetState(
        GuiAction,
        gui_alloc,
        &allocators.scratch,
        &allocators.scratch_gl,
    );

    const widget_factory = gui_state.factory(gui_alloc);
    const layout = try widget_factory.makeLayout();
    try layout.pushWidget(try widget_factory.makeButton("Overview", GuiAction{ .go_overview = {} }));

    try layout.pushWidget(try widget_factory.makeButton("Prev", GuiAction{ .inspect_prev = {} }));
    try layout.pushWidget(try widget_factory.makeButton("Next", GuiAction{ .inspect_next = {} }));

    try layout.pushWidget(try widget_factory.makeLabel(InfoImgIdLabel{
        .displayed_img_info = displayed_img_info,
        .infos = infos,
    }));
    try layout.pushWidget(try widget_factory.makeLabel(InfoImgCorrectLabel{
        .displayed_img_info = displayed_img_info,
        .infos = infos,
    }));

    const image_layout = try widget_factory.makeLayout();
    try layout.pushWidget(try widget_factory.makeScrollView(image_layout.asWidget()));

    const input_image = try makeImageView(gui_alloc, gui_state);
    const low_res_image = try makeImageView(gui_alloc, gui_state);
    const extracted_image = try makeImageView(gui_alloc, gui_state);
    const extracted_flipped_image = try makeImageView(gui_alloc, gui_state);

    const comparison = try bar_comparison_widget.makeComparisonView(GuiAction, widget_factory);
    const flipped_comparison = try bar_comparison_widget.makeComparisonView(GuiAction, widget_factory);

    try pushImageView(image_layout, widget_factory, "input", input_image);
    try pushImageView(image_layout, widget_factory, "low_res", low_res_image);

    try image_layout.pushWidget(try widget_factory.makeLabel("extracted"));
    try image_layout.pushWidget(comparison.widget);
    try image_layout.pushWidget(try boxImageView(widget_factory, extracted_image));

    try image_layout.pushWidget(try widget_factory.makeLabel("flipped"));
    try image_layout.pushWidget(flipped_comparison.widget);
    try image_layout.pushWidget(try boxImageView(widget_factory, extracted_flipped_image));

    const overview_layout = try widget_factory.makeGrid(
        &.{ .{
            .width = .{ .ratio = 0.8 },
            .horizontal_justify = .left,
            .vertical_justify = .center,
        }, .{
            .width = .{ .ratio = 0.2 },
            .horizontal_justify = .right,
            .vertical_justify = .center,
        } },
        infos.len * 2,
        infos.len * 2,
    );

    for (infos, 0..) |info, info_idx| {
        try overview_layout.pushWidget(
            try widget_factory.makeInteractable(
                try widget_factory.makeLabel(info.name),
                GuiAction{ .inspect_img = info_idx },
                null,
            ),
        );
        try overview_layout.pushWidget(
            try widget_factory.makeInteractable(
                try widget_factory.makeLabel(info.correct_label),
                GuiAction{ .inspect_img = info_idx },
                null,
            ),
        );
    }

    const one_of = try widget_factory.makeOneOf(
        AppViewRetriever{
            .display_mode = display_mode,
        },
        &.{
            try widget_factory.makeScrollView(try widget_factory.makeFrame(
                overview_layout.asWidget(),
            )),
            layout.asWidget(),
        },
    );

    return .{
        .input_image = input_image,
        .low_res_image = low_res_image,
        .extracted_image = extracted_image,
        // FIXME: Flipped shouldn't be needed if our model is better
        .extracted_flipped_image = extracted_flipped_image,
        .solid_color_renderer = &gui_state.solid_color_renderer,
        .comparison = comparison,
        .flipped_comparison = flipped_comparison,
        .root = try widget_factory.makeRunner(one_of),
    };
}

const ImgInfo = struct {
    name: []const u8,
    correct_label: []const u8,
    correct_bars: usize,
    id: u32,

    fn lessThanCtx(_: void, a: ImgInfo, b: ImgInfo) bool {
        if (a.correct_bars < b.correct_bars) {
            return true;
        } else if (a.correct_bars > b.correct_bars) return false;

        return std.mem.order(u8, a.name, b.name) == .lt;
    }
};

const bars_per_img = 95;

fn countCorrectPredictions(
    predicted: []const f32,
    expected: []const f32,
) usize {
    std.debug.assert(predicted.len == bars_per_img);
    std.debug.assert(expected.len == bars_per_img);
    var correct: usize = 0;
    for (predicted, expected) |p, e| {
        const predicted_true = p > 0;
        const expected_true = e > 0.5;
        if (predicted_true == expected_true) {
            correct += 1;
        }
    }
    return correct;
}

fn preprocessImgInfos(
    out_alloc: std.mem.Allocator,
    cl_alloc: *cl.Alloc,
    math_executor: math.Executor,
    predicted: math.Executor.Tensor,
    flipped_predicted: math.Executor.Tensor,
    expected: math.Executor.Tensor,
    batch_size: usize,
) ![]const ImgInfo {
    var img_infos = std.ArrayList(ImgInfo).initBuffer(
        try out_alloc.alloc(ImgInfo, batch_size),
    );

    const cp = cl_alloc.checkpoint();
    defer cl_alloc.reset(cp);

    const predicted_codes_cpu = try math_executor.toCpu(cl_alloc.heap(), cl_alloc, predicted);
    const predicted_codes_flipped_cpu = try math_executor.toCpu(cl_alloc.heap(), cl_alloc, flipped_predicted);
    const expected_cpu = try math_executor.toCpu(cl_alloc.heap(), cl_alloc, expected);

    for (0..batch_size) |img_id| {
        const output_start = img_id * bars_per_img;
        const output_end = output_start + bars_per_img;

        const correct_bars = countCorrectPredictions(
            predicted_codes_cpu[output_start..output_end],
            expected_cpu[output_start..output_end],
        );

        const correct_flipped = countCorrectPredictions(
            predicted_codes_flipped_cpu[output_start..output_end],
            expected_cpu[output_start..output_end],
        );

        const max_correct = @max(correct_bars, correct_flipped);
        try img_infos.appendBounded(.{
            .name = try std.fmt.allocPrint(out_alloc, "img {d:0>3}", .{img_id}),
            .correct_bars = max_correct,
            .id = @intCast(img_id),
            .correct_label = try std.fmt.allocPrint(
                out_alloc,
                "{d}/95",
                .{max_correct},
            ),
        });
    }

    std.mem.sort(ImgInfo, img_infos.items, {}, ImgInfo.lessThanCtx);

    return img_infos.items;
}

// Tuned to make sense, it would be better if the downsampling code
// automatically determined what good blur/multisampling params made sense, but
// since it does not, and it matters, we pick some and pass them along
const stage1_multisample = 1;
const stage2_multisample = 4;

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
        var reader_buf: [4096]u8 = undefined;
        var f_reader = f.reader(&reader_buf);
        var json_reader = std.json.Reader.init(allocators.root.arena(), &f_reader.interface);
        break :blk try std.json.parseFromTokenSourceLeaky(Config, allocators.root.arena(), &json_reader, .{ .ignore_unknown_fields = true });
    };

    const stage2_config = blk: {
        const f = try std.fs.cwd().openFile(args.stage2_config, .{});
        var reader_buf: [4096]u8 = undefined;
        var f_reader = f.reader(&reader_buf);
        var json_reader = std.json.Reader.init(allocators.root.arena(), &f_reader.interface);
        break :blk try std.json.parseFromTokenSourceLeaky(Config, allocators.root.arena(), &json_reader, .{ .ignore_unknown_fields = true });
    };

    var math_executor = try math.Executor.init(&cl_alloc, &cl_executor);

    const rand_params = &stage1_config.data.rand_params;
    rand_params.x_offs_range[0] = rand_params.x_offs_range[0] * high_res_resolution / asf32(stage1_config.data.render_size);
    rand_params.x_offs_range[1] = rand_params.x_offs_range[1] * high_res_resolution / asf32(stage1_config.data.render_size);
    rand_params.y_offs_range[0] = rand_params.y_offs_range[0] * high_res_resolution / asf32(stage1_config.data.render_size);
    rand_params.y_offs_range[1] = rand_params.y_offs_range[1] * high_res_resolution / asf32(stage1_config.data.render_size);

    var barcode_gen = try BarcodeGen.init(allocators.scratch.linear(), &cl_alloc, math_executor, args.background_dir, high_res_resolution);

    gl.glEnable(gl.GL_SCISSOR_TEST);
    gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA);
    gl.glEnable(gl.GL_BLEND);

    var displayed_img_info: usize = 0;
    var display_mode = DisplayMode.overview;

    var rand_source = math.RandSource{
        .seed = 0,
        .ctr = 0,
    };

    var bars = try barcode_gen.makeBars(
        .{ .cl_alloc = &cl_alloc, .rand_params = stage1_config.data.rand_params, .extract_params = stage1_config.data.extract_params, .enable_backgrounds = stage1_config.data.enable_backgrounds, .num_images = stage1_config.data.batch_size, .label_in_frame = stage1_config.data.label_in_frame, .confidence_metric = stage1_config.data.confidence_metric, .rand_source = &rand_source, .output_size = high_res_resolution },
    );

    const reshaped_bars = try math_executor.reshape(&cl_alloc, bars.imgs, &.{ bars.imgs.dims.get(0), bars.imgs.dims.get(1), 1, bars.imgs.dims.get(2) });

    const resampled = try math_executor.downsample(
        &cl_alloc,
        reshaped_bars,
        stage1_config.data.output_size,
        stage1_multisample,
    );

    const initializers = nn.makeInitializers(&math_executor, &rand_source);

    const stage1_checkpoint = try nn_checkpoint.loadCheckpoint(&cl_alloc, &math_executor, args.stage1_checkpoint);
    const stage1_layers = try nn.modelFromConfig(&cl_alloc, &math_executor, &initializers, stage1_config.network.layers, stage1_checkpoint);

    const stage2_checkpoint = try nn_checkpoint.loadCheckpoint(&cl_alloc, &math_executor, args.stage2_checkpoint);
    const stage2_layers = try nn.modelFromConfig(&cl_alloc, &math_executor, &initializers, stage2_config.network.layers, stage2_checkpoint);

    const box_predictions = try nn.runLayersUntraced(&cl_alloc, resampled, stage1_layers, &math_executor);
    const flipped_box_predictions = try barcode_gen.flipBoxes(&cl_alloc, box_predictions);
    const boxes_cpu = try math_executor.toCpu(cl_alloc.heap(), &cl_alloc, box_predictions);

    const boxes = try barcode_gen.boxPredictionToBox(&cl_alloc, box_predictions, 1.1);
    const flipped_boxes = try barcode_gen.boxPredictionToBox(&cl_alloc, flipped_box_predictions, 1.1);

    const extracted = try math_executor.downsampleBox(
        &cl_alloc,
        reshaped_bars,
        boxes,
        stage2_config.data.output_size,
        stage2_multisample,
    );

    const extracted_flipped = try math_executor.downsampleBox(
        &cl_alloc,
        reshaped_bars,
        flipped_boxes,
        stage2_config.data.output_size,
        stage2_multisample,
    );

    const predicted_codes = try nn.runLayersUntraced(&cl_alloc, extracted, stage2_layers, &math_executor);
    const predicted_codes_flipped = try nn.runLayersUntraced(&cl_alloc, extracted_flipped, stage2_layers, &math_executor);

    const img_infos = try preprocessImgInfos(allocators.root.arena(), &cl_alloc, math_executor, predicted_codes, predicted_codes_flipped, bars.bars, stage1_config.data.batch_size);

    const gui_alloc = try allocators.root_render.makeSubAlloc("gui");
    var widgets = try buildUi(&allocators, gui_alloc, &displayed_img_info, &display_mode, img_infos);

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
        .predicted_bars = predicted_codes,
        .predicted_bars_flipped = predicted_codes_flipped,
        .boxes = boxes_cpu,
    };

    try updater.update(img_infos[displayed_img_info].id);

    while (!window.closed()) {
        allocators.resetScratch();
        const width, const height = window.getWindowSize();

        gl.glViewport(0, 0, @intCast(width), @intCast(height));
        gl.glScissor(0, 0, @intCast(width), @intCast(height));

        const background_color = gui.widget_factory.StyleColors.background_color;
        gl.glClearColor(background_color.r, background_color.g, background_color.b, background_color.a);
        gl.glClear(gl.GL_COLOR_BUFFER_BIT);

        const response = try widgets.root.step(1.0, .{
            .width = @intCast(width),
            .height = @intCast(height),
        }, &window.queue);

        if (response.action) |a| switch (a) {
            .inspect_img => |val| {
                display_mode = .inspection;
                displayed_img_info = val;
                try updater.update(img_infos[displayed_img_info].id);
            },
            .inspect_prev => {
                displayed_img_info -|= 1;
                try updater.update(img_infos[displayed_img_info].id);
            },
            .inspect_next => {
                displayed_img_info = @min(displayed_img_info + 1, stage1_config.data.batch_size - 1);
                try updater.update(img_infos[displayed_img_info].id);
            },
            .go_overview => {
                display_mode = .overview;
            },
        };
        window.swapBuffers();
    }
}
