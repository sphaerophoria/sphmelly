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

const GuiAction = union(enum) {
    request_image_stat: ImagePixelPos,
    selected_image: usize,
    toggle_mask,
    seed: usize,

    fn generateSeed(val: usize) GuiAction {
        return .{ .seed = val };
    }

    fn generateSelectedImage(val: usize) GuiAction {
        return .{ .selected_image = val };
    }

    fn generateRequestImageStat(pos: ImagePixelPos) GuiAction {
        return .{
            .request_image_stat = pos,
        };
    }

    fn generateToggleMask(val: bool) GuiAction {
        return .{ .toggle_mask = val };
    }
};

const ImageDimsRetriever = struct {
    image: *const GlImage,
    buf: [12]u8,

    pub fn getText(self: *ImageDimsRetriever) []const u8 {
        return std.fmt.bufPrint(&self.buf, "{d}x{d}", .{ self.image.width, self.image.height }) catch return &.{};
    }
};

const ImageUpdater = struct {
    barcode_gen: *BarcodeGen,
    cl_alloc: *cl.Alloc,
    scratch_gl: *sphrender.GlAlloc,
    scratch: std.mem.Allocator,
    math_executor: math.Executor,
    image_view: *tsv.ImageView(GuiAction),
    solid_color_renderer: *sphrender.xyt_program.SolidColorProgram,
    visualize_mask: bool,
    config: Config,
    in_frame: *bool,

    seed: u64,
    selected_image: usize = 0,

    fn tensorToRgbaCpu(self: ImageUpdater, img_tensor: math.Executor.Tensor) !CpuImage {
        const single_img_size = img_tensor.dims.get(0) * img_tensor.dims.get(1);
        const img_cpu_data = try self.scratch.alloc(f32, single_img_size);

        const offset = self.selected_image * single_img_size * @sizeOf(f32);
        const event = try self.math_executor.executor.readBuffer(
            self.cl_alloc,
            img_tensor.buf,
            offset,
            std.mem.sliceAsBytes(img_cpu_data),
        );
        try event.wait();

        return tsv.greyTensorToRgbaCpu(self.scratch, .{
            .buf = img_cpu_data,
            .dims = .{
                .inner = try self.scratch.dupe(u32, &.{ img_tensor.dims.get(0), img_tensor.dims.get(1) }),
            },
        });
    }

    fn extractOrientation(self: ImageUpdater, orientation_tensor: math.Executor.Tensor) ![2]f32 {
        var ret: [2]f32 = undefined;
        const offset = 2 * @sizeOf(f32) * self.selected_image;
        const event = try self.math_executor.executor.readBuffer(
            self.cl_alloc,
            orientation_tensor.buf,
            offset,
            std.mem.sliceAsBytes(&ret),
        );
        try event.wait();

        return ret;
    }

    fn update(self: ImageUpdater) !void {
        const cp = self.cl_alloc.checkpoint();
        defer self.cl_alloc.reset(cp);

        var rand_source = math.RandSource{
            .ctr = 0,
            .seed = @intCast(self.seed % (1 << 32)),
        };

        const bars = try self.barcode_gen.makeBars(
            .{
                .cl_alloc = self.cl_alloc,
                .rand_params = self.config.data.rand_params,
                .enable_backgrounds = true,
                .num_images = self.config.data.batch_size,
                .label_in_frame = self.config.data.label_in_frame,
                .label_iou = self.config.data.label_iou,
                .rand_source = &rand_source,
            },
        );

        try self.math_executor.executor.finish();

        const source_img = if (self.visualize_mask) bars.masks else bars.imgs;
        const cpu_image = try self.tensorToRgbaCpu(source_img);

        gl.glLineWidth(2.0);

        try self.image_view.setImg(cpu_image);

        const gl_cp = self.scratch_gl.checkpoint();
        defer self.scratch_gl.restore(gl_cp);

        const render_ctx = try tsv.ImageRenderContext.init(self.image_view.image);
        defer render_ctx.reset();

        {
            // Take bars for image we are visualizing
            const box_slice = try bars.box_labels.indexOuter(self.selected_image);
            const res = try self.math_executor.sliceToCpuDeferred(self.scratch, self.cl_alloc, box_slice);
            try res.event.wait();

            const render_source = try tsv.makeBBoxGLBuffer(self.scratch_gl, res.val[0..6], self.solid_color_renderer);
            self.solid_color_renderer.renderLineStrip(render_source, .{
                .color = .{ 0.0, 0.0, 1.0 },
                .transform = sphtud.math.Transform.identity.inner,
            });

            if (self.config.data.label_in_frame) {
                self.in_frame.* = res.val[6] > 0.5;
            }
        }
    }
};

const Args = struct {
    background_dir: []const u8,
    config: []const u8,

    const Switch = enum {
        @"--background-dir",
        @"--config",
    };

    fn parse(alloc: std.mem.Allocator) !Args {
        var it = try std.process.argsWithAllocator(alloc);

        const process_name = it.next() orelse "imagegen";

        var background_dir: ?[]const u8 = null;
        var config: ?[]const u8 = null;

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
                .@"--config" => config = it.next() orelse {
                    std.log.err("Missing config arg", .{});
                    help(process_name);
                },
            }
        }

        return .{
            .background_dir = background_dir orelse {
                std.log.err("background dir not provided", .{});
                help(process_name);
            },
            .config = config orelse {
                std.log.err("config not provided", .{});
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
            \\--config: Data configuration
            \\
        , .{process_name}) catch {};

        std.process.exit(1);
    }
};

const Config = struct {
    data: struct {
        batch_size: u32,
        label_in_frame: bool,
        label_iou: bool,
        img_size: u32,
        rand_params: BarcodeGen.RandomizationParams,
    },
};

const InFrameLabelRetriever = struct {
    in_frame: *bool,
    buf: [20]u8 = undefined,

    pub fn getText(self: *InFrameLabelRetriever) []const u8 {
        return std.fmt.bufPrint(&self.buf, "in frame: {}", .{self.in_frame.*}) catch unreachable;
    }
};

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

    const config = blk: {
        const f = try std.fs.cwd().openFile(args.config, .{});
        var json_reader = std.json.reader(allocators.root.arena(), f.reader());
        break :blk try std.json.parseFromTokenSourceLeaky(Config, allocators.root.arena(), &json_reader, .{ .ignore_unknown_fields = true });
    };

    const math_executor = try math.Executor.init(&cl_alloc, &cl_executor);

    var barcode_gen = try BarcodeGen.init(allocators.scratch.linear(), &cl_alloc, math_executor, args.background_dir, config.data.img_size);

    gl.glEnable(gl.GL_SCISSOR_TEST);
    gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA);
    gl.glEnable(gl.GL_BLEND);

    const gui_alloc = try allocators.root_render.makeSubAlloc("gui");
    const image_view_alloc = try gui_alloc.gl.makeSubAlloc(gui_alloc.heap);

    const gui_state = try gui.widget_factory.widgetState(
        GuiAction,
        gui_alloc,
        &allocators.scratch,
        &allocators.scratch_gl,
    );

    var solid_color_renderer = try sphrender.xyt_program.solidColorProgram(gui_alloc.gl);

    var image_view = tsv.ImageView(GuiAction){
        .alloc = image_view_alloc,
        .image = undefined,
        .image_renderer = &gui_state.image_renderer,
        .onReqStat = &GuiAction.generateRequestImageStat,
    };

    var in_frame: bool = true;
    var image_view_updater = ImageUpdater{
        .barcode_gen = &barcode_gen,
        .cl_alloc = &cl_alloc,
        .scratch = allocators.scratch.allocator(),
        .scratch_gl = &allocators.scratch_gl,
        .math_executor = math_executor,
        .image_view = &image_view,
        .solid_color_renderer = &solid_color_renderer,
        .config = config,
        .seed = 0,
        .visualize_mask = false,
        .in_frame = &in_frame,
    };

    try image_view_updater.update();

    const widget_factory = gui_state.factory(gui_alloc);
    const layout = try widget_factory.makeLayout();
    try layout.pushWidget(try widget_factory.makeLabel(ImageDimsRetriever{
        .image = &image_view.image,
        .buf = undefined,
    }));

    try layout.pushWidget(try widget_factory.makeLabel("seed"));
    try layout.pushWidget(try widget_factory.makeDrag(u64, &image_view_updater.seed, &GuiAction.generateSeed, 1, 5));
    try layout.pushWidget(try widget_factory.makeLabel("selected"));
    try layout.pushWidget(try widget_factory.makeDrag(usize, &image_view_updater.selected_image, &GuiAction.generateSelectedImage, 1, 5));
    try layout.pushWidget(try widget_factory.makeLabel("Visualize mask"));
    try layout.pushWidget(try widget_factory.makeCheckbox(&image_view_updater.visualize_mask, GuiAction.toggle_mask));
    try layout.pushWidget(try widget_factory.makeLabel(InFrameLabelRetriever{ .in_frame = &in_frame }));

    try layout.pushWidget(image_view.asWidget());

    var runner = try widget_factory.makeRunner(layout.asWidget());

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
            .request_image_stat => |r| {
                std.debug.print("{d}, {d}\n", .{ r.x, r.y });
            },
            .selected_image => |idx| {
                image_view_updater.selected_image = std.math.clamp(idx, 0, config.data.batch_size - 1);
                try image_view_updater.update();
            },
            .seed => |val| {
                image_view_updater.seed = val;
                try image_view_updater.update();
            },
            .toggle_mask => {
                image_view_updater.visualize_mask = !image_view_updater.visualize_mask;
                try image_view_updater.update();
            },
        };
        window.swapBuffers();
    }
}
