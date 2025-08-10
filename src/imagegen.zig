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
};

const ImageDimsRetriever = struct {
    image: GlImage,
    buf: [12]u8,

    pub fn getText(self: *ImageDimsRetriever) []const u8 {
        return std.fmt.bufPrint(&self.buf, "{d}x{d}", .{ self.image.width, self.image.height }) catch return &.{};
    }
};

const ImageUpdater = struct {
    barcode_gen: *BarcodeGen,
    cl_alloc: *cl.Alloc,
    scratch: std.mem.Allocator,
    math_executor: math.Executor,
    image_view: *tsv.ImageView(GuiAction),
    orientation_view: *tsv.OrientationRenderer(GuiAction),

    seed: u64,
    selected_image: usize = 0,
    param_gen: BarcodeGen.RandomizationParams,

    const num_images = 9;

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
            self.cl_alloc,
            self.param_gen,
            &.{ 64, 64, num_images },
            &rand_source,
        );

        try self.math_executor.executor.finish();

        const cpu_image = try self.tensorToRgbaCpu(bars.imgs);
        const cpu_orientation = try self.extractOrientation(bars.orientations);

        gl.glLineWidth(5.0);
        try self.image_view.setImg(cpu_image);
        self.orientation_view.setOrientation(cpu_orientation);
    }
};

pub fn main() !void {
    var allocators: sphrender.AppAllocators(100) = undefined;
    try allocators.initPinned(50 * 1024 * 1024);

    var window: sphwindow.Window = undefined;
    try window.initPinned("sphui demo", 800, 600);

    const cl_executor = try cl.Executor.init();
    defer cl_executor.deinit();

    var cl_alloc: cl.Alloc = undefined;
    try cl_alloc.initPinned(try allocators.root.arena().alloc(u8, 1 * 1024 * 1024));
    defer cl_alloc.deinit();

    const math_executor = try math.Executor.init(&cl_alloc, cl_executor);

    var barcode_gen = try BarcodeGen.init(&cl_alloc, math_executor);

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

    var orientation_view = try tsv.orientationRenderer(GuiAction, gui_alloc, &solid_color_renderer, .{ .r = 1, .g = 0, .b = 0, .a = 1 });

    var image_view_updater = ImageUpdater{
        .barcode_gen = &barcode_gen,
        .cl_alloc = &cl_alloc,
        .scratch = allocators.scratch.allocator(),
        .math_executor = math_executor,
        .image_view = &image_view,
        .orientation_view = orientation_view,
        .param_gen = .{
            // FIXME offset range should probably be a ratio of image size, not absolute pixels
            .x_offs_range = .{ -50, 50 },
            .y_offs_range = .{ -50, 50 },
            .x_scale_range = .{ 2.0, 3.0 },
            .aspect_range = .{ 1.0, 2.0 },
            .min_contrast = 0.2,
            .x_noise_multiplier_range = .{ 5.0, 10.1 },
            .y_noise_multiplier_range = .{ 5.0, 10.1 },
            .perlin_grid_size_range = .{ 10, 100 },
            .background_color_range = .{ 0.0, 1.0 },
            // FIXME Blur amount is in pixel space, maybe these should be
            // scaled by resolution of inputs
            .blur_stddev_range = .{ 0.0001, 3.0 },
        },
        .seed = 0,
    };

    try image_view_updater.update();

    const widget_factory = gui_state.factory(gui_alloc);
    const layout = try widget_factory.makeLayout();
    try layout.pushWidget(try widget_factory.makeLabel(ImageDimsRetriever{
        .image = image_view.image,
        .buf = undefined,
    }));

    try layout.pushWidget(try widget_factory.makeLabel("seed"));
    try layout.pushWidget(try widget_factory.makeDrag(u64, &image_view_updater.seed, &GuiAction.generateSeed, 1, 5));
    try layout.pushWidget(try widget_factory.makeLabel("selected"));
    try layout.pushWidget(try widget_factory.makeDrag(usize, &image_view_updater.selected_image, &GuiAction.generateSelectedImage, 1, 5));

    const image_view_stack = try widget_factory.makeStack(2);
    try image_view_stack.pushWidget(image_view.asWidget(), .{});
    try image_view_stack.pushWidget(orientation_view.asWidget(), .{ .vertical_justify = .center, .horizontal_justify = .center });
    try layout.pushWidget(image_view_stack.asWidget());

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
                image_view_updater.selected_image = std.math.clamp(idx, 0, ImageUpdater.num_images - 1);
                try image_view_updater.update();
            },
            .seed => |val| {
                image_view_updater.seed = val;
                try image_view_updater.update();
            },
        };
        window.swapBuffers();
    }
}
