const std = @import("std");
const sphtud = @import("sphtud");
const sphalloc = sphtud.alloc;
const sphrender = sphtud.render;
const gl = sphrender.gl;
const sphwindow = sphtud.window;
const gui = sphtud.ui;
const cl = @import("cl.zig");
const math = @import("math.zig");
const BarcodeGen = @import("BarcodeGen.zig");
const tsv = @import("training_sample_view.zig");

const GuiAction = union(enum) {
    update_box_param: struct {
        param: *f32,
        val: f32,
    },
};

const Box = struct {
    x: f32,
    y: f32,
    w: f32,
    h: f32,
    r: f32,

    fn toCl(self: Box, cl_alloc: *cl.Alloc, executor: math.Executor) !math.Executor.Tensor {
        const data: [5]f32 = .{ self.x, self.y, self.w, self.h, self.r };
        const res = try executor.createTensor(cl_alloc, &data, &.{ 5, 1 });
        try res.event.wait();

        return res.val;
    }

    const default = Box{
        .x = 0,
        .y = 0,
        .w = 0.2,
        .h = 0.2,
        .r = 0,
    };
};

fn appendBoxWidgets(widget_factory: gui.widget_factory.WidgetFactory(GuiAction), layout: *gui.layout.Layout(GuiAction), box: *Box) !void {
    const ActionGen = struct {
        param: *f32,

        pub fn generate(self: @This(), val: f32) GuiAction {
            return .{
                .update_box_param = .{
                    .param = self.param,
                    .val = val,
                },
            };
        }
    };

    const grid = try widget_factory.makeGrid(
        &.{
            .{
                .horizontal_justify = .right,
                .vertical_justify = .bottom,
                .width = .{ .ratio = 1.0 },
            },
            .{
                .horizontal_justify = .center,
                .vertical_justify = .center,
                .width = .{ .fixed = 150 },
            },
        },
        10,
        10,
    );

    try layout.pushWidget(grid.asWidget());

    inline for (std.meta.fields(Box)) |field| {
        const param = &@field(box, field.name);
        try grid.pushWidget(
            try widget_factory.makeLabel(field.name),
        );
        try grid.pushWidget(
            try widget_factory.makeDragFloat(param, ActionGen{ .param = param }, 0.005),
        );
    }
}

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

pub fn main() !void {
    var allocators: sphrender.AppAllocators(100) = undefined;
    try allocators.initPinned(10 * 1024 * 1024);

    const sidebar_width = 200;

    var cl_alloc: cl.Alloc = undefined;
    try cl_alloc.initPinned(try allocators.root.arena().alloc(u8, 4 * 1024));
    defer cl_alloc.deinit();

    var cl_executor = try cl.Executor.init(cl_alloc.heap(), .non_profiling);
    defer cl_executor.deinit();

    const math_executor = try math.Executor.init(&cl_alloc, &cl_executor);

    var rand_source = math.RandSource{
        .ctr = 0,
        .seed = 0,
    };
    const barcode_gen = try BarcodeGen.init(allocators.scratch.linear(), &cl_alloc, math_executor, "backgrounds3", 1024);
    const bars = try barcode_gen.makeBars(.{
        .cl_alloc = &cl_alloc,
        .extract_params = null,
        .rand_params = .{
            .x_offs_range = .{ 0, 0 },
            .y_offs_range = .{ 0, 0 },
            .x_scale_range = .{ 0.8, 0.8 },
            .rot_range = .{ std.math.pi / 10.0, std.math.pi / 10.0 },
            .aspect_range = .{ 1.0, 1.0 },
            .min_contrast = 0.7,
            .perlin_grid_size_range = .{ 10, 10 },
            .x_noise_multiplier_range = .{ 0, 0 },
            .y_noise_multiplier_range = .{ 0, 0 },
            .background_color_range = .{ 0, 0 },
            .blur_stddev_range = .{ 0.001, 0.001 },
            .no_code_prob = 0,
        },
        .enable_backgrounds = true,
        .num_images = 1,
        .output_size = 1024,
        .label_in_frame = false,
        .confidence_metric = .none,
        .rand_source = &rand_source,
    });

    var window: sphwindow.Window = undefined;
    try window.initPinned("sphui demo", 600 + sidebar_width, 600);

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

    const widget_factory = gui_state.factory(gui_alloc);

    const sidebar_layout = try widget_factory.makeLayout();

    var box_1 = Box{
        .x = 0.025,
        .y = 0,
        .w = 0.93,
        .h = 0.415,
        .r = 0,
    };

    try sidebar_layout.pushWidget(
        try widget_factory.makeLabel("Box 1"),
    );
    try appendBoxWidgets(widget_factory, sidebar_layout, &box_1);

    const toplevel_layout = try widget_factory.makeLayout();
    toplevel_layout.cursor.direction = .left_to_right;
    toplevel_layout.item_pad = 0;

    const image_view = try makeImageView(widget_factory.alloc, widget_factory.state);
    const downsampled_image_view = try makeImageView(widget_factory.alloc, widget_factory.state);

    {
        const img_cpu_buf = try math_executor.toCpu(allocators.scratch.allocator(), &cl_alloc, bars.imgs);
        const img_cpu = try tsv.greyTensorToRgbaCpu(allocators.scratch.allocator(), .{
            .buf = img_cpu_buf,
            .dims = try .init(allocators.scratch.allocator(), &.{ 1024, 1024 }),
        });
        try image_view.setImg(img_cpu);
    }

    try toplevel_layout.pushWidget(
        try widget_factory.makeBox(
            sidebar_layout.asWidget(),
            .{ .width = sidebar_width, .height = 0 },
            .fill_height,
        ),
    );

    const images_layout = try widget_factory.makeLayout();
    try toplevel_layout.pushWidget(try widget_factory.makeScrollView(images_layout.asWidget()));

    try images_layout.pushWidget(
        try widget_factory.makeBox(image_view.asWidget(), .{ .height = 400, .width = 0 }, .fill_width),
    );

    try images_layout.pushWidget(
        try widget_factory.makeBox(downsampled_image_view.asWidget(), .{ .height = 400, .width = 0 }, .fill_width),
    );

    var runner = try widget_factory.makeRunner(
        toplevel_layout.asWidget(),
    );

    const default_box_buf = try sphrender.xyt_program.Buffer.init(gui_alloc.gl, &.{
        .{ .vPos = .{ -1, -1 } },
        .{ .vPos = .{ -1, 1 } },
        .{ .vPos = .{ 1, 1 } },
        .{ .vPos = .{ 1, -1 } },
    });
    var default_box_source = try sphrender.xyt_program.RenderSource.init(gui_alloc.gl);
    default_box_source.bindData(gui_state.solid_color_renderer.handle(), default_box_buf);

    const cl_checkpoint = cl_alloc.checkpoint();

    while (!window.closed()) {
        allocators.resetScratch();
        cl_alloc.reset(cl_checkpoint);
        const width, const height = window.getWindowSize();

        gl.glViewport(0, 0, @intCast(width), @intCast(height));
        gl.glScissor(0, 0, @intCast(width), @intCast(height));

        const background_color = gui.widget_factory.StyleColors.background_color;
        gl.glClearColor(background_color.r, background_color.g, background_color.b, background_color.a);
        gl.glClear(gl.GL_COLOR_BUFFER_BIT);

        const response = try runner.step(1.0, .{
            .width = @intCast(width),
            .height = @intCast(height),
        }, &window.queue);

        if (response.action) |a| switch (a) {
            .update_box_param => |param| {
                param.param.* = param.val;

                {
                    const img_cpu_buf = try math_executor.toCpu(allocators.scratch.allocator(), &cl_alloc, bars.imgs);
                    const img_cpu = try tsv.greyTensorToRgbaCpu(allocators.scratch.allocator(), .{
                        .buf = img_cpu_buf,
                        .dims = try .init(allocators.scratch.allocator(), &.{ 1024, 1024 }),
                    });
                    try image_view.setImg(img_cpu);
                }

                const render_ctx = try tsv.ImageRenderContext.init(image_view.image);
                defer render_ctx.reset();

                gl.glLineWidth(10.0);

                const b1_txfm = sphtud.math.Transform.scale(
                    box_1.w,
                    box_1.h,
                ).then(.rotate(box_1.r))
                    .then(.translate(box_1.x, box_1.y));

                gui_state.solid_color_renderer.renderLineLoop(default_box_source, .{
                    .color = .{ 1, 0, 0 },
                    .transform = b1_txfm.inner,
                });

                const cl_box = try box_1.toCl(&cl_alloc, math_executor);
                const bars_reshaped = try math_executor.reshape(&cl_alloc, bars.imgs, &.{ 1024, 1024, 1, 1 });
                const downsampled = try math_executor.downsampleBox(&cl_alloc, bars_reshaped, cl_box, 400, 4);

                {
                    const img_cpu_buf = try math_executor.toCpu(allocators.scratch.allocator(), &cl_alloc, downsampled);
                    const slice = try downsampled.indexOuter(0);
                    const img_cpu = try tsv.greyTensorToRgbaCpu(allocators.scratch.allocator(), .{
                        .buf = img_cpu_buf,
                        .dims = slice.dims,
                    });
                    try downsampled_image_view.setImg(img_cpu);
                }
            },
        };

        window.swapBuffers();
    }
}
