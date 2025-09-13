const std = @import("std");
const sphtud = @import("sphtud");
const sphalloc = sphtud.alloc;
const sphrender = sphtud.render;
const gl = sphrender.gl;
const sphwindow = sphtud.window;
const gui = sphtud.ui;
const cl = @import("cl.zig");
const math = @import("math.zig");

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

const IouRetriever = struct {
    iou: *f32,
    buf: [20]u8 = undefined,
    pub fn getText(self: *IouRetriever) []const u8 {
        return std.fmt.bufPrint(&self.buf, "iou: {d}", .{self.iou.*}) catch &self.buf;
    }
};

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
        .x = 0,
        .y = 0,
        .w = 0.3,
        .h = 0.6,
        .r = 0,
    };

    var box_2 = Box{
        .x = 0,
        .y = 0,
        .w = 0.3,
        .h = 0.6,
        .r = 0,
    };

    try sidebar_layout.pushWidget(
        try widget_factory.makeLabel("Box 1"),
    );
    try appendBoxWidgets(widget_factory, sidebar_layout, &box_1);

    try sidebar_layout.pushWidget(
        try widget_factory.makeLabel("Box 2"),
    );
    try appendBoxWidgets(widget_factory, sidebar_layout, &box_2);
    var iou: f32 = 0.0;
    const iou_retriever = IouRetriever{ .iou = &iou };
    try sidebar_layout.pushWidget(
        try widget_factory.makeLabel(iou_retriever),
    );

    var runner = try widget_factory.makeRunner(
        try widget_factory.makeBox(
            sidebar_layout.asWidget(),
            .{ .width = sidebar_width, .height = 0 },
            .fill_height,
        ),
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
            },
        };

        gl.glViewport(sidebar_width, 0, @intCast(width - sidebar_width), @intCast(height));
        gl.glScissor(sidebar_width, 0, @intCast(width - sidebar_width), @intCast(height));

        gl.glClearColor(0, 0, 0, 0);
        gl.glClear(gl.GL_COLOR_BUFFER_BIT);

        gl.glLineWidth(3.0);

        const iou_gpu = try math_executor.calcIou(
            &cl_alloc,
            try box_1.toCl(&cl_alloc, math_executor),
            try box_2.toCl(&cl_alloc, math_executor),
        );

        iou = (try math_executor.toCpu(cl_alloc.heap(), &cl_alloc, iou_gpu))[0];

        const b1_txfm = sphtud.math.Transform.scale(
            box_1.w / 2,
            box_1.h / 2,
        ).then(.rotate(box_1.r))
            .then(.translate(box_1.x, box_1.y));

        const b2_txfm = sphtud.math.Transform.scale(
            box_2.w / 2,
            box_2.h / 2,
        ).then(.rotate(box_2.r))
            .then(.translate(box_2.x, box_2.y));

        gui_state.solid_color_renderer.renderLineLoop(default_box_source, .{
            .color = .{ 1, 0, 0 },
            .transform = b1_txfm.inner,
        });

        gui_state.solid_color_renderer.renderLineLoop(default_box_source, .{
            .color = .{ 0, 1, 1 },
            .transform = b2_txfm.inner,
        });

        window.swapBuffers();
    }
}
