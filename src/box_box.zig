const std = @import("std");
const sphtud = @import("sphtud");
const sphalloc = sphtud.alloc;
const sphrender = sphtud.render;
const gl = sphrender.gl;
const sphwindow = sphtud.window;
const gui = sphtud.ui;

const Line = struct {
    a: sphtud.math.Vec2,
    b: sphtud.math.Vec2,
};

fn lineLineIntersection(l1: Line, l2: Line) sphtud.math.Vec2 {
    const a = l1.a;
    const b = l1.b;
    const c = l2.a;
    const d = l2.b;

    const r = b - a;
    const s = d - c;

    const denom = sphtud.math.cross2(r, s);
    if (denom == 0) return @splat(std.math.inf(f32));

    const t = sphtud.math.cross2(c - a, s) / denom;
    return a + r * @as(sphtud.math.Vec2, @splat(t));
}

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

    const default = Box{
        .x = 0,
        .y = 0,
        .w = 0.2,
        .h = 0.2,
        .r = 0,
    };

    fn left(self: Box) Line {
        return .{
            .a = self.bl(),
            .b = self.tl(),
        };
    }
    fn right(self: Box) Line {
        return .{
            .a = self.tr(),
            .b = self.br(),
        };
    }
    fn top(self: Box) Line {
        return .{
            .a = self.tl(),
            .b = self.tr(),
        };
    }

    fn bottom(self: Box) Line {
        return .{
            .a = self.br(),
            .b = self.bl(),
        };
    }

    fn tl(self: Box) sphtud.math.Vec2 {
        const center = sphtud.math.Vec2{ self.x, self.y };
        const x_axis = sphtud.math.Vec2{ @cos(self.r), @sin(self.r) };
        const y_axis = sphtud.math.Vec2{ -x_axis[1], x_axis[0] };

        return center - (x_axis * @as(sphtud.math.Vec2, @splat(self.w / 2))) + (y_axis * @as(sphtud.math.Vec2, @splat(self.h / 2)));
    }

    fn tr(self: Box) sphtud.math.Vec2 {
        const center = sphtud.math.Vec2{ self.x, self.y };
        const x_axis = sphtud.math.Vec2{ @cos(self.r), @sin(self.r) };
        const y_axis = sphtud.math.Vec2{ -x_axis[1], x_axis[0] };

        return center + (x_axis * @as(sphtud.math.Vec2, @splat(self.w / 2))) + (y_axis * @as(sphtud.math.Vec2, @splat(self.h / 2)));
    }
    fn br(self: Box) sphtud.math.Vec2 {
        const center = sphtud.math.Vec2{ self.x, self.y };
        const x_axis = sphtud.math.Vec2{ @cos(self.r), @sin(self.r) };
        const y_axis = sphtud.math.Vec2{ -x_axis[1], x_axis[0] };

        return center + (x_axis * @as(sphtud.math.Vec2, @splat(self.w / 2))) - (y_axis * @as(sphtud.math.Vec2, @splat(self.h / 2)));
    }

    fn bl(self: Box) sphtud.math.Vec2 {
        const center = sphtud.math.Vec2{ self.x, self.y };
        const x_axis = sphtud.math.Vec2{ @cos(self.r), @sin(self.r) };
        const y_axis = sphtud.math.Vec2{ -x_axis[1], x_axis[0] };

        return center - (x_axis * @as(sphtud.math.Vec2, @splat(self.w / 2))) - (y_axis * @as(sphtud.math.Vec2, @splat(self.h / 2)));
    }
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

fn shouldClipVertex(line: Line, point: sphtud.math.Vec2) bool {
    const ab = line.b - line.a;
    const rot_90_axis = sphtud.math.normalize(sphtud.math.Vec2{ ab[1], -ab[0] });

    const ap = point - line.a;
    return sphtud.math.dot(ap, rot_90_axis) < 0;
}

fn pointInBounds(point: sphtud.math.Vec2, line: Line) bool {
    const ap = point - line.a;
    const ab = line.b - line.a;
    const ab_len = sphtud.math.length(ab);
    const ab_norm = ab / @as(sphtud.math.Vec2, @splat(ab_len));
    const d = sphtud.math.dot(ap, ab_norm);
    return d <= ab_len and d >= 0;
}

fn calcIntersection(alloc: std.mem.Allocator, b1: Box, b2: Box) ![]sphtud.math.Vec2 {
    var ret: [2]sphtud.util.RuntimeBoundedArray(sphtud.math.Vec2) = .{
        try .init(alloc, 24),
        try .init(alloc, 24),
    };

    try ret[0].append(b2.tl());
    try ret[0].append(b2.tr());
    try ret[0].append(b2.br());
    try ret[0].append(b2.bl());

    var out_idx: u1 = 0;

    const edges: []const Line = &.{ b1.left(), b1.top(), b1.right(), b1.bottom() };

    for (edges) |clip_edge| {
        out_idx +%= 1;
        const in_idx = out_idx +% 1;

        ret[out_idx].clear();
        for (ret[in_idx].items, 0..) |p, i| {
            if (!shouldClipVertex(clip_edge, p)) {
                try ret[out_idx].append(p);
            }

            const intersection_line = Line{
                .a = p,
                .b = ret[in_idx].items[(i + 1) % ret[in_idx].items.len],
            };
            const intersection_point = lineLineIntersection(clip_edge, intersection_line);
            if (pointInBounds(intersection_point, intersection_line)) {
                try ret[out_idx].append(intersection_point);
            }
        }
    }
    return ret[out_idx].items;
}

pub fn main() !void {
    var allocators: sphrender.AppAllocators(100) = undefined;
    try allocators.initPinned(10 * 1024 * 1024);

    const sidebar_width = 200;

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
        .r = std.math.pi / 4.0,
    };

    try sidebar_layout.pushWidget(
        try widget_factory.makeLabel("Box 1"),
    );
    try appendBoxWidgets(widget_factory, sidebar_layout, &box_1);

    try sidebar_layout.pushWidget(
        try widget_factory.makeLabel("Box 2"),
    );
    try appendBoxWidgets(widget_factory, sidebar_layout, &box_2);

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
    // Cenetered box at 0, 0, w 1, h, 1
    // transform = scale(w, h).then(rot(rot)).then(translate(xc, yc))

    var point_buf = try sphrender.xyt_program.Buffer.init(gui_alloc.gl, &.{
        .{ .vPos = .{ 0, 0 } },
    });
    var point_source = try sphrender.xyt_program.RenderSource.init(gui_alloc.gl);
    point_source.bindData(gui_state.solid_color_renderer.handle(), point_buf);

    while (!window.closed()) {
        allocators.resetScratch();
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

        gl.glPointSize(5.0);
        const intersection_points = try calcIntersection(allocators.scratch.allocator(), box_1, box_2);
        //const intersection_points: []const sphtud.math.Vec2 = &.{lineLineIntersection(box_1.top(), box_2.top())};

        var average: sphtud.math.Vec2 = @splat(0);

        for (intersection_points) |p| {
            average += p;
        }
        average /= @splat(@floatFromInt(intersection_points.len));

        var gl_buf = try sphtud.util.RuntimeBoundedArray(sphtud.render.xyt_program.Vertex).init(allocators.scratch.allocator(), 30);

        try gl_buf.append(.{ .vPos = average });
        for (intersection_points) |p| {
            try gl_buf.append(.{ .vPos = p });
        }
        if (intersection_points.len > 0) {
            try gl_buf.append(.{ .vPos = intersection_points[0] });
        }

        point_buf.updateBuffer(gl_buf.items);
        point_source.bindData(gui_state.solid_color_renderer.handle(), point_buf);
        gui_state.solid_color_renderer.renderFan(point_source, .{
            .color = .{ 0, 1, 0 },
            .transform = sphtud.math.Transform.identity.inner,
        });

        window.swapBuffers();
    }
}
