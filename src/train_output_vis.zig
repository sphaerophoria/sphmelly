const std = @import("std");
const sphtud = @import("sphtud");
const sphalloc = sphtud.alloc;
const sphrender = sphtud.render;
const gl = sphrender.gl;
const sphwindow = sphtud.window;
const gui = sphtud.ui;

const GuiAction = union(enum) {};

const Elem = struct {
    key: usize,
    val: f32,
};

const Step = struct {
    iter: usize,
    time_ns: u64,
    img: usize,

    elems: sphtud.util.RuntimeSegmentedList(Elem),
};

const Data = struct {
    keys: sphtud.util.RuntimeSegmentedList([]const u8),
    maxes: sphtud.util.RuntimeSegmentedList(f32),
    steps: sphtud.util.RuntimeSegmentedList(Step),
};

const NextStep = struct {
    iter: usize,
    time_ns: u64,
    img: usize,
};

fn parseStep(line: []const u8) !NextStep {
    var line_it = std.mem.splitScalar(u8, line, ',');
    const line_type = line_it.next() orelse return error.NoContent;

    if (!std.mem.eql(u8, line_type, "step")) {
        return error.NotStep;
    }

    const iter_s = line_it.next() orelse return error.InvalidData;
    const time_ns_s = line_it.next() orelse return error.InvalidData;
    const img_s = line_it.next() orelse return error.InvalidData;

    return .{
        .iter = try std.fmt.parseInt(usize, iter_s, 0),
        .time_ns = try std.fmt.parseInt(u64, time_ns_s, 0),
        .img = try std.fmt.parseInt(usize, img_s, 0),
    };
}

fn parseElem(line: []const u8, keys: *sphtud.util.RuntimeSegmentedList([]const u8), keys_alloc: std.mem.Allocator, key_lookup: *std.StringHashMapUnmanaged(usize), key_lookup_alloc: std.mem.Allocator) !Elem {
    var line_it = std.mem.splitScalar(u8, line, ',');
    const key = line_it.next() orelse return error.NoKey;
    const val_s = line_it.next() orelse return error.NoVal;

    const gop = try key_lookup.getOrPut(key_lookup_alloc, key);

    if (!gop.found_existing) {
        try keys.append(try keys_alloc.dupe(u8, key));
        gop.value_ptr.* = keys.len - 1;
    }

    return .{
        .key = gop.value_ptr.*,
        .val = try std.fmt.parseFloat(f32, val_s),
    };
}

pub fn parseData(alloc: std.mem.Allocator, scratch: sphtud.alloc.LinearAllocator, path: []const u8) !Data {
    const cp = scratch.checkpoint();
    defer scratch.restore(cp);

    const f = try std.fs.cwd().openFile(path, .{});
    defer f.close();

    const reader = f.reader();

    var line_buf: [4096]u8 = undefined;

    var step: NextStep = blk: {
        const line = (try reader.readUntilDelimiterOrEof(&line_buf, '\n')) orelse {
            return error.NoData;
        };
        break :blk try parseStep(line);
    };

    var keys = try sphtud.util.RuntimeSegmentedList([]const u8).init(
        alloc,
        alloc,
        16,
        10000,
    );

    var maxes = try sphtud.util.RuntimeSegmentedList(f32).init(
        alloc,
        alloc,
        16,
        10000,
    );

    var steps = try sphtud.util.RuntimeSegmentedList(Step).init(
        alloc,
        alloc,
        100,
        1000000,
    );

    const min_elems = 10;
    const max_elems = 1000;
    var elems = try sphtud.util.RuntimeSegmentedList(Elem).init(
        alloc,
        alloc,
        min_elems,
        max_elems,
    );

    var key_lookup = std.StringHashMapUnmanaged(usize){};

    while (true) {
        const line = (try reader.readUntilDelimiterOrEof(&line_buf, '\n')) orelse {
            break;
        };

        var line_it = std.mem.splitScalar(u8, line, ',');
        const line_type = line_it.next() orelse return error.EmptyLine;

        if (std.mem.eql(u8, line_type, "step")) {
            try steps.append(.{
                .iter = step.iter,
                .time_ns = step.time_ns,
                .img = step.img,
                .elems = elems,
            });
            elems = try sphtud.util.RuntimeSegmentedList(Elem).init(
                alloc,
                alloc,
                min_elems,
                max_elems,
            );
            step = try parseStep(line);
        } else {
            const elem = try parseElem(line, &keys, alloc, &key_lookup, scratch.allocator());
            // FIXME: mins as well
            while (maxes.len <= elem.key) {
                try maxes.append(0);
            }
            const max_ptr = maxes.getPtr(elem.key);
            max_ptr.* = @max(max_ptr.*, elem.val);
            try elems.append(elem);
        }
    }

    try steps.append(.{
        .iter = step.iter,
        .time_ns = step.time_ns,
        .img = step.img,
        .elems = elems,
    });

    return .{
        .keys = keys,
        .steps = steps,
        .maxes = maxes,
    };
}

const GraphWidget = struct {
    size: gui.PixelSize = .{},
    render_source: sphrender.xyt_program.RenderSource,
    render_buf: sphrender.xyt_program.Buffer,

    hover_source: sphrender.xyt_program.RenderSource,
    hovering_pos: ?gui.PixelBBox,

    solid_color_renderer: *sphrender.xyt_program.SolidColorProgram,
    label_retriever: LabelRetriever,
    label_text: gui.gui_text.GuiText(*LabelRetriever),
    data: *const Data,
    key_idx: usize,

    const LabelRetriever = struct {
        buf: [4096]u8 = undefined,
        len: usize = 0,

        pub fn getText(self: *LabelRetriever) []const u8 {
            return self.buf[0..self.len];
        }
    };

    const vtable = gui.Widget(GuiAction).VTable{
        .render = render,
        .getSize = getSize,
        .update = update,
        .setInputState = setInputState,
        .setFocused = null,
        .reset = null,
    };

    fn widget(self: *GraphWidget) gui.Widget(GuiAction) {
        return .{
            .vtable = &vtable,
            .name = "graph",
            .ctx = self,
        };
    }

    fn render(ctx: ?*anyopaque, widget_bounds: gui.PixelBBox, window_bounds: gui.PixelBBox) void {
        const self: *GraphWidget = @ptrCast(@alignCast(ctx));
        gl.glLineWidth(3.0);

        const graph_bounds = self.graphBoundsFromWidgetBounds(widget_bounds);

        // FIXME: Center relative to thing
        var label_bounds = widget_bounds;
        label_bounds.top = graph_bounds.bottom;
        label_bounds.right = label_bounds.left + self.label_text.size().width;

        self.solid_color_renderer.renderLineStrip(self.render_source, .{
            .color = .{ 0.3, 0.3, 1 },
            .transform = gui.util.widgetToClipTransform(graph_bounds, window_bounds).inner,
        });

        if (self.hovering_pos) |pos| {
            std.debug.print("Rendering at {any}\n", .{pos});
            self.solid_color_renderer.render(self.hover_source, .{
                .color = .{ 0.3, 0.3, 1 },
                .transform = gui.util.widgetToClipTransform(pos, window_bounds).inner,
            });
        }

        self.label_text.render(gui.util.widgetToClipTransform(label_bounds, window_bounds));
    }

    fn getSize(ctx: ?*anyopaque) gui.PixelSize {
        const self: *GraphWidget = @ptrCast(@alignCast(ctx));
        return self.size;
    }

    fn update(ctx: ?*anyopaque, available_size: gui.PixelSize, delta_s: f32) anyerror!void {
        _ = delta_s;
        const self: *GraphWidget = @ptrCast(@alignCast(ctx));
        self.size = available_size;
        try self.label_text.update(available_size.width);
    }

    fn graphBoundsFromWidgetBounds(self: GraphWidget, widget_bounds: gui.PixelBBox) gui.PixelBBox {
        var graph_bounds = widget_bounds;
        graph_bounds.bottom -= self.label_text.size().height;
        return graph_bounds;
    }

    fn setInputState(ctx: ?*anyopaque, widget_bounds: gui.PixelBBox, input_bounds: gui.PixelBBox, input_state: gui.InputState) gui.InputResponse(GuiAction) {
        const self: *GraphWidget = @ptrCast(@alignCast(ctx));

        self.hovering_pos = null;
        if (!input_bounds.containsMousePos(input_state.mouse_pos)) {
            return .{};
        }

        // FIXME: Ew nasty casting in general
        const widget_left: f32 = @floatFromInt(widget_bounds.left);
        const mouse_offs_px = input_state.mouse_pos.x - widget_left;
        const mouse_offs_norm = mouse_offs_px / @as(f32, @floatFromInt(widget_bounds.calcWidth()));
        const mouse_step_idx: usize = @intFromFloat(mouse_offs_norm * @as(f32, @floatFromInt(self.data.steps.len)));

        var it = mouse_step_idx + 1;
        outer: while (it > 0) {
            it -= 1;
            const step = self.data.steps.get(it);
            var elem_it = step.elems.iter();
            while (elem_it.next()) |elem| {
                if (elem.key == self.key_idx) {
                    std.debug.print("Loss: {d}\n", .{elem.val});

                    const slice = std.fmt.bufPrint(&self.label_retriever.buf, "loss: {d}", .{elem.val}) catch &.{};
                    self.label_retriever.len = slice.len;
                    const graph_bounds = self.graphBoundsFromWidgetBounds(widget_bounds);

                    const hover_center_y: i32 = @intFromFloat(elem.val / self.data.maxes.get(elem.key) * @as(f32, @floatFromInt(graph_bounds.calcHeight())));
                    const hover_center_x: i32 = @intCast(it * graph_bounds.calcWidth() / self.data.steps.len);

                    self.hovering_pos = .{
                        .left = graph_bounds.left + hover_center_x - 5,
                        .right = graph_bounds.left + hover_center_x + 5,
                        .top = graph_bounds.bottom - hover_center_y - 5,
                        .bottom = graph_bounds.bottom - hover_center_y + 5,
                    };

                    break :outer;
                }
            }
        }

        return .{};
    }
};

fn glBufferFromDataKey(scratch: sphtud.alloc.LinearAllocator, gl_alloc: *sphrender.GlAlloc, data: Data, key_idx: usize) !sphrender.xyt_program.Buffer {
    const cp = scratch.checkpoint();
    defer scratch.restore(cp);

    var cpu_data = try sphtud.util.RuntimeBoundedArray(sphrender.xyt_program.Vertex).init(scratch.allocator(), data.steps.len);

    var step_it = data.steps.iter();
    var x_idx: usize = 0;
    while (step_it.next()) |step| {
        defer x_idx += 1;
        var elem_it = step.elems.iter();
        while (elem_it.next()) |elem| {
            if (elem.key == key_idx) {
                const gl_x: f32 = @as(f32, @floatFromInt(x_idx)) / @as(f32, @floatFromInt(data.steps.len)) * 2.0 - 1.0;
                const gl_y: f32 = elem.val / data.maxes.get(elem.key) * 2.0 - 1.0;

                try cpu_data.append(.{
                    .vPos = .{ gl_x, gl_y },
                });
            }
        }
    }

    return try sphrender.xyt_program.Buffer.init(gl_alloc, cpu_data.items);
}

pub fn main() !void {
    var allocators: sphrender.AppAllocators(100) = undefined;
    try allocators.initPinned(10 * 1024 * 1024);

    var window: sphwindow.Window = undefined;
    try window.initPinned("sphui demo", 800, 600);

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
    const layout = try widget_factory.makeLayout();
    try layout.pushWidget(try widget_factory.makeLabel("Hello world"));

    var solid_color_renderer = try sphrender.xyt_program.solidColorProgram(gui_alloc.gl);

    const data = try parseData(allocators.root.arena(), allocators.scratch.linear(), "train_output_saved/log.csv");

    var step_it = data.steps.iter();
    while (step_it.next()) |step| {
        std.debug.print("{d} {d} {d}\n", .{ step.iter, step.time_ns, step.img });

        var kv_it = step.elems.iter();
        while (kv_it.next()) |kv| {
            std.debug.print("\t{s} {d}\n", .{ data.keys.get(kv.key), kv.val });
        }
    }

    const graph_buf = try glBufferFromDataKey(allocators.scratch.linear(), gui_alloc.gl, data, 0);
    var graph_render_source = try sphrender.xyt_program.RenderSource.init(gui_alloc.gl);
    graph_render_source.bindData(solid_color_renderer.handle(), graph_buf);

    const hover_buf = try sphrender.xyt_program.Buffer.init(gui_alloc.gl, &.{
        .{ .vPos = .{ -1, -1 } },
        .{ .vPos = .{ -1, 1 } },
        .{ .vPos = .{ 1, 1 } },

        .{ .vPos = .{ -1, -1 } },
        .{ .vPos = .{ 1, 1 } },
        .{ .vPos = .{ 1, -1 } },
    });
    var hover_source = try sphrender.xyt_program.RenderSource.init(gui_alloc.gl);
    hover_source.bindData(solid_color_renderer.handle(), hover_buf);

    var graph_widget = GraphWidget{
        .solid_color_renderer = &solid_color_renderer,
        .render_source = graph_render_source,
        .render_buf = graph_buf,
        .hover_source = hover_source,
        .hovering_pos = null,
        .data = &data,
        .label_retriever = .{},
        .label_text = undefined,
        .key_idx = 0,
    };

    graph_widget.label_text = try gui.gui_text.guiText(gui_alloc, &gui_state.guitext_state, &graph_widget.label_retriever);
    try layout.pushWidget(graph_widget.widget());
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
        _ = response;
        window.swapBuffers();
    }
}
