const std = @import("std");
const sphtud = @import("sphtud");
const sphalloc = sphtud.alloc;
const sphrender = sphtud.render;
const gl = sphrender.gl;
const sphwindow = sphtud.window;
const gui = sphtud.ui;
const GraphPoint = gui.multi_line_graph.GraphPoint;

const parser = @import("train_output_vis/parser.zig");

const GuiAction = union(enum) {
    update_scale_method: XAxisScaleMethod.Method,
    update_log_color: struct {
        idx: usize,
        color: gui.Color,
    },
    update_log_enable: usize,
    update_smoothing: u32,

    fn genUpdateScaleMethod(idx: usize) GuiAction {
        return .{ .update_scale_method = @enumFromInt(idx) };
    }

    fn genUpdateSmoothing(val: u32) GuiAction {
        return .{ .update_smoothing = val };
    }
};

const MovingAverage = struct {
    data: []f32,
    size: usize,
    idx: usize,

    pub fn push(self: *MovingAverage, elem: f32) void {
        self.data[self.idx] = elem;
        self.size = @min(self.data.len, self.size + 1);
        self.idx = (self.idx + 1) % self.data.len;
    }

    pub fn average(self: MovingAverage) f32 {
        var sum: f32 = 0;
        for (self.data) |d| sum += d;
        return sum / asf32(self.size);
    }
};

fn asf32(in: anytype) f32 {
    return @floatFromInt(in);
}

const Smoothing = struct {
    generation: u32,
    val: u32,
};

const TrainingDataRetriever = struct {
    scratch: std.mem.Allocator,
    data_generation: usize = 0,
    scale_method: *XAxisScaleMethod,
    vis_state: *const TrainingLogVisState,
    key_data: ?*parser.KeyData,
    smoothing: *Smoothing,
    steps: *const parser.Steps,

    pub fn getGeneration(self: TrainingDataRetriever) usize {
        return self.smoothing.generation + self.scale_method.generation;
    }

    pub fn getColor(self: TrainingDataRetriever) gui.Color {
        return self.vis_state.color;
    }

    pub fn getEnable(self: TrainingDataRetriever) bool {
        return self.vis_state.enable;
    }

    pub const Iter = struct {
        pub const empty = Iter{ .inner = null };

        inner: ?struct {
            parent: *TrainingDataRetriever,
            step_id_it: sphtud.util.RuntimeSegmentedList(parser.StepId).Iter,
            val_it: sphtud.util.RuntimeSegmentedList(f32).Iter,
            moving_average: MovingAverage,
        },

        pub fn next(self: *Iter) ?GraphPoint {
            const inner = &(self.inner orelse return null);
            const step_id = inner.step_id_it.next() orelse return null;
            const step = inner.parent.steps.get(step_id.*);

            inner.moving_average.push((inner.val_it.next() orelse unreachable).*);

            return .{
                .x = resolveStepX(step, inner.parent.scale_method.*),
                .y = inner.moving_average.average(),
            };
        }
    };

    pub fn iter(self: *TrainingDataRetriever) Iter {
        const key_data = self.key_data orelse return .empty;
        return .{ .inner = .{
            .parent = self,
            .step_id_it = key_data.steps.iter(),
            .val_it = key_data.vals.iter(),
            .moving_average = .{
                .data = self.scratch.alloc(f32, self.smoothing.val) catch unreachable,
                .size = 0,
                .idx = 0,
            },
        } };
    }

    pub fn len(self: TrainingDataRetriever) usize {
        const key_data = self.key_data orelse return 0;
        return key_data.steps.len;
    }

    pub fn getBounds(self: TrainingDataRetriever) GraphPoint {
        const key_data = self.key_data orelse return .{ .x = 0, .y = 0 };
        const last_step_id = key_data.steps.get(key_data.steps.len - 1);
        const last_step = self.steps.get(last_step_id);
        return .{
            .x = resolveStepX(last_step, self.scale_method.*),
            .y = key_data.max_val,
        };
    }

    pub fn closestPoint(self: TrainingDataRetriever, x_val: f32) ?GraphPoint {
        const key_data = self.key_data orelse return null;
        // FIXME: This should probably be done one time off rip :)
        const step_ids = key_data.steps.makeContiguous(self.scratch) catch unreachable;
        const StepSearchCtx = struct {
            desired: f32,
            steps: *const parser.Steps,
            scale_method: XAxisScaleMethod,

            fn compare(search_ctx: @This(), other_id: parser.StepId) std.math.Order {
                const other_step = search_ctx.steps.get(other_id);
                return std.math.order(search_ctx.desired, resolveStepX(other_step, search_ctx.scale_method));
            }
        };
        const idx = std.sort.lowerBound(parser.StepId, step_ids, StepSearchCtx{ .desired = x_val, .steps = self.steps, .scale_method = self.scale_method.* }, StepSearchCtx.compare) -| 1;

        const closest_step_id = key_data.steps.get(idx);
        const closest_step = self.steps.get(closest_step_id);
        const closest_val = key_data.vals.get(idx);

        var smoothed = closest_val;
        var smoothed_num: u32 = 1;
        for (1..self.smoothing.val) |offs| {
            if (offs > idx) break;
            smoothed += key_data.vals.get(idx - offs);
            smoothed_num += 1;
        }

        return .{
            .x = resolveStepX(closest_step, self.scale_method.*),
            .y = smoothed / asf32(smoothed_num),
        };
    }
};

fn resolveStepX(step: parser.Step, method: XAxisScaleMethod) f32 {
    return switch (method.method) {
        .iter => asf32(step.iter),
        .@"wall time" => asf32(step.time_ns) / std.time.ns_per_s,
        .@"images processed" => asf32(step.img),
    };
}

const Args = struct {
    training_dirs: []const []const u8,

    fn init(alloc: std.mem.Allocator) !Args {
        var args = try std.process.argsAlloc(alloc);
        if (args.len < 2) {
            return error.NoTrainingLogs;
        }

        return .{
            .training_dirs = args[1..],
        };
    }
};

const XAxisScaleMethod = struct {
    method: Method,
    generation: usize,

    const Method = enum {
        iter,
        @"wall time",
        @"images processed",
    };
};

const ScaleMethodAdaptor = struct {
    scale_method: *XAxisScaleMethod,

    pub fn getText(_: ScaleMethodAdaptor, idx: usize) []const u8 {
        return @tagName(@as(XAxisScaleMethod.Method, @enumFromInt(idx)));
    }

    pub fn selectedId(self: ScaleMethodAdaptor) usize {
        return @intFromEnum(self.scale_method.method);
    }

    pub fn numItems(_: ScaleMethodAdaptor) usize {
        return std.meta.fields(XAxisScaleMethod.Method).len;
    }
};

const ColorUpdateGenerator = struct {
    idx: usize,

    pub fn generate(self: ColorUpdateGenerator, color: gui.Color) GuiAction {
        return .{ .update_log_color = .{
            .idx = self.idx,
            .color = color,
        } };
    }
};

const TrainingLogVisState = struct {
    enable: bool,
    color: gui.Color,
};

pub fn makeSidebar(training_logs: []const parser.TrainingLog, vis_state: []const TrainingLogVisState, color_shared: *gui.color_picker.SharedColorPickerState, scale_method: *XAxisScaleMethod, smoothing: *u32, widget_factory: gui.widget_factory.WidgetFactory(GuiAction)) !gui.Widget(GuiAction) {
    const layout = try widget_factory.makeLayout();

    try layout.pushWidget(
        try widget_factory.makeLabel("X axis scaling"),
    );

    try layout.pushWidget(
        try widget_factory.makeSelectableList(
            ScaleMethodAdaptor{ .scale_method = scale_method },
            &GuiAction.genUpdateScaleMethod,
        ),
    );

    try layout.pushWidget(
        try widget_factory.makeLabel("Smoothing"),
    );

    try layout.pushWidget(
        try widget_factory.makeDrag(u32, smoothing, &GuiAction.genUpdateSmoothing, 1, 20),
    );

    try layout.pushWidget(
        try widget_factory.makeLabel("Data"),
    );

    var log_name_grid = try widget_factory.makeGrid(
        &.{
            .{
                .width = .{ .fixed = widget_factory.state.checkbox_shared.style.outer_size },
                .horizontal_justify = .center,
                .vertical_justify = .center,
            },
            .{
                .width = .{ .ratio = 1.0 },
                .horizontal_justify = .left,
                .vertical_justify = .center,
            },
            .{
                .width = .{ .fixed = color_shared.style.preview_width },
                .horizontal_justify = .center,
                .vertical_justify = .center,
            },
        },
        training_logs.len * 3,
        training_logs.len * 3,
    );

    try layout.pushWidget(log_name_grid.asWidget());

    for (training_logs, 0..) |log, i| {
        try log_name_grid.pushWidget(
            try widget_factory.makeCheckbox(
                &vis_state[i].enable,
                GuiAction{ .update_log_enable = i },
            ),
        );

        try log_name_grid.pushWidget(
            try widget_factory.makeLabel(log.source),
        );

        const preview = try widget_factory.makeColorPickerWithShared(
            color_shared,
            &vis_state[i].color,
            ColorUpdateGenerator{
                .idx = i,
            },
        );

        try log_name_grid.pushWidget(preview);
    }

    return try widget_factory.makeBox(
        layout.asWidget(),
        .{ .width = 300, .height = 0 },
        .fill_height,
    );
}

pub fn initVisState(alloc: std.mem.Allocator, num_elems: usize) ![]TrainingLogVisState {
    const palette: []const gui.Color = &.{
        .{ .r = 1, .g = 0.35, .b = 0.87, .a = 1 },
        .{ .r = 0.47, .g = 1, .b = 0.26, .a = 1 },
        .{ .r = 1, .g = 0.73, .b = 0.35, .a = 1 },
        .{ .r = 0.31, .g = 1, .b = 0.94, .a = 1 },
        .{ .r = 0.73, .g = 0.69, .b = 1, .a = 1 },
        .{ .r = 1, .g = 1, .b = 0, .a = 1 },
        .{ .r = 1, .g = 0, .b = 0, .a = 1 },
    };

    const ret = try alloc.alloc(TrainingLogVisState, num_elems);
    for (ret, 0..) |*out, i| {
        out.* = .{
            .enable = true,
            .color = palette[i % palette.len],
        };
    }

    return ret;
}

pub fn makeGraphLayout(
    widget_factory: gui.widget_factory.WidgetFactory(GuiAction),
    scratch: sphtud.alloc.LinearAllocator,
    training_data: []const parser.TrainingLog,
    vis_state: []const TrainingLogVisState,
    keys: *parser.Keys,
    scale_method: *XAxisScaleMethod,
    smoothing: *Smoothing,
) !gui.Widget(GuiAction) {
    const graph_layout = try widget_factory.makeLayout();

    var key_it = keys.iter();
    while (key_it.next()) |key_id| {
        const retrievers = try widget_factory.alloc.heap.arena().alloc(TrainingDataRetriever, training_data.len);

        for (training_data, retrievers, 0..) |*in, *retriever, i| {
            retriever.* = .{
                .scratch = scratch.allocator(),
                .vis_state = &vis_state[i],
                .scale_method = scale_method,
                .key_data = in.keyed_data.getPtr(key_id),
                .steps = &in.steps,
                .smoothing = smoothing,
            };
        }

        const graph_widget = try widget_factory.makeMultiLineGraph(retrievers);

        try graph_layout.pushWidget(try widget_factory.makeLabel(keys.get(key_id)));
        try graph_layout.pushWidget(
            try widget_factory.makeBox(graph_widget, .{ .width = 300, .height = 300 }, .fill_none),
        );
    }

    return try widget_factory.makeScrollView(graph_layout.asWidget());
}

pub fn main() !void {
    var allocators: sphrender.AppAllocators(100) = undefined;
    try allocators.initPinned(10 * 1024 * 1024);

    var window: sphwindow.Window = undefined;
    try window.initPinned("training data vis", 1080, 600);

    gl.glEnable(gl.GL_SCISSOR_TEST);
    gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA);
    gl.glEnable(gl.GL_BLEND);

    const args = try Args.init(allocators.root.arena());

    const gui_alloc = try allocators.root_render.makeSubAlloc("gui");

    const gui_state = try gui.widget_factory.widgetState(
        GuiAction,
        gui_alloc,
        &allocators.scratch,
        &allocators.scratch_gl,
    );

    const widget_factory = gui_state.factory(gui_alloc);

    const training_data: []parser.TrainingLog = try allocators.root.arena().alloc(parser.TrainingLog, args.training_dirs.len);
    var keys = try parser.Keys.init(allocators.root.general());
    for (args.training_dirs, training_data) |p, *out| {
        out.* = try parser.parseData(allocators.root.arena(), allocators.scratch.linear(), p, &keys);
    }

    var scale_method = XAxisScaleMethod{
        .method = .iter,
        .generation = 0,
    };
    var smoothing: Smoothing = .{ .generation = 0, .val = 1 };

    const vis_state = try initVisState(allocators.root.arena(), training_data.len);
    const graph_scroll = try makeGraphLayout(widget_factory, allocators.scratch.linear(), training_data, vis_state, &keys, &scale_method, &smoothing);

    var sidebar_color_shared = gui_state.shared_color;
    sidebar_color_shared.style.preview_width = sidebar_color_shared.style.preview_height;

    const sidebar_layout = try makeSidebar(training_data, vis_state, &sidebar_color_shared, &scale_method, &smoothing.val, widget_factory);

    var toplevel_layout = try widget_factory.makeLayout();
    toplevel_layout.cursor.direction = .left_to_right;

    try toplevel_layout.pushWidget(sidebar_layout);
    try toplevel_layout.pushWidget(graph_scroll);

    var runner = try widget_factory.makeRunner(toplevel_layout.asWidget());

    const background = gui.widget_factory.StyleColors.background_color;
    while (!window.closed()) {
        allocators.resetScratch();
        const width, const height = window.getWindowSize();

        gl.glViewport(0, 0, @intCast(width), @intCast(height));
        gl.glScissor(0, 0, @intCast(width), @intCast(height));

        gl.glClearColor(background.r, background.g, background.b, background.a);
        gl.glClear(gl.GL_COLOR_BUFFER_BIT);

        const response = try runner.step(1.0, .{
            .width = @intCast(width),
            .height = @intCast(height),
        }, &window.queue);

        if (response.action) |a| switch (a) {
            .update_scale_method => |new_method| {
                scale_method.method = new_method;
                scale_method.generation += 1;
            },
            .update_log_color => |params| {
                vis_state[params.idx].color = params.color;
            },
            .update_log_enable => |idx| {
                vis_state[idx].enable = !vis_state[idx].enable;
            },
            .update_smoothing => |val| {
                if (smoothing.val != val) {
                    smoothing.generation += 1;
                    smoothing.val = @max(val, 1);
                }
            },
        };
        window.swapBuffers();
    }
}
