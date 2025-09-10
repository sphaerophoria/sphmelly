const std = @import("std");
const sphtud = @import("sphtud");
const math = @import("math.zig");
const tsv = @import("training_sample_view.zig");
const gl = sphtud.render.gl;
const Config = @import("Config.zig");

pub const Gui = struct {
    window: sphtud.window.Window,
    params: GuiParams,
    widgets: Widgets,
    train_num_images: u32,

    pub fn initPinned(self: *Gui, allocators: anytype, initial_lr: f32, train_num_images: u32, train_mode: Config.TrainTarget) !void {
        self.* = .{
            .params = .init(initial_lr),
            .window = undefined,
            .widgets = undefined,
            .train_num_images = train_num_images,
        };

        try self.window.initPinned("sphmelly", 800, 600);

        gl.glEnable(gl.GL_SCISSOR_TEST);
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA);
        gl.glEnable(gl.GL_BLEND);
        gl.glLineWidth(2.0);

        const gui_alloc = try allocators.root_render.makeSubAlloc("gui");
        self.widgets = try makeGuiWidgets(gui_alloc, &allocators.scratch, &allocators.scratch_gl, &self.params, train_mode);
    }

    pub fn clear(self: *Gui) !void {
        try self.widgets.image_view.setImg(.null_img);
        self.params.current_img_data = .null_img;
    }

    pub fn setImageGrayscale(self: *Gui, scratch: std.mem.Allocator, img: CpuTensor) !void {
        const rgba = try tsv.greyTensorToRgbaCpu(scratch, img);
        try self.widgets.image_view.setImg(rgba);
        self.params.current_img_data = .{
            .data = img.buf,
            .width = img.dims.get(0),
        };
    }

    pub fn setImageHotCold(self: *Gui, scratch: std.mem.Allocator, img: CpuTensor) !void {
        const rgba = try tsv.gradTensorToRgbaCpu(scratch, img.buf, img.dims.inner, self.params.grad_mul);
        try self.widgets.image_view.setImg(rgba);
        self.params.current_img_data = .{
            .data = img.buf,
            .width = img.dims.get(0),
        };
    }

    pub fn addImgOverlayHotCold(self: *Gui, scratch: std.mem.Allocator, scratch_gl: *sphtud.render.GlAlloc, overlay: CpuTensor, extra_mul: f32) !void {
        const rgba = try tsv.gradTensorToRgbaCpuAlpha(scratch, overlay.buf, overlay.dims.inner, self.params.grad_mul * extra_mul, 0.5);

        const render_ctx = try tsv.ImageRenderContext.init(self.widgets.image_view.image);
        defer render_ctx.reset();

        const gl_cp = scratch_gl.checkpoint();
        defer scratch_gl.restore(gl_cp);

        const overlay_tex = try sphtud.render.makeTextureFromRgba(scratch_gl, rgba.data, rgba.width);
        self.widgets.image_view.image_renderer.renderTexture(overlay_tex, .identity);
    }

    pub fn renderBBoxOverlay(self: *Gui, scratch_gl: *sphtud.render.GlAlloc, overlay: CpuTensor, color: sphtud.math.Vec3) !void {
        const render_ctx = try tsv.ImageRenderContext.init(self.widgets.image_view.image);
        defer render_ctx.reset();

        const gl_cp = scratch_gl.checkpoint();
        defer scratch_gl.restore(gl_cp);

        const bbox_source = try tsv.makeBBoxGLBuffer(scratch_gl, overlay.buf, self.widgets.solid_color_renderer);
        self.widgets.solid_color_renderer.renderLineStrip(bbox_source, .{
            .color = color,
            .transform = sphtud.math.Transform.identity.inner,
        });
    }

    pub fn renderBarComparison(self: *Gui, _: *sphtud.render.GlAlloc, predicted: CpuTensor, expected: CpuTensor) !void {
        std.debug.assert(predicted.dims.eql(&.{barcode_size}));
        std.debug.assert(expected.dims.eql(&.{barcode_size}));

        var rgba: [barcode_size]TexColor = undefined;

        for (predicted.buf, 0..) |v_in, i| {
            const v = 1.0 / (1.0 + std.math.exp(-v_in));
            rgba[i] = .grey(v);
        }
        self.widgets.bar_retrievers.?.predicted.setData(&rgba);

        for (expected.buf, 0..) |v, i| {
            rgba[i] = .grey(v);
        }
        self.widgets.bar_retrievers.?.actual.setData(&rgba);

        for (0..barcode_size) |i| {
            const predicted_true = predicted.buf[i] > 0;
            const expected_true = expected.buf[i] > 0.5;
            const matches = predicted_true == expected_true;
            rgba[i] = if (matches) .green else .red;
        }
        self.widgets.bar_retrievers.?.comparison.setData(&rgba);
    }

    pub fn step(ui: *Gui, width: u31, height: u31) !?GuiAction {
        const res = try ui.widgets.runner.step(1.0, .{ .width = width, .height = height }, &ui.window.queue);

        if (res.action) |a| switch (a) {
            .pause => {},
            .update_img => |img| {
                ui.params.current_img = std.math.clamp(img, 0, ui.train_num_images - 1);
            },
            .update_layer => |v| {
                ui.params.current_layer = @min(v, ui.params.num_layers -| 1);
            },
            .update_param => |v| {
                ui.params.current_param = v;
            },
            .select_view_mode => |mode| {
                ui.params.view_mode = @enumFromInt(mode);
                ui.widgets.image_view.resetPosition();
            },
            .grad_mul => |v| {
                ui.params.grad_mul = v;
            },
            .img_stat_req => |v| {
                ui.params.selected_pixel = v;
            },
            .lr_update => |v| {
                ui.params.lr = std.math.clamp(v, 0, 1);
            },
            .none => {},
        };

        return res.action;
    }
};

pub const GuiAction = union(enum) {
    none,
    pause,
    update_img: u32,
    update_layer: u32,
    update_param: u32,
    grad_mul: f32,
    select_view_mode: usize,
    img_stat_req: tsv.ImagePixelPos,
    lr_update: f32,

    fn genUpdateImg(val: u32) GuiAction {
        return .{ .update_img = val };
    }

    fn genUpdateLayer(val: u32) GuiAction {
        return .{ .update_layer = val };
    }

    fn genUpdateParam(val: u32) GuiAction {
        return .{ .update_param = val };
    }

    fn genUpdateGradMul(val: f32) GuiAction {
        return .{ .grad_mul = val };
    }

    fn genSelectViewMode(val: usize) GuiAction {
        return .{ .select_view_mode = val };
    }

    fn genImgStatReq(val: tsv.ImagePixelPos) GuiAction {
        return .{ .img_stat_req = val };
    }

    fn genLrUpdate(val: f32) GuiAction {
        return .{ .lr_update = val };
    }
};

const CpuTensor = math.Tensor([]f32);

const barcode_size = 95;

const TexColor = packed struct {
    r: u8,
    g: u8,
    b: u8,
    a: u8,

    const red = @This(){ .r = 255, .g = 0, .b = 0, .a = 255 };
    const green = @This(){ .r = 0, .g = 255, .b = 0, .a = 255 };

    fn grey(val: f32) @This() {
        const val_u8: u8 = @intFromFloat(std.math.clamp(val * 255, 0, 255));
        return .{
            .r = val_u8,
            .g = val_u8,
            .b = val_u8,
            .a = 255,
        };
    }
};

const BarRetriever = struct {
    texture: sphtud.render.Texture,

    pub fn init(alloc: *sphtud.render.GlAlloc) !BarRetriever {
        return .{
            .texture = try sphtud.render.makeTextureCommon(alloc),
        };
    }
    pub fn getSize(_: BarRetriever) sphtud.ui.PixelSize {
        return .{
            .width = barcode_size,
            .height = 1,
        };
    }

    pub fn getTexture(self: BarRetriever) sphtud.render.Texture {
        return self.texture;
    }

    pub fn setData(self: BarRetriever, data: []const TexColor) void {
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture.inner);
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, barcode_size, 1, 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, data.ptr);
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0);
    }
};

const BarRetrievers = struct {
    predicted: BarRetriever,
    actual: BarRetriever,
    comparison: BarRetriever,
};
const Widgets = struct {
    image_view: *tsv.ImageView(GuiAction),
    runner: sphtud.ui.runner.Runner(GuiAction),
    solid_color_renderer: *sphtud.render.xyt_program.SolidColorProgram,
    bar_retrievers: ?BarRetrievers,
};

fn makeGuiWidgets(gui_alloc: sphtud.ui.GuiAlloc, scratch: *sphtud.alloc.BufAllocator, scratch_gl: *sphtud.render.GlAlloc, gui_params: *GuiParams, train_target: Config.TrainTarget) !Widgets {
    const gui_state = try sphtud.ui.widget_factory.widgetState(
        GuiAction,
        gui_alloc,
        scratch,
        scratch_gl,
    );

    const widget_factory = gui_state.factory(gui_alloc);

    const solid_color_renderer = try gui_alloc.heap.arena().create(sphtud.render.xyt_program.SolidColorProgram);
    solid_color_renderer.* = try sphtud.render.xyt_program.solidColorProgram(gui_alloc.gl);
    var image_view = try makeImageView(gui_alloc, gui_state);

    const left_to_right = try widget_factory.makeLayout();
    left_to_right.cursor.direction = .left_to_right;

    const left_bar_layout = try widget_factory.makeLayout();
    try left_to_right.pushWidget(
        try widget_factory.makeBox(left_bar_layout.asWidget(), .{ .width = 300, .height = 0 }, .fill_height),
    );

    const pause_action: GuiAction = GuiAction.pause;
    try left_bar_layout.pushWidget(try widget_factory.makeButton("Pause", pause_action));

    try left_bar_layout.pushWidget(try widget_factory.makeLabel("Learning rate"));
    try left_bar_layout.pushWidget(try widget_factory.makeDrag(f32, &gui_params.lr, &GuiAction.genLrUpdate, 1, 1000));

    try left_bar_layout.pushWidget(try widget_factory.makeBox(
        try widget_factory.makeSelectableList(
            ViewModeRetriever{ .current_mode = &gui_params.view_mode },
            &GuiAction.genSelectViewMode,
        ),
        .{ .width = 300, .height = 0 },
        .fill_height,
    ));

    try left_bar_layout.pushWidget(try widget_factory.makeLabel("Image id"));
    try left_bar_layout.pushWidget(try widget_factory.makeDrag(u32, &gui_params.current_img, &GuiAction.genUpdateImg, 1, 20));

    try left_bar_layout.pushWidget(try widget_factory.makeLabel("Layer ID"));
    try left_bar_layout.pushWidget(try widget_factory.makeLabel(&gui_params.layer_name));
    try left_bar_layout.pushWidget(try widget_factory.makeDrag(u32, &gui_params.current_layer, &GuiAction.genUpdateLayer, 1, 20));

    try left_bar_layout.pushWidget(try widget_factory.makeLabel("Param ID"));
    try left_bar_layout.pushWidget(try widget_factory.makeDrag(u32, &gui_params.current_param, &GuiAction.genUpdateParam, 1, 20));

    try left_bar_layout.pushWidget(try widget_factory.makeLabel("Gradient multiplier"));
    try left_bar_layout.pushWidget(try widget_factory.makeDrag(f32, &gui_params.grad_mul, &GuiAction.genUpdateGradMul, 1, 100));

    const right_layout = try widget_factory.makeLayout();
    try left_to_right.pushWidget(right_layout.asWidget());

    try right_layout.pushWidget(try widget_factory.makeLabel(LossLabelRetriever{ .loss = &gui_params.current_loss }));

    try right_layout.pushWidget(try widget_factory.makeLabel(ImgValueRetriever{ .current_pixel_data = &gui_params.current_img_data, .current_pixel_pos = &gui_params.selected_pixel, .buf = undefined }));

    try right_layout.pushWidget(try widget_factory.makeLabel(TrainingMetaRetriever{
        .img_loss = &gui_params.img_loss,
    }));

    var bar_retrievers: ?BarRetrievers = null;
    if (train_target == .bars) {
        bar_retrievers = BarRetrievers{
            .predicted = try .init(gui_alloc.gl),
            .actual = try .init(gui_alloc.gl),
            .comparison = try .init(gui_alloc.gl),
        };

        const pairs = &.{
            .{ bar_retrievers.?.predicted, "predicted" },
            .{ bar_retrievers.?.actual, "actual" },
            .{ bar_retrievers.?.comparison, "comparison" },
        };

        inline for (pairs) |pair| {
            try right_layout.pushWidget(try widget_factory.makeLabel(pair[1]));
            try right_layout.pushWidget(try widget_factory.makeBox(
                try widget_factory.makeThumbnail(pair[0]),
                .{ .width = 0, .height = 20 },
                .fill_width,
            ));
        }
    }

    try right_layout.pushWidget(image_view.asWidget());

    return .{
        .image_view = image_view,
        .runner = try widget_factory.makeRunner(left_to_right.asWidget()),
        .solid_color_renderer = solid_color_renderer,
        .bar_retrievers = bar_retrievers,
    };
}

const ViewMode = enum {
    training_sample,
    gradients,
    weights,
    layer_out,
};

const ViewModeRetriever = struct {
    current_mode: *ViewMode,

    pub fn numItems(_: ViewModeRetriever) usize {
        return @typeInfo(ViewMode).@"enum".fields.len;
    }

    pub fn selectedId(self: ViewModeRetriever) usize {
        return @intFromEnum(self.current_mode.*);
    }

    pub fn getText(_: ViewModeRetriever, idx: usize) []const u8 {
        const idx_enum: ViewMode = @enumFromInt(idx);
        return @tagName(idx_enum);
    }
};

const CpuTensorFlat = struct {
    const null_img = CpuTensorFlat{
        .data = &.{},
        .width = 0,
    };

    data: []const f32,
    width: usize,
};

const ImgValueRetriever = struct {
    current_pixel_data: *CpuTensorFlat,
    current_pixel_pos: *?tsv.ImagePixelPos,
    buf: [30]u8,

    pub fn getText(self: *ImgValueRetriever) []const u8 {
        const none_string = "selected value: none";

        const pos = self.current_pixel_pos.* orelse return none_string;

        const idx = pos.y * self.current_pixel_data.width + pos.x;
        if (idx < self.current_pixel_data.data.len) {
            return std.fmt.bufPrint(&self.buf, "selected value ({d},{d}): {d:.5}", .{ pos.x, pos.y, self.current_pixel_data.data[idx] }) catch &self.buf;
        } else {
            return none_string;
        }
    }
};

const LossLabelRetriever = struct {
    loss: *f32,
    buf: [20]u8 = undefined,

    pub fn getText(self: *LossLabelRetriever) []const u8 {
        return std.fmt.bufPrint(&self.buf, "{d}", .{self.loss.*}) catch &self.buf;
    }
};

const TrainingMetaRetriever = struct {
    img_loss: *f32,
    buf: [200]u8 = undefined,

    pub fn getText(self: *TrainingMetaRetriever) []const u8 {
        return std.fmt.bufPrint(&self.buf,
            \\Elem loss: {d}
        , .{self.img_loss.*}) catch &self.buf;
    }
};

fn makeImageView(gui_alloc: sphtud.render.RenderAlloc, gui_state: *sphtud.ui.widget_factory.WidgetState(GuiAction)) !*tsv.ImageView(GuiAction) {
    const alloc = try gui_alloc.makeSubAlloc("image view");

    const widget = try gui_alloc.heap.arena().create(tsv.ImageView(GuiAction));
    widget.* = tsv.ImageView(GuiAction){
        .alloc = alloc.gl,
        .image = .{
            .tex = .invalid,
            .width = 0,
            .height = 0,
        },
        .image_renderer = &gui_state.image_renderer,
        .onReqStat = &GuiAction.genImgStatReq,
    };
    return widget;
}

const GuiParams = struct {
    current_loss: f32 = 0.0,
    current_img: u32 = 0,
    current_layer: u32 = 0,
    current_param: u32 = 0,
    img_loss: f32 = 0.0,
    num_layers: u32 = 0,
    grad_mul: f32 = 1.0,
    view_mode: ViewMode = .training_sample,
    layer_name: []const u8 = &.{},
    current_img_data: CpuTensorFlat = .null_img,
    selected_pixel: ?tsv.ImagePixelPos = null,
    lr: f32,

    fn init(initial_lr: f32) GuiParams {
        return .{
            .lr = initial_lr,
        };
    }
};
