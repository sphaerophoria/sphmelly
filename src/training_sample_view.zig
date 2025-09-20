const std = @import("std");
const sphtud = @import("sphtud");
const BarcodeGen = @import("BarcodeGen.zig");
const sphalloc = sphtud.alloc;
const sphrender = sphtud.render;
const gl = sphrender.gl;
const sphwindow = sphtud.window;
const gui = sphtud.ui;
const cl = @import("cl.zig");
const math = @import("math.zig");

pub const GlImage = struct {
    pub const empty = GlImage{
        .tex = .invalid,
        .width = 0,
        .height = 0,
    };
    tex: sphrender.Texture,
    width: usize,
    height: usize,
};

pub const ImagePixelPos = struct {
    x: u31,
    y: u31,
};

pub const CpuImage = struct {
    data: []u8,
    width: usize,

    pub const null_img = CpuImage{
        .data = &.{},
        .width = 0,
    };

    pub fn toGl(self: CpuImage, gl_alloc: *sphrender.GlAlloc) !GlImage {
        const tex = if (self.data.len != 0)
            try sphrender.makeTextureFromRgba(gl_alloc, self.data, self.width)
        else
            sphrender.Texture.invalid;
        return .{
            .tex = tex,
            .width = self.width,
            .height = self.calcHeight(),
        };
    }

    pub fn calcHeight(self: CpuImage) usize {
        if (self.data.len == 0) return 0;
        return self.data.len / self.width / 4;
    }
};

pub fn makeOrientationBuffer(gl_alloc: *sphrender.GlAlloc, orientation: [2]f32, solid_color_renderer: *sphrender.xyt_program.SolidColorProgram) !sphrender.xyt_program.RenderSource {
    const orientation_buffer = try sphrender.xyt_program.Buffer.init(gl_alloc, &.{
        .{ .vPos = .{ 0, 0 } },
        .{ .vPos = .{ orientation[0], orientation[1] } },
    });

    var orientation_render_source = try sphrender.xyt_program.RenderSource.init(gl_alloc);
    orientation_render_source.bindData(solid_color_renderer.handle(), orientation_buffer);
    orientation_render_source.setLen(2);

    return orientation_render_source;
}

pub const ImageRenderContext = struct {
    fbo: sphrender.FramebufferRenderContext,
    temporary_viewport: sphrender.TemporaryViewport,
    temporary_scissor: sphrender.TemporaryScissor,

    pub fn init(image: GlImage) !ImageRenderContext {
        const fbo = try sphrender.FramebufferRenderContext.init(image.tex, null);
        fbo.bind();

        const temporary_viewport = sphrender.TemporaryViewport.init();
        temporary_viewport.setViewport(@intCast(image.width), @intCast(image.height));

        const temporary_scissor = sphrender.TemporaryScissor.init();
        temporary_scissor.setAbsolute(0, 0, @intCast(image.width), @intCast(image.height));

        return .{
            .fbo = fbo,
            .temporary_viewport = temporary_viewport,
            .temporary_scissor = temporary_scissor,
        };
    }

    pub fn reset(self: ImageRenderContext) void {
        self.temporary_viewport.reset();
        self.temporary_scissor.reset();
        self.fbo.reset();
    }
};

pub fn ImageView(comptime Action: type) type {
    return struct {
        alloc: *sphrender.GlAlloc,
        image: GlImage,
        image_renderer: *sphrender.xyuvt_program.ImageRenderer,
        onReqStat: ?*const fn (pos: ImagePixelPos) Action,
        size: gui.PixelSize = .{},
        scale: f32 = 1.0,
        drag_state: DragState = .none,
        img_offset: ImgOffset = .{ .x = 0, .y = 0 },

        const ImgOffset = struct {
            x: f32,
            y: f32,
        };
        const DragState = union(enum) {
            dragging: struct {
                initial_mouse_pos: gui.MousePos,
                initial_offset: ImgOffset,
            },
            none,
        };

        const Self = @This();

        const vtable = gui.Widget(Action).VTable{
            .render = render,
            .getSize = getSize,
            .setInputState = setInputState,
            .update = update,
            .setFocused = null,
            .reset = null,
        };

        pub fn asWidget(self: *Self) gui.Widget(Action) {
            return .{
                .ctx = self,
                .name = "ImageView",
                .vtable = &vtable,
            };
        }

        pub fn setImg(self: *Self, img_data: CpuImage) !void {
            self.alloc.reset();
            self.image = try img_data.toGl(self.alloc);
        }

        pub fn resetPosition(self: *Self) void {
            self.scale = 1.0;
            self.img_offset = .{ .x = 0, .y = 0 };
        }

        fn render(ctx: ?*anyopaque, widget_bounds: gui.PixelBBox, window_bounds: gui.PixelBBox) void {
            const self: *Self = @ptrCast(@alignCast(ctx));

            var scissor = sphrender.TemporaryScissor.init();
            defer scissor.reset();

            scissor.set(widget_bounds.left, window_bounds.calcHeight() - widget_bounds.bottom, widget_bounds.calcWidth(), widget_bounds.calcHeight());

            var img_aspect: f32 = @floatFromInt(self.image.width);
            img_aspect /= @floatFromInt(self.image.height);

            const img_width_f: f32 = @floatFromInt(self.image.width);
            const img_height_f: f32 = @floatFromInt(self.image.height);

            var widget_aspect: f32 = @floatFromInt(widget_bounds.calcWidth());
            widget_aspect /= @floatFromInt(widget_bounds.calcHeight());

            const txfm =
                // [-1,1] -> img space
                sphtud.math.Transform.translate(
                    self.img_offset.x / img_width_f * 2.0,
                    -self.img_offset.y / img_height_f * 2.0,
                )
                    .then(sphtud.math.Transform.scale(
                        self.scale,
                        self.scale,
                    ))
                    // [-1,1] -> widget space
                    .then(sphtud.math.Transform.scale(
                        1.0,
                        widget_aspect / img_aspect,
                    ))
                    // [-1,1] -> window space
                    .then(gui.util.widgetToClipTransform(widget_bounds, window_bounds));

            self.image_renderer.renderTexture(
                self.image.tex,
                txfm,
            );
        }

        fn getSize(ctx: ?*anyopaque) gui.PixelSize {
            const self: *Self = @ptrCast(@alignCast(ctx));
            return self.size;
        }

        fn update(ctx: ?*anyopaque, available_size: gui.PixelSize, delta_s: f32) anyerror!void {
            const self: *Self = @ptrCast(@alignCast(ctx));
            _ = delta_s;
            self.size = available_size;
        }

        fn setInputState(ctx: ?*anyopaque, widget_bounds: gui.PixelBBox, input_bounds: gui.PixelBBox, input_state: *gui.InputState) gui.InputResponse(Action) {
            const self: *Self = @ptrCast(@alignCast(ctx));

            // image is scaled so it's width fits the widget width
            const widget_to_pixel_dist =
                asf32(self.image.width) / asf32(widget_bounds.calcWidth()) / self.scale;

            self.applyDrag(input_bounds, input_state, widget_to_pixel_dist);
            if (input_bounds.containsMousePos(input_state.mouse_pos)) {
                self.applyZoom(input_state);
                input_state.consumeScroll();
            }
            self.applyReset(input_state);
            return self.handleRightClick(widget_bounds, input_bounds, input_state, widget_to_pixel_dist);
        }

        fn applyDrag(self: *Self, input_bounds: gui.PixelBBox, input_state: *gui.InputState, widget_to_pixel_dist: f32) void {
            switch (self.drag_state) {
                .dragging => |state| blk: {
                    if (input_state.mouse_released) {
                        self.drag_state = .none;
                        break :blk;
                    }

                    const mouse_x_offs = input_state.mouse_pos.x - state.initial_mouse_pos.x;
                    const mouse_y_offs = input_state.mouse_pos.y - state.initial_mouse_pos.y;
                    self.img_offset = .{
                        .x = state.initial_offset.x + mouse_x_offs * widget_to_pixel_dist,
                        .y = state.initial_offset.y + mouse_y_offs * widget_to_pixel_dist,
                    };
                },
                .none => {
                    if (input_state.mouse_pressed and input_bounds.containsMousePos(input_state.mouse_pos)) {
                        self.drag_state = .{ .dragging = .{
                            .initial_mouse_pos = input_state.mouse_pos,
                            .initial_offset = self.img_offset,
                        } };
                    }
                },
            }
        }

        fn applyZoom(self: *Self, input_state: *gui.InputState) void {
            // Note that amount is in range [-N,N]
            // If we want the zoom adjustment to feel consistent, we need the
            // change from 4-8x to feel the same as the change from 1-2x
            // This means that a multiplicative level feels better than an additive one
            // So we need a function that goes from [-N,N] -> [lower than 1, greater than 1]
            // If we take this to the extreme, we want -inf -> 0, inf -> inf, 1 ->
            // 0. x^y provides this.
            // x^y also has the nice property that x^y*x^z == x^(y+z), which
            // results in merged scroll events acting the same as multiple split
            // events
            // Constant tuned until whatever scroll value we were getting felt ok
            //
            //
            // 1.1^(x+y) == 1.1^x * 1.1^y
            self.scale *= std.math.pow(f32, 1.1, input_state.frame_scroll);
        }

        fn applyReset(self: *Self, input_state: *gui.InputState) void {
            if (input_state.key_tracker.isKeyDown(.{ .ascii = 'r' })) {
                self.scale = 1.0;
                self.img_offset = .{
                    .x = 0.0,
                    .y = 0.0,
                };
            }
        }

        fn handleRightClick(self: *Self, widget_bounds: gui.PixelBBox, input_bounds: gui.PixelBBox, input_state: *gui.InputState, widget_to_pixel_dist: f32) gui.InputResponse(Action) {
            var response: gui.InputResponse(Action) = .{};

            if (!input_state.mouse_right_pressed or !input_bounds.containsMousePos(input_state.mouse_pos)) {
                return response;
            }

            const mouse_widget_x_offs = input_state.mouse_pos.x - widget_bounds.cx();
            const mouse_widget_y_offs = input_state.mouse_pos.y - widget_bounds.cy();

            const img_center_x = asf32(self.image.width) / 2.0;
            const img_center_y = asf32(self.image.height) / 2.0;

            const mouse_img_x_offs = mouse_widget_x_offs * widget_to_pixel_dist;
            const mouse_img_y_offs = mouse_widget_y_offs * widget_to_pixel_dist;

            const pixel_x: i32 = @intFromFloat(img_center_x - self.img_offset.x + mouse_img_x_offs);
            const pixel_y: i32 = @intFromFloat(img_center_y - self.img_offset.y + mouse_img_y_offs);

            if (pixel_x < 0 or pixel_x >= self.image.width or pixel_y < 0 or pixel_y >= self.image.height) {
                return response;
            }

            const image_height_i: i32 = @intCast(self.image.height);

            if (self.onReqStat) |f| {
                response.action = f(.{
                    .x = @intCast(pixel_x),
                    .y = @intCast(image_height_i - pixel_y - 1),
                });
            }

            return response;
        }

        fn asf32(val: anytype) f32 {
            return @floatFromInt(val);
        }
    };
}

pub fn greyTensorToRgbaCpu(alloc: std.mem.Allocator, img_tensor: math.Tensor([]f32)) !CpuImage {
    const img_cpu_rgba: []u8 = try alloc.alloc(u8, img_tensor.dims.numElems() * 4);

    for (0..img_tensor.buf.len) |i| {
        const luma: u8 = @intFromFloat(@max(0, @min(255 * img_tensor.buf[i], 255)));
        img_cpu_rgba[i * 4 + 0] = luma;
        img_cpu_rgba[i * 4 + 1] = luma;
        img_cpu_rgba[i * 4 + 2] = luma;
        img_cpu_rgba[i * 4 + 3] = 255;
    }

    return CpuImage{
        .data = img_cpu_rgba,
        .width = img_tensor.dims.get(0),
    };
}

pub fn gradTensorToRgbaCpu(alloc: std.mem.Allocator, data: []const f32, dims: []const u32, mul: f32) !CpuImage {
    return gradTensorToRgbaCpuAlpha(alloc, data, dims, mul, 1.0);
}

pub fn gradTensorToRgbaCpuAlpha(alloc: std.mem.Allocator, data: []const f32, dims: []const u32, mul: f32, alpha: f32) !CpuImage {
    var size = dims[0];
    for (dims[1..]) |v| {
        size *= v;
    }

    const img_cpu_rgba: []u8 = try alloc.alloc(u8, size * 4);

    for (0..data.len) |i| {
        const abs_scaled_data = @abs(std.math.clamp(data[i] * mul, -1, 1));
        const abs: u8 = @intFromFloat(abs_scaled_data * 255.0);
        if (std.math.isNan(data[i])) {
            img_cpu_rgba[i * 4 + 0] = 0;
            img_cpu_rgba[i * 4 + 1] = 255;
            img_cpu_rgba[i * 4 + 2] = 0;
        } else if (data[i] * mul < 0) {
            img_cpu_rgba[i * 4 + 0] = 0;
            img_cpu_rgba[i * 4 + 1] = 0;
            img_cpu_rgba[i * 4 + 2] = abs;
        } else {
            img_cpu_rgba[i * 4 + 0] = abs;
            img_cpu_rgba[i * 4 + 1] = 0;
            img_cpu_rgba[i * 4 + 2] = 0;
        }
        img_cpu_rgba[i * 4 + 3] = @intFromFloat(abs_scaled_data * alpha * 255);
    }

    return CpuImage{
        .data = img_cpu_rgba,
        .width = dims[0],
    };
}

pub fn makeBBoxGLBuffer(gl_alloc: *sphrender.GlAlloc, box: []const f32, program: *sphrender.xyt_program.SolidColorProgram) !sphrender.xyt_program.RenderSource {
    // [5]f32, x, y, w, h, rx, ry
    const x_x = box[4];
    const x_y = box[5];

    const theta = std.math.atan2(x_y, x_x);

    // OpenGL space is 2x as wide, so we take half width in normalized
    // space and double for opengl lol
    //
    // Note width and height are sqrt as inspired by YOLOv1 paper
    const half_width: sphtud.math.Vec2 = @splat(box[2] * box[2]);
    const half_height: sphtud.math.Vec2 = @splat(box[3] * box[3]);

    // cx/cy are represented as offsets from center in normalized space, OpenGL
    // space is the same but 2x the distance
    const cx = box[0] * 2.0;
    const cy = box[1] * 2.0;

    const center = sphtud.math.Vec2{ cx, cy };

    const sint = @sin(theta);
    const cost = @cos(theta);
    const x_axis = sphtud.math.Vec2{ cost, sint };
    const y_axis = sphtud.math.Vec2{ -sint, cost };

    const points: []const sphrender.xyt_program.Vertex = &.{
        .{ .vPos = center + half_width * x_axis },
        .{ .vPos = center + half_width * x_axis - half_height * y_axis },
        .{ .vPos = center - half_width * x_axis - half_height * y_axis },
        .{ .vPos = center - half_width * x_axis + half_height * y_axis },
        .{ .vPos = center + half_width * x_axis + half_height * y_axis },
        .{ .vPos = center + half_width * x_axis },
        .{ .vPos = center },
    };

    const gl_buf = try sphrender.xyt_program.Buffer.init(gl_alloc, points);
    var render_source = try sphrender.xyt_program.RenderSource.init(gl_alloc);
    render_source.bindData(program.handle(), gl_buf);

    return render_source;
}
