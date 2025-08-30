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

pub fn OrientationRenderer(comptime Action: type) type {
    return struct {
        orientation_buffer: sphrender.xyt_program.Buffer,
        orientation_render_source: sphrender.xyt_program.RenderSource,
        orientation_renderer: *sphrender.xyt_program.SolidColorProgram,
        color: gui.Color,
        size: u31 = 0,

        const Self = @This();

        const vtable = gui.Widget(Action).VTable{
            .render = render,
            .getSize = getSize,
            .update = update,
            .setInputState = null,
            .setFocused = null,
            .reset = null,
        };

        pub fn asWidget(self: *Self) gui.Widget(Action) {
            return .{
                .ctx = self,
                .name = "OrientationRenderer",
                .vtable = &vtable,
            };
        }

        pub fn setOrientation(self: *Self, orientation: [2]f32) void {
            self.orientation_buffer.updateBuffer(&.{
                .{ .vPos = .{ 0, 0 } },
                .{ .vPos = .{ orientation[0], orientation[1] } },
            });
        }

        fn render(ctx: ?*anyopaque, widget_bounds: gui.PixelBBox, window_bounds: gui.PixelBBox) void {
            const self: *Self = @ptrCast(@alignCast(ctx));

            const txfm = gui.util.widgetToClipTransform(widget_bounds, window_bounds);

            self.orientation_renderer.renderLines(self.orientation_render_source, .{
                .color = .{ self.color.r, self.color.g, self.color.b },
                .transform = txfm.inner,
            });
        }

        fn getSize(ctx: ?*anyopaque) gui.PixelSize {
            const self: *Self = @ptrCast(@alignCast(ctx));
            return .{
                .width = self.size,
                .height = self.size,
            };
        }

        fn update(ctx: ?*anyopaque, available_size: gui.PixelSize, delta_s: f32) anyerror!void {
            const self: *Self = @ptrCast(@alignCast(ctx));
            _ = delta_s;
            self.size = @min(available_size.width, available_size.height);
        }
    };
}

pub fn orientationRenderer(comptime Action: type, alloc: sphrender.RenderAlloc, solid_color_renderer: *sphrender.xyt_program.SolidColorProgram, color: gui.Color) !*OrientationRenderer(Action) {
    const widget = try alloc.heap.arena().create(OrientationRenderer(Action));
    const orientation_buffer = try sphrender.xyt_program.Buffer.init(alloc.gl, &.{
        .{ .vPos = .{ 0, 0 } },
        .{ .vPos = .{ 0, 0 } },
    });
    var orientation_render_source = try sphrender.xyt_program.RenderSource.init(alloc.gl);
    orientation_render_source.bindData(solid_color_renderer.handle(), orientation_buffer);
    orientation_render_source.setLen(2);

    widget.* = OrientationRenderer(Action){
        .orientation_buffer = orientation_buffer,
        .orientation_render_source = orientation_render_source,
        .orientation_renderer = solid_color_renderer,
        .color = color,
    };
    return widget;
}

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

        fn setInputState(ctx: ?*anyopaque, widget_bounds: gui.PixelBBox, input_bounds: gui.PixelBBox, input_state: gui.InputState) gui.InputResponse(Action) {
            const self: *Self = @ptrCast(@alignCast(ctx));

            // image is scaled so it's width fits the widget width
            const widget_to_pixel_dist =
                asf32(self.image.width) / asf32(widget_bounds.calcWidth()) / self.scale;

            self.applyDrag(input_bounds, input_state, widget_to_pixel_dist);
            self.applyZoom(input_state);
            self.applyReset(input_state);
            return self.handleRightClick(widget_bounds, input_bounds, input_state, widget_to_pixel_dist);
        }

        fn applyDrag(self: *Self, input_bounds: gui.PixelBBox, input_state: gui.InputState, widget_to_pixel_dist: f32) void {
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

        fn applyZoom(self: *Self, input_state: gui.InputState) void {
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

        fn applyReset(self: *Self, input_state: gui.InputState) void {
            if (input_state.key_tracker.isKeyDown(.{ .ascii = 'r' })) {
                self.scale = 1.0;
                self.img_offset = .{
                    .x = 0.0,
                    .y = 0.0,
                };
            }
        }

        fn handleRightClick(self: *Self, widget_bounds: gui.PixelBBox, input_bounds: gui.PixelBBox, input_state: gui.InputState, widget_to_pixel_dist: f32) gui.InputResponse(Action) {
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
    var size = dims[0];
    for (dims[1..]) |v| {
        size *= v;
    }

    const img_cpu_rgba: []u8 = try alloc.alloc(u8, size * 4);

    for (0..data.len) |i| {
        const abs: u8 = @intFromFloat(@max(0, @min(@abs(255 * data[i] * mul), 255)));
        if (data[i] < 0) {
            img_cpu_rgba[i * 4 + 0] = 0;
            img_cpu_rgba[i * 4 + 2] = abs;
        } else {
            img_cpu_rgba[i * 4 + 0] = abs;
            img_cpu_rgba[i * 4 + 2] = 0;
        }
        img_cpu_rgba[i * 4 + 1] = 0;
        img_cpu_rgba[i * 4 + 3] = 255;
    }

    return CpuImage{
        .data = img_cpu_rgba,
        .width = dims[0],
    };
}
