const std = @import("std");
const sphtud = @import("sphtud");
const gl = sphtud.render.gl;
const math = @import("math.zig");

const barcode_size = 95;

pub const TexColor = packed struct {
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

pub fn ComparisonView(comptime Action: type) type {
    return struct {
        predicted: BarRetriever,
        actual: BarRetriever,
        comparison: BarRetriever,
        widget: sphtud.ui.Widget(Action),

        const CpuTensor = math.Tensor([]f32);

        const Self = @This();

        pub fn renderBarComparison(self: *Self, predicted: CpuTensor, expected: CpuTensor) !void {
            std.debug.assert(predicted.dims.eql(&.{barcode_size}));
            std.debug.assert(expected.dims.eql(&.{barcode_size}));

            var rgba: [barcode_size]TexColor = undefined;

            for (predicted.buf, 0..) |v_in, i| {
                const v = 1.0 / (1.0 + std.math.exp(-v_in));
                rgba[i] = .grey(v);
            }
            self.predicted.setData(&rgba);

            for (expected.buf, 0..) |v, i| {
                rgba[i] = .grey(v);
            }
            self.actual.setData(&rgba);

            for (0..barcode_size) |i| {
                const predicted_true = predicted.buf[i] > 0;
                const expected_true = expected.buf[i] > 0.5;
                const matches = predicted_true == expected_true;
                rgba[i] = if (matches) .green else .red;
            }
            self.comparison.setData(&rgba);
        }
    };
}

pub fn makeComparisonView(comptime Action: type, widget_factory: sphtud.ui.widget_factory.WidgetFactory(Action)) !ComparisonView(Action) {
    const layout = try widget_factory.makeLayout();
    const predicted = try BarRetriever.init(widget_factory.alloc.gl);
    const actual = try BarRetriever.init(widget_factory.alloc.gl);
    const comparison = try BarRetriever.init(widget_factory.alloc.gl);

    const pairs = &.{
        .{ predicted, "predicted" },
        .{ actual, "actual" },
        .{ comparison, "comparison" },
    };

    inline for (pairs) |pair| {
        try layout.pushWidget(try widget_factory.makeLabel(pair[1]));
        try layout.pushWidget(try widget_factory.makeBox(
            try widget_factory.makeThumbnail(pair[0]),
            .{ .width = 0, .height = 20 },
            .fill_width,
        ));
    }

    return .{
        .predicted = predicted,
        .actual = actual,
        .comparison = comparison,
        .widget = layout.asWidget(),
    };
}
