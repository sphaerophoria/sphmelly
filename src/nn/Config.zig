const std = @import("std");

lr: f32,
layers: []const LayerDef,

const Config = @This();

pub const Initializer = enum {
    he,
    zero,
};

pub const LayerDef = union(enum) {
    conv: struct { Initializer, u32, u32, u32, u32 },
    relu: f32,
    maxpool: [2]u32,
    reshape: []const u32,
    fully_connected: struct { Initializer, Initializer, u32, u32 },
};

pub fn printExample() !void {
    const example = Config{
        .lr = 0.001,
        .layers = &.{
            .{
                .conv = .{ .he, 1, 2, 3, 4 },
            },
            .{ .relu = 0.1 },
            .{ .maxpool = 2 },
            .{ .reshape = &.{ 3, 4, 7 } },
            .{ .fully_connected = .{ .he, .zero, 128, 256 } },
        },
    };

    const stdout = std.io.getStdOut();
    try std.json.stringify(example, .{ .whitespace = .indent_2 }, stdout.writer());
    try stdout.writeAll("\n");
}
