const std = @import("std");

layers: []const LayerDef,

const Config = @This();

pub const Initializer = enum {
    he,
    zero,
};

pub const LayerDef = union(enum) {
    conv: struct {
        Initializer,
        u32,
        u32,
        u32,
        u32,
    },
    relu,
    maxpool: u32,
    reshape: []const u32,
    fully_connected: struct {
        Initializer,
        Initializer,
        u32,
        u32,
    },
};

pub fn printExample() !void {
    const example = Config{
        .layers = &.{
            .{
                .conv = .{ .he, 1, 2, 3, 4 },
            },
            .relu,
            .{ .maxpool = 2 },
            .{ .reshape = &.{ 3, 4, 7 } },
            .{ .fully_connected = .{ .he, .zero, 128, 256 } },
        },
    };

    const stdout = std.io.getStdOut();
    try std.json.stringify(example, .{ .whitespace = .indent_2 }, stdout.writer());
    try stdout.writeAll("\n");
}

pub fn parse(leaky: std.mem.Allocator, path: []const u8) !Config {
    const f = try std.fs.cwd().openFile(path, .{});
    defer f.close();

    var json_reader = std.json.reader(leaky, f.reader());
    return std.json.parseFromTokenSourceLeaky(Config, leaky, &json_reader, .{});
}
