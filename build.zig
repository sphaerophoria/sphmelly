const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const sphtud = b.dependency("sphtud", .{});
    const exe = b.addExecutable(.{
        .name = "sphmelly",
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    exe.root_module.addImport("sphtud", sphtud.module("sphtud"));

    b.installArtifact(exe);
}
