const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const sphtud = b.dependency("sphtud", .{
        .with_gl = true,
        .with_glfw = true,
    });
    const exe = b.addExecutable(.{
        .name = "sphmelly",
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    exe.root_module.addImport("sphtud", sphtud.module("sphtud"));
    exe.linkSystemLibrary("OpenCL");
    exe.linkLibC();

    const imagegen = b.addExecutable(.{
        .name = "imagegen",
        .root_source_file = b.path("src/imagegen.zig"),
        .target = target,
        .optimize = optimize,
    });
    imagegen.root_module.addImport("sphtud", sphtud.module("sphtud"));
    imagegen.linkSystemLibrary("OpenCL");
    imagegen.linkLibC();

    const test_exe = b.addTest(.{
        .name = "test",
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    test_exe.root_module.addImport("sphtud", sphtud.module("sphtud"));
    test_exe.linkSystemLibrary("OpenCL");
    test_exe.linkLibC();

    b.installArtifact(test_exe);
    b.installArtifact(exe);
    b.installArtifact(imagegen);
}
