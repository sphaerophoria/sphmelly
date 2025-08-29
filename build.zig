const std = @import("std");

fn generateConvTestData(b: *std.Build) std.Build.LazyPath {
    const cmd = b.addSystemCommand(&.{"python3"});
    cmd.addFileArg(b.path("src/math/Executor/generate_conv_test.py"));

    return cmd.captureStdOut();
}

fn addCommonDependencies(b: *std.Build, exe: *std.Build.Step.Compile, sphtud_mod: *std.Build.Module) void {
    exe.root_module.addImport("sphtud", sphtud_mod);
    exe.root_module.addCSourceFile(.{
        .file = b.path("src/stb_image.c"),
    });
    exe.root_module.addIncludePath(b.path("src"));
    exe.linkSystemLibrary("OpenCL");
    exe.linkLibC();
}

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const sphtud = b.dependency("sphtud", .{
        .with_gl = true,
        .with_glfw = true,
    });
    const sphtud_mod = sphtud.module("sphtud");

    const exe = b.addExecutable(.{
        .name = "sphmelly",
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    addCommonDependencies(b, exe, sphtud_mod);

    const imagegen = b.addExecutable(.{
        .name = "imagegen",
        .root_source_file = b.path("src/imagegen.zig"),
        .target = target,
        .optimize = optimize,
    });
    addCommonDependencies(b, imagegen, sphtud_mod);

    const conv_test_data = generateConvTestData(b);

    const test_exe = b.addTest(.{
        .name = "test",
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    addCommonDependencies(b, test_exe, sphtud_mod);
    test_exe.root_module.addAnonymousImport("conv_test_data", .{
        .root_source_file = conv_test_data,
    });

    b.installArtifact(test_exe);
    b.installArtifact(exe);
    b.installArtifact(imagegen);
}
