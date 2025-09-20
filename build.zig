const std = @import("std");

fn generateConvTestData(b: *std.Build) std.Build.LazyPath {
    const cmd = b.addSystemCommand(&.{"python3"});
    cmd.addFileArg(b.path("src/math/Executor/generate_conv_test.py"));

    return cmd.captureStdOut();
}

fn generateBceTestData(b: *std.Build) std.Build.LazyPath {
    const cmd = b.addSystemCommand(&.{"python3"});
    cmd.addFileArg(b.path("src/math/Executor/generate_bce_logits_test.py"));

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

fn install(b: *std.Build, check: bool, exe: *std.Build.Step.Compile) void {
    if (check) {
        b.getInstallStep().dependOn(&exe.step);
    } else {
        b.installArtifact(exe);
    }
}

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const check = b.option(bool, "check", "check") orelse false;

    const sphtud = b.dependency("sphtud", .{
        .with_gl = true,
        .with_glfw = true,
    });
    const sphtud_mod = sphtud.module("sphtud");

    const exe = b.addExecutable(.{
        .name = "sphmelly",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    addCommonDependencies(b, exe, sphtud_mod);

    const train_output_vis = b.addExecutable(.{
        .name = "train_output_vis",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/train_output_vis.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    addCommonDependencies(b, train_output_vis, sphtud_mod);

    const iou_demo = b.addExecutable(.{
        .name = "iou_demo",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/iou_demo.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    addCommonDependencies(b, iou_demo, sphtud_mod);

    const imagegen = b.addExecutable(.{
        .name = "imagegen",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/imagegen.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    addCommonDependencies(b, imagegen, sphtud_mod);

    const scanner = b.addExecutable(.{
        .name = "scanner",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/scanner.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    addCommonDependencies(b, scanner, sphtud_mod);

    const gen_validation_stats = b.addExecutable(.{
        .name = "gen_validation_stats",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/gen_validation_stats.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    addCommonDependencies(b, gen_validation_stats, sphtud_mod);

    const conv_test_data = generateConvTestData(b);
    const bce_test_data = generateBceTestData(b);

    const test_exe = b.addTest(.{
        .name = "test",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    addCommonDependencies(b, test_exe, sphtud_mod);
    test_exe.root_module.addAnonymousImport("conv_test_data", .{
        .root_source_file = conv_test_data,
    });
    test_exe.root_module.addAnonymousImport("bce_test_data", .{
        .root_source_file = bce_test_data,
    });

    install(b, check, test_exe);
    install(b, check, exe);
    install(b, check, imagegen);
    install(b, check, train_output_vis);
    install(b, check, iou_demo);
    install(b, check, scanner);
    install(b, check, gen_validation_stats);
}
