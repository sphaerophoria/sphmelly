const std = @import("std");
const nn = @import("../nn.zig");
const cl = @import("../cl.zig");

const Spec = struct {
    layer_checkpoints: [][]Buffer,
    weights: []const u8,
};

pub const Buffer = struct {
    key: []const u8,
    dims: []const u32,
    byte_offs: usize,
};

pub fn loadCheckpoint(cl_alloc: *cl.Alloc, executor: anytype, spec_path: []const u8) ![]nn.LayerWeights(@TypeOf(executor.*)) {
    const Executor = @TypeOf(executor.*);
    const spec_f = try std.fs.cwd().openFile(spec_path, .{});
    defer spec_f.close();

    const scratch = cl_alloc.buf_alloc.backLinear();
    const scratch_cp = scratch.checkpoint();
    defer scratch.restore(scratch_cp);

    var json_reader = std.json.reader(scratch.allocator(), spec_f.reader());
    const spec = try std.json.parseFromTokenSourceLeaky(Spec, scratch.allocator(), &json_reader, .{});

    const bin_path = try std.fmt.allocPrint(scratch.allocator(), "{s}/{s}", .{ std.fs.path.dirname(spec_path).?, spec.weights });
    const data_f = try std.fs.cwd().openFile(bin_path, .{});
    defer data_f.close();

    const ret = try cl_alloc.heap().alloc(nn.LayerWeights(Executor), spec.layer_checkpoints.len);
    for (spec.layer_checkpoints, ret) |layer_weight_def, *weight_param_out| {
        if (layer_weight_def.len == 0) {
            weight_param_out.* = &.{};
            continue;
        }

        weight_param_out.* = try cl_alloc.heap().alloc(nn.WeightParam(Executor), layer_weight_def.len);
        for (layer_weight_def, weight_param_out.*) |param_def, *param_out| {
            try data_f.seekTo(param_def.byte_offs);

            const scratch_cp2 = scratch.checkpoint();
            defer scratch.restore(scratch_cp2);

            const initial_data = try scratch.allocator().alloc(f32, totalSize(param_def.dims));
            _ = try data_f.readAll(std.mem.sliceAsBytes(initial_data));
            const tensor_load_res = try executor.createTensor(cl_alloc, initial_data, param_def.dims);
            try tensor_load_res.event.wait();
            param_out.* = .{
                .key = try cl_alloc.heap().dupe(u8, param_def.key),
                .tensor = tensor_load_res.val,
            };
        }
    }

    return ret;
}

fn totalSize(dims: []const u32) u32 {
    var out: u32 = 1;
    for (dims) |d| {
        out *= d;
    }
    return out;
}

pub fn LoadedCheckpoint(comptime Executor: type) type {
    return struct { layer_checkpoints: []?nn.WeightParam(Executor) };
}

pub fn write(cl_alloc: *cl.Alloc, executor: anytype, out_dir: std.fs.Dir, name: []const u8, layers: []const nn.Layer(@TypeOf(executor.*))) !void {
    try out_dir.makeDir(name);
    var checkpoint = try out_dir.openDir(name, .{});
    defer checkpoint.close();

    const bin_data = try checkpoint.createFile("data.bin", .{});
    var bin_data_buf_writer = std.io.bufferedWriter(bin_data.writer());

    var counting_writer = std.io.countingWriter(bin_data_buf_writer.writer());

    const data_writer = counting_writer.writer();

    const layer_checkpoints = try cl_alloc.heap().alloc([]Buffer, layers.len);
    for (layers, layer_checkpoints) |layer, *layer_out| {
        const weights = try layer.exportWeights(cl_alloc.heap());

        if (weights.len == 0) {
            layer_out.* = &.{};
            continue;
        }
        layer_out.* = try cl_alloc.heap().alloc(Buffer, weights.len);
        for (weights, layer_out.*) |param, *param_out| {
            param_out.* = .{
                .key = param.key,
                .dims = param.tensor.dims.inner,
                .byte_offs = @intCast(counting_writer.bytes_written),
            };

            const cp = cl_alloc.checkpoint();
            defer cl_alloc.reset(cp);

            const cpu_data = try executor.toCpu(cl_alloc.heap(), cl_alloc, param.tensor);
            try data_writer.writeAll(std.mem.sliceAsBytes(cpu_data));
        }
    }

    try bin_data_buf_writer.flush();

    const spec_file = try checkpoint.createFile("spec.json", .{});
    defer spec_file.close();

    try std.json.stringify(Spec{
        .layer_checkpoints = layer_checkpoints,
        .weights = "data.bin",
    }, .{ .whitespace = .indent_2 }, spec_file.writer());
}
