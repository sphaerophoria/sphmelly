const std = @import("std");
const BarcodeGen = @import("BarcodeGen.zig");
const nn = @import("nn.zig");
const Config = @This();

data: struct {
    batch_size: u32,
    label_in_frame: bool,
    img_size: u32,
    rand_params: BarcodeGen.RandomizationParams,
    enable_backgrounds: bool,
},
log_freq: u32,
val_freq: u32,
val_size: u32,
checkpoint_freq: u32,
train_target: TrainTarget,
loss_multipliers: []f32,
network: nn.Config,

pub const TrainTarget = enum {
    bbox,
    bars,
};

pub fn parse(leaky: std.mem.Allocator, path: []const u8) !Config {
    const f = try std.fs.cwd().openFile(path, .{});
    defer f.close();

    var json_reader = std.json.reader(leaky, f.reader());
    const ret = try std.json.parseFromTokenSourceLeaky(Config, leaky, &json_reader, .{});

    if (ret.val_freq % ret.log_freq != 0) {
        return error.InvalidValFreq;
    }

    return ret;
}
