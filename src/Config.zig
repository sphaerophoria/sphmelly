const std = @import("std");
const BarcodeGen = @import("BarcodeGen.zig");
const nn = @import("nn.zig");
const Config = @This();

data: struct {
    batch_size: u32,
    label_in_frame: bool,
    label_iou: bool,
    img_size: u32,
    rand_params: BarcodeGen.RandomizationParams,
    val_rand_params: BarcodeGen.RandomizationParams,
    enable_backgrounds: bool,
},
log_freq: u32,
val_freq: u32,
val_size: u32,
checkpoint_freq: u32,
train_target: TrainTarget,
loss_multipliers: []f32,
disable_bbox_loss_if_out_of_frame: bool = false,
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

    if (ret.disable_bbox_loss_if_out_of_frame and !ret.data.label_in_frame) {
        return error.InvalidBboxLabel;
    }

    return ret;
}
