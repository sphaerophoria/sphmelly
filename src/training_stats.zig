const std = @import("std");
const cl = @import("cl.zig");
const math = @import("math.zig");
const nn = @import("nn.zig");
const BarcodeGen = @import("BarcodeGen.zig");
const sphtud = @import("sphtud");

pub const BboxLosses = struct {
    total: f32,
    x: f32,
    y: f32,
    w: f32,
    h: f32,
    rx: f32,
    ry: f32,
    in_frame: ?f32,
    confidence: ?f32,
};

pub const EnabledLabels = struct {
    in_frame: bool,
    confidence: bool,
};
const BboxIndexes = struct {
    const x = 0;
    const y = 1;
    const w = 2;
    const h = 3;
    const rx = 4;
    const ry = 5;

    in_frame: ?usize,
    confidence: ?usize,

    fn labelStride(self: BboxIndexes) usize {
        var stride: usize = 6;
        if (self.in_frame != null) stride += 1;
        if (self.confidence != null) stride += 1;

        return stride;
    }

    fn resolve(labels: EnabledLabels) BboxIndexes {
        var idx: usize = ry + 1;

        var in_frame: ?usize = null;
        var confidence: ?usize = null;
        if (labels.in_frame) {
            in_frame = idx;
            idx += 1;
        }

        if (labels.confidence) {
            confidence = idx;
            idx += 1;
        }

        return .{
            .in_frame = in_frame,
            .confidence = confidence,
        };
    }
};

fn sumItems(loss: []const f32) f32 {
    var sum: f32 = 0;
    for (loss) |l| {
        sum += l;
    }
    return sum;
}

// x, y, sqrt(w), sqrt(h), cos(rot), sin(rot), in_frame, confidence
pub fn extractBboxLosses(comptime Executor: type, cl_alloc: *cl.Alloc, enabled_labels: EnabledLabels, executor: *Executor, loss: Executor.Tensor) !BboxLosses {
    const indexes = BboxIndexes.resolve(enabled_labels);

    const loss_cpu = try executor.toCpu(cl_alloc.heap(), cl_alloc, loss);

    return .{
        .x = loss_cpu[BboxIndexes.x],
        .y = loss_cpu[BboxIndexes.y],
        .w = loss_cpu[BboxIndexes.w],
        .h = loss_cpu[BboxIndexes.h],
        .rx = loss_cpu[BboxIndexes.rx],
        .ry = loss_cpu[BboxIndexes.ry],
        .in_frame = if (indexes.in_frame) |i| loss_cpu[i] else null,
        .confidence = if (indexes.confidence) |i| loss_cpu[i] else null,
        .total = sumItems(loss_cpu),
    };
}

pub const BboxValidationData = struct {
    box_iou: SegmentedStats,
    dilated_box_iou: SegmentedStats,
    x_err: SegmentedStats,
    y_err: SegmentedStats,
    width_err: SegmentedStats,
    height_err: SegmentedStats,
    rx_err: SegmentedStats,
    ry_err: SegmentedStats,
    confidence_err: ?SegmentedStats,
    in_frame_err: ?SegmentedStats,
};

pub const SegmentedStats = struct {
    average: f32,
    at_90: f32,
    at_99: f32,
};

fn calcSegmented(vals: []f32, sort_fn: fn (f32, f32) bool) SegmentedStats {
    std.mem.sort(f32, vals, {}, struct {
        fn f(_: void, a: f32, b: f32) bool {
            return sort_fn(a, b);
        }
    }.f);

    const idx_90 = vals.len * 9 / 10;
    const idx_99 = vals.len * 99 / 100;

    return .{
        .at_90 = vals[idx_90],
        .at_99 = vals[idx_99],
        .average = sumItems(vals) / asf32(vals.len),
    };
}

fn asf32(in: anytype) f32 {
    return @floatFromInt(in);
}

fn lessThan(a: f32, b: f32) bool {
    return a < b;
}

fn greaterThan(a: f32, b: f32) bool {
    return a > b;
}

const ErrCalculator = struct {
    scratch: sphtud.alloc.LinearAllocator,
    results: []const f32,
    expected: []const f32,
    stride: usize,

    fn linearErr(self: ErrCalculator, index: usize) !SegmentedStats {
        const cp = self.scratch.checkpoint();
        defer self.scratch.restore(cp);

        const errs = try self.allocOutput();
        for (errs, 0..) |*out, i| {
            out.* = @abs(self.results[i * self.stride + index] -
                self.expected[i * self.stride + index]);
        }
        return calcSegmented(errs, lessThan);
    }

    fn squaredErr(self: ErrCalculator, index: usize) !SegmentedStats {
        const cp = self.scratch.checkpoint();
        defer self.scratch.restore(cp);

        const errs = try self.allocOutput();
        for (errs, 0..) |*out, i| {
            const a = self.results[i * self.stride + index];
            const b = self.expected[i * self.stride + index];
            out.* = @abs(a * a - b * b);
        }
        return calcSegmented(errs, lessThan);
    }

    fn allocOutput(self: ErrCalculator) ![]f32 {
        const num_items = self.results.len / self.stride;
        return try self.scratch.allocator().alloc(f32, num_items);
    }
};

fn calcSegmentedIous(cl_alloc: *cl.Alloc, barcode_gen: *const BarcodeGen, executor: math.Executor, predictions: math.Executor.Tensor, expected: math.Executor.Tensor, dilation: f32) !SegmentedStats {
    const cp = cl_alloc.checkpoint();
    defer cl_alloc.reset(cp);

    const boxes = try barcode_gen.boxPredictionToBox(cl_alloc, predictions, dilation);
    const expected_boxes = try barcode_gen.boxPredictionToBox(cl_alloc, expected, dilation);
    const ious = try executor.calcIou(cl_alloc, boxes, expected_boxes);
    const ious_cpu = try executor.toCpu(cl_alloc.heap(), cl_alloc, ious);

    return calcSegmented(ious_cpu, greaterThan);
}

pub fn calcBboxValidationData(
    cl_alloc: *cl.Alloc,
    enabled_labels: EnabledLabels,
    barcode_gen: *const BarcodeGen,
    executor: math.Executor,
    val_results: math.Executor.Tensor,
    expected: math.Executor.Tensor,
) !BboxValidationData {
    const val_results_cpu = try executor.toCpu(cl_alloc.heap(), cl_alloc, val_results);
    const expected_cpu = try executor.toCpu(cl_alloc.heap(), cl_alloc, expected);

    const indexes = BboxIndexes.resolve(enabled_labels);
    const label_stride = indexes.labelStride();

    const err_calculator = ErrCalculator{
        .scratch = cl_alloc.buf_alloc.linear(),
        .results = val_results_cpu,
        .expected = expected_cpu,
        .stride = label_stride,
    };

    return .{
        .box_iou = try calcSegmentedIous(cl_alloc, barcode_gen, executor, val_results, expected, 1.0),
        .dilated_box_iou = try calcSegmentedIous(cl_alloc, barcode_gen, executor, val_results, expected, 1.1),
        .x_err = try err_calculator.linearErr(BboxIndexes.x),
        .y_err = try err_calculator.linearErr(BboxIndexes.y),
        .rx_err = try err_calculator.linearErr(BboxIndexes.rx),
        .ry_err = try err_calculator.linearErr(BboxIndexes.ry),

        // Width and height are calculated as sqrt, so square to get back to real life
        .width_err = try err_calculator.squaredErr(BboxIndexes.rx),
        .height_err = try err_calculator.squaredErr(BboxIndexes.ry),
        .confidence_err = if (indexes.confidence) |i| try err_calculator.linearErr(i) else null,
        .in_frame_err = if (indexes.in_frame) |i| try err_calculator.linearErr(i) else null,
    };
}
