const std = @import("std");

pub const Executor = @import("math/Executor.zig");
pub const TracingExecutor = @import("math/TracingExecutor.zig");

pub const TensorDims = struct {
    inner: []u32,

    pub fn init(alloc: std.mem.Allocator, vals: anytype) !TensorDims {
        return switch (@TypeOf(vals)) {
            TensorDims => try vals.clone(alloc),
            else => .{ .inner = try alloc.dupe(u32, vals) },
        };
    }

    pub fn initEmpty(alloc: std.mem.Allocator, num_dims: usize) !TensorDims {
        return .{
            .inner = try alloc.alloc(u32, num_dims),
        };
    }

    pub fn clone(self: TensorDims, alloc: std.mem.Allocator) !TensorDims {
        return .{
            .inner = try alloc.dupe(u32, self.inner),
        };
    }

    pub fn get(self: TensorDims, idx: usize) u32 {
        return self.inner[idx];
    }

    pub fn getPtr(self: TensorDims, idx: usize) *u32 {
        return &self.inner[idx];
    }

    pub fn len(self: TensorDims) usize {
        return self.inner.len;
    }

    pub fn stride(self: TensorDims, dim: usize) u32 {
        var res: u32 = 1;
        for (self.inner[0..dim]) |v| {
            res *= v;
        }
        return res;
    }

    pub fn numElems(self: TensorDims) u32 {
        var ret: u32 = 1;
        for (self.inner) |d| {
            ret *= d;
        }
        return ret;
    }

    pub fn byteSize(self: TensorDims) u32 {
        return self.numElems() * @sizeOf(f32);
    }

    pub fn eql(self: TensorDims, other: TensorDims) bool {
        if (self.inner.len != other.inner.len) return false;
        for (self.inner, other.inner) |a, b| {
            if (a != b) return false;
        }

        return true;
    }

    pub fn format(self: TensorDims, comptime _: []const u8, _: std.fmt.FormatOptions, writer: anytype) !void {
        try writer.print("{d}", .{self.inner[0]});
        var it: usize = 1;
        while (it < self.inner.len) {
            defer it += 1;

            try writer.print(", {d}", .{self.inner[it]});
        }
    }
};

pub fn Tensor(comptime Store: type) type {
    return struct {
        buf: Store,
        dims: TensorDims,
    };
}
