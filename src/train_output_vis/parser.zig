const sphtud = @import("sphtud");
const std = @import("std");

pub const KeyVal = struct {
    key: []const u8,
    val: f32,
};

pub const Step = struct {
    iter: usize,
    time_ns: u64,
    img: usize,
};

pub const KeyId = struct {
    inner: u32,

    fn init(val: anytype) !KeyId {
        const truncated = std.math.cast(u32, val) orelse return error.TooManyKeys;
        return .{ .inner = truncated };
    }
};

pub const StepId = struct {
    inner: u32,

    fn init(val: anytype) !StepId {
        const truncated = std.math.cast(u32, val) orelse return error.TooManySteps;
        return .{ .inner = truncated };
    }
};

pub const KeyData = struct {
    steps: sphtud.util.RuntimeSegmentedList(StepId),
    vals: sphtud.util.RuntimeSegmentedList(f32),
    max_val: f32,

    fn init(alloc: std.mem.Allocator, expected_steps: usize, max_steps: usize) !KeyData {
        return .{
            .steps = try .init(
                alloc,
                alloc,
                expected_steps,
                max_steps,
            ),
            .vals = try .init(
                alloc,
                alloc,
                expected_steps,
                max_steps,
            ),
            .max_val = 0,
        };
    }

    pub const Iter = struct {
        parent: *const KeyData,
        steps: *const Steps,
        idx: usize,

        const Item = struct {
            step: Step,
            val: f32,
        };

        fn next(self: *Iter) ?Item {
            if (self.idx >= self.parent.steps.len) return null;
            defer self.idx += 1;

            const step_id = self.parent.steps.get(self.idx);
            const step = self.steps.get(step_id);

            // Iterating with list iter is better, but whatever
            const val = self.parent.vals.get(self.idx);

            return .{
                .step = step,
                .val = val,
            };
        }
    };

    pub fn iter(self: *KeyData, steps: *const Steps) Iter {
        return .{
            .parent = self,
            .steps = steps,
            .idx = 0,
        };
    }
};

pub const Steps = struct {
    inner: sphtud.util.RuntimeSegmentedList(Step),

    pub fn get(self: Steps, id: StepId) Step {
        return self.inner.get(id.inner);
    }

    pub fn last(self: Steps) Step {
        return self.inner.get(self.inner.len - 1);
    }

    pub fn len(self: Steps) usize {
        return self.inner.len;
    }
};

pub const Keys = struct {
    alloc: std.mem.Allocator,
    inner: sphtud.util.RuntimeSegmentedList([]const u8),
    lookup: std.StringHashMapUnmanaged(KeyId),

    pub fn init(alloc: std.mem.Allocator) !Keys {
        return .{
            .alloc = alloc,
            .inner = try .init(alloc, alloc, 16, 1000),
            .lookup = .{},
        };
    }

    pub fn get(self: Keys, id: KeyId) []const u8 {
        return self.inner.get(id.inner);
    }

    pub fn resolveId(self: *Keys, name: []const u8) !KeyId {
        const gop = try self.lookup.getOrPut(self.alloc, name);
        if (!gop.found_existing) {
            const duped_name = try self.alloc.dupe(u8, name);
            try self.inner.append(duped_name);
            gop.value_ptr.* = try .init(self.inner.len - 1);
            gop.key_ptr.* = duped_name;
        }
        return gop.value_ptr.*;
    }

    const Iter = struct {
        idx: u32,
        max: u32,

        pub fn next(self: *Iter) ?KeyId {
            if (self.idx >= self.max) return null;
            defer self.idx += 1;

            return .{ .inner = self.idx };
        }
    };

    pub fn iter(self: *Keys) Iter {
        return .{
            .idx = 0,
            .max = @intCast(self.inner.len),
        };
    }
};

pub const TrainingLog = struct {
    source: []const u8,
    steps: Steps,
    keyed_data: std.AutoHashMapUnmanaged(KeyId, KeyData),
};

fn parseStep(data: []const u8) !Step {
    var data_it = std.mem.splitScalar(u8, data, ',');

    const iter_s = data_it.next() orelse return error.InvalidData;
    const time_ns_s = data_it.next() orelse return error.InvalidData;
    const img_s = data_it.next() orelse return error.InvalidData;

    return .{
        .iter = try std.fmt.parseInt(usize, iter_s, 0),
        .time_ns = try std.fmt.parseInt(u64, time_ns_s, 0),
        .img = try std.fmt.parseInt(usize, img_s, 0),
    };
}

const TrainingLogLine = union(enum) {
    step: Step,
    kv: KeyVal,
};

fn parseTrainingLogLine(line: []const u8) !TrainingLogLine {
    const type_end = std.mem.indexOfScalar(u8, line, ',') orelse return error.InvalidLine;

    const type_name = line[0..type_end];
    const data_start = type_end + 1;
    if (line.len <= data_start) {
        return error.InvalidLine;
    }

    const data = line[data_start..];

    if (std.mem.eql(u8, "step", type_name)) {
        return .{
            .step = try parseStep(data),
        };
    } else {
        return .{
            .kv = .{
                .key = type_name,
                .val = try std.fmt.parseFloat(f32, data),
            },
        };
    }
}

pub fn parseData(alloc: std.mem.Allocator, scratch: sphtud.alloc.LinearAllocator, path: []const u8, keys: *Keys) !TrainingLog {
    const cp = scratch.checkpoint();
    defer scratch.restore(cp);

    const f = try std.fs.cwd().openFile(path, .{});
    defer f.close();

    var reader_buf: [4096]u8 = undefined;
    var reader = f.reader(&reader_buf);

    var steps = try sphtud.util.RuntimeSegmentedList(Step).init(
        alloc,
        alloc,
        128,
        1000000,
    );

    var keyed_data = std.AutoHashMapUnmanaged(KeyId, KeyData){};

    {
        const first_line = try reader.interface.takeDelimiterExclusive('\n');
        switch (try parseTrainingLogLine(first_line)) {
            .step => |s| try steps.append(s),
            .kv => return error.InvalidFirstLine,
        }
    }

    while (true) {
        const line = reader.interface.takeDelimiterExclusive('\n') catch |e| {
            if (e == error.EndOfStream) {
                break;
            }
            return e;
        };

        switch (try parseTrainingLogLine(line)) {
            .step => |s| try steps.append(s),
            .kv => |elem| {
                const key_id = try keys.resolveId(elem.key);

                {
                    const gop = try keyed_data.getOrPut(alloc, key_id);
                    if (!gop.found_existing) {
                        gop.value_ptr.* = try KeyData.init(alloc, 64, 1000000);
                    }

                    gop.value_ptr.max_val = @max(gop.value_ptr.max_val, elem.val);
                    try gop.value_ptr.steps.append(try .init(steps.len - 1));
                    try gop.value_ptr.vals.append(elem.val);
                }
            },
        }
    }

    return .{
        .source = path,
        .steps = .{ .inner = steps },
        .keyed_data = keyed_data,
    };
}
