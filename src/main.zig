const std = @import("std");
const sphtud = @import("sphtud");

const Operation = enum {
    init,
    add,
    sub,
    mul,
    sigmoid,
    pow,
};

const TrackedF32 = struct {
    val: f32,
    op: Operation,
    gradient: f32 = 0.0,
    in1: ?*TrackedF32,
    in2: ?*TrackedF32,
    in2_raw: ?f32 = null,

    fn init(alloc: std.mem.Allocator, val: f32) !*TrackedF32 {
        const ret = try alloc.create(TrackedF32);
        ret.* = .{
            .val = val,
            .op = .init,
            .in1 = null,
            .in2 = null,
        };
        return ret;
    }

    fn add(alloc: std.mem.Allocator, a: *TrackedF32, b: *TrackedF32) !*TrackedF32 {
        const ret = try alloc.create(TrackedF32);
        ret.* = .{
            .val = a.val + b.val,
            .op = .add,
            .in1 = a,
            .in2 = b,
        };
        return ret;
    }

    fn sub(alloc: std.mem.Allocator, a: *TrackedF32, b: *TrackedF32) !*TrackedF32 {
        const ret = try alloc.create(TrackedF32);
        ret.* = .{
            .val = a.val - b.val,
            .op = .sub,
            .in1 = a,
            .in2 = b,
        };
        return ret;
    }

    fn mul(alloc: std.mem.Allocator, a: *TrackedF32, b: *TrackedF32) !*TrackedF32 {
        const ret = try alloc.create(TrackedF32);
        ret.* = .{
            .val = a.val * b.val,
            .op = .mul,
            .in1 = a,
            .in2 = b,
        };
        return ret;
    }

    fn sigmoid(alloc: std.mem.Allocator, a: *TrackedF32) !*TrackedF32 {
        const ret = try alloc.create(TrackedF32);
        ret.* = .{
            .val = 1.0 / (1.0 + std.math.exp(-a.val)),
            .op = .sigmoid,
            .in1 = a,
            .in2 = null,
        };
        return ret;
    }

    fn pow(alloc: std.mem.Allocator, a: *TrackedF32, b: f32) !*TrackedF32 {
        const ret = try alloc.create(TrackedF32);
        ret.* = .{
            .val = std.math.pow(f32, a.val, b),
            .op = .pow,
            .in1 = a,
            .in2 = null,
            .in2_raw = b,
        };
        return ret;
    }

    fn backprop(self: *TrackedF32, downstream_gradient: f32) void {
        switch (self.op) {
            .init => return,
            .add => {
                const in1_grad = 1.0 * downstream_gradient;
                const in2_grad = 1.0 * downstream_gradient;

                self.in1.?.gradient += in1_grad;
                self.in2.?.gradient += in2_grad;

                self.in1.?.backprop(in1_grad);
                self.in2.?.backprop(in2_grad);
            },
            .mul => {
                const in1_grad = self.in2.?.val * downstream_gradient;
                const in2_grad = self.in1.?.val * downstream_gradient;

                self.in1.?.gradient += in1_grad;
                self.in2.?.gradient += in2_grad;

                self.in1.?.backprop(in1_grad);
                self.in2.?.backprop(in2_grad);
            },
            .pow => {
                const grad = self.in1.?.val * self.in2_raw.? * downstream_gradient;
                self.in1.?.gradient += grad;
                self.in1.?.backprop(grad);
            },
            .sub => {
                const in1_grad = 1.0 * downstream_gradient;
                const in2_grad = -1.0 * downstream_gradient;

                self.in1.?.gradient += in1_grad;
                self.in2.?.gradient += in2_grad;

                self.in1.?.backprop(in1_grad);
                self.in2.?.backprop(in2_grad);
            },
            .sigmoid => {
                const x = self.in1.?.val;
                const enx = std.math.exp(-x);
                const enxp1 = (enx + 1);

                const grad = enx / enxp1 / enxp1 * downstream_gradient;
                self.in1.?.gradient += grad;
                self.in1.?.backprop(grad);
            },
        }
    }
};

fn sigmoid(in: f32) f32 {
    return 1.0 / (1.0 + std.math.exp(-in));
}

const Network = struct {
    weights: [2]*TrackedF32,
    biases: [2]*TrackedF32,

    fn init(alloc: std.mem.Allocator, rng: std.Random) !Network {
        return .{
            .weights = .{
                try TrackedF32.init(alloc, (rng.float(f32) - 0.5) * 10.0),
                try TrackedF32.init(alloc, (rng.float(f32) - 0.5) * 10.0),
            },
            .biases = .{
                try TrackedF32.init(alloc, (rng.float(f32) - 0.5) * 10.0),
                try TrackedF32.init(alloc, (rng.float(f32) - 0.5) * 10.0),
            },
        };
    }

    fn run(self: *Network, alloc: std.mem.Allocator, a: f32, b: f32) !*TrackedF32 {
        const a_tracked = try TrackedF32.init(alloc, a);
        const b_tracked = try TrackedF32.init(alloc, b);

        const a_out = try TrackedF32.add(
            alloc,
            try TrackedF32.mul(alloc, self.weights[0], a_tracked),
            self.biases[0],
        );

        const b_out = try TrackedF32.add(
            alloc,
            try TrackedF32.mul(alloc, self.weights[1], b_tracked),
            self.biases[1],
        );

        return TrackedF32.sigmoid(
            alloc,
            try TrackedF32.add(alloc, a_out, b_out),
        );
    }

    fn optimize(self: *Network) void {
        const lr = 0.001;
        for (&self.weights) |w| {
            w.val -= w.gradient * lr;
        }

        for (&self.biases) |b| {
            b.val -= b.gradient * lr;
        }
        self.clearGrad();
    }

    fn clearGrad(self: *Network) void {
        for (&self.weights) |w| {
            w.gradient = 0;
        }

        for (&self.biases) |w| {
            w.gradient = 0;
        }
    }
};

fn printNet(input: *TrackedF32, indent_level: usize) void {
    for (0..indent_level) |_| {
        std.debug.print("\t", .{});
    }
    std.debug.print("{s}: {d} {d}\n", .{ @tagName(input.op), input.val, input.gradient });

    if (input.in1) |in1| {
        printNet(in1, indent_level + 1);
    }

    if (input.in2) |in2| {
        printNet(in2, indent_level + 1);
    }
}

pub fn main() !void {
    var arena = sphtud.alloc.BufAllocator.init(try std.heap.page_allocator.alloc(u8, 10 * 1024 * 1024));

    var rng = std.Random.DefaultPrng.init(0);
    var rand = rng.random();

    var net = try Network.init(arena.allocator(), rand);

    const checkpoint = arena.checkpoint();

    var epoch_idx: usize = 0;
    while (true) {
        defer epoch_idx += 1;

        arena.restore(checkpoint);
        var loss = try TrackedF32.init(arena.allocator(), 0);

        for (0..1000) |_| {
            const a = rand.float(f32);
            const b = rand.float(f32);
            const output = try net.run(arena.allocator(), a, b);

            const expected = if (a > b)
                try TrackedF32.init(arena.allocator(), 1)
            else
                try TrackedF32.init(arena.allocator(), 0);

            const this_loss = try TrackedF32.pow(
                arena.allocator(),
                try TrackedF32.sub(arena.allocator(), output, expected),
                2,
            );

            loss = try TrackedF32.add(arena.allocator(), this_loss, loss);
        }

        loss.backprop(1.0);

        if (epoch_idx % 1000 == 0) {
            std.debug.print("loss: {d}\n", .{loss.val});
            std.debug.print("{any}\n", .{net});
        }

        if (std.math.isNan(loss.val)) {
            net.clearGrad();
            continue;
        }

        net.optimize();
    }
}
