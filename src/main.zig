const std = @import("std");
const sphtud = @import("sphtud");
const cl = @import("cl.zig");
const math = @import("math.zig");

fn OpenClLayerGen(comptime Executor: type) type {
    return struct {
        cl_alloc: *cl.Alloc,
        math_executor: *Executor,

        const Self = @This();

        const FullyConnected = struct {
            math_executor: *Executor,
            weights: Executor.Tensor,
            biases: Executor.Tensor,

            fn init(scratch: *sphtud.alloc.BufAllocator, cl_alloc: *cl.Alloc, math_executor: *Executor, inputs: u32, outputs: u32) !FullyConnected {
                const cp = scratch.checkpoint();
                defer scratch.restore(cp);

                // 2 inputs, 1 output
                const initial_weights = try scratch.allocator().alloc(f32, inputs * outputs);
                @memset(initial_weights, 0);

                const weights = try math_executor.createTensor(cl_alloc, initial_weights, &.{ inputs, outputs });
                const biases = try math_executor.createTensor(cl_alloc, initial_weights[0..outputs], &.{outputs});

                try weights.event.wait();
                try biases.event.wait();

                return .{
                    .math_executor = math_executor,
                    .weights = weights.val,
                    .biases = biases.val,
                };
            }

            fn execute(self: FullyConnected, cl_alloc: *cl.Alloc, input: Executor.Tensor) !Executor.Tensor {
                const mul_res = try self.math_executor.matmul(cl_alloc, self.weights, input);
                const with_bias = try self.math_executor.addSplatHorizontal(cl_alloc, self.biases, mul_res);
                return self.math_executor.sigmoid(cl_alloc, with_bias);
            }
        };

        fn fullyConnected(self: Self, scratch: *sphtud.alloc.BufAllocator, inputs: u32, outputs: u32) !FullyConnected {
            return FullyConnected.init(
                scratch,
                self.cl_alloc,
                self.math_executor,
                inputs,
                outputs,
            );
        }
    };
}

const Optimizer = struct {
    weights: sphtud.util.RuntimeBoundedArray(math.TracingExecutor.Tensor),
    traced: *math.TracingExecutor,
    executor: math.Executor,

    fn registerWeights(self: *Optimizer, weights: math.TracingExecutor.Tensor) !void {
        try self.weights.append(weights);
    }

    fn optimize(self: Optimizer, cl_alloc: *cl.Alloc, gradients: math.TracingExecutor.Gradients) !void {
        const lr = 0.001;

        for (self.weights.items) |item| {
            const item_grad = gradients.get(item.buf);
            const tensor = self.traced.getClTensor(item.buf);
            const weighted_grad = try self.executor.mulScalar(cl_alloc, item_grad, -lr);
            try self.executor.addAssign(tensor, weighted_grad);
        }
    }
};

pub fn main() !void {
    var arena = sphtud.alloc.BufAllocator.init(try std.heap.page_allocator.alloc(u8, 10 * 1024 * 1024));

    const cl_executor = try cl.Executor.init();
    defer cl_executor.deinit();

    var cl_alloc = try cl.Alloc.init(try arena.allocator().alloc(u8, 1 * 1024 * 1024));
    defer cl_alloc.deinit();

    var rng = std.Random.DefaultPrng.init(0);
    var rand = rng.random();

    const math_executor = try math.Executor.init(&cl_alloc, cl_executor);
    var tracing_executor = try math.TracingExecutor.init(math_executor, arena.allocator(), 100);

    const layer_gen = OpenClLayerGen(math.TracingExecutor){
        .math_executor = &tracing_executor,
        .cl_alloc = &cl_alloc,
    };

    var optimizer = Optimizer{
        .executor = math_executor,
        .traced = &tracing_executor,
        .weights = try .init(arena.allocator(), 100),
    };

    const layer = try layer_gen.fullyConnected(&arena, 2, 1);
    try optimizer.registerWeights(layer.weights);
    try optimizer.registerWeights(layer.biases);

    const checkpoint = arena.checkpoint();
    const math_checkpoint = tracing_executor.checkpoint();
    const cl_alloc_checkpoint = cl_alloc.checkpoint();

    while (true) {
        arena.restore(checkpoint);
        tracing_executor.restore(math_checkpoint);
        cl_alloc.reset(cl_alloc_checkpoint);

        const batch_size = 1000;
        const batch_cpu = try arena.allocator().alloc(f32, batch_size * 2);
        for (batch_cpu) |*elem| {
            elem.* = rand.float(f32);
        }

        const batch = try tracing_executor.createTensorUntracked(&cl_alloc, batch_cpu, &.{ batch_size, 2 });
        const results_a = try layer.execute(&cl_alloc, batch);
        const results = try tracing_executor.reshape(&cl_alloc, results_a, &.{batch_size});

        const res_cpu = try tracing_executor.toCpu(arena.allocator(), &cl_alloc, results);

        const expected = try tracing_executor.gt(&cl_alloc, batch, 1);
        const loss = try tracing_executor.squaredErr(&cl_alloc, expected, results);

        const expected_cpu = try tracing_executor.toCpu(arena.allocator(), &cl_alloc, expected);
        //tracing_executor.printBackwards(loss.buf, 0);

        const gradients = try tracing_executor.backprop(&cl_alloc, loss.buf);

        var correct: usize = 0;

        for (0..res_cpu.len) |i| {
            if ((res_cpu[i] > 0.5) == (expected_cpu[i] > 0.5)) correct += 1;
        }

        try optimizer.optimize(&cl_alloc, gradients);
        try cl_executor.finish();

        std.debug.print("{any}/{d} correct\n", .{ correct, batch_size });
    }
}

test {
    _ = std.testing.refAllDeclsRecursive(@This());
}
