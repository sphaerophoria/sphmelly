const std = @import("std");
const sphtud = @import("sphtud");
const cl = @import("cl.zig");
const math = @import("math.zig");

fn isTracing(comptime T: type) bool {
    switch (T) {
        math.TracingExecutor => return true,
        math.Executor => return false,
        else => @compileError("Unknown executor"),
    }
}

fn IfTracing(comptime T: type, comptime U: type) type {
    if (isTracing(T)) return U;
    return void;
}

fn assignIfTracing(comptime T: type, val: anytype) IfTracing(T, @TypeOf(val)) {
    if (isTracing(T)) return val;
}

pub fn Layer(comptime Executor: type) type {
    return struct {
        vtable: *const VTable,
        name: []const u8,
        ctx: ?*anyopaque,

        const Self = @This();

        pub const VTable = struct {
            getWeights: *const fn (ctx: ?*anyopaque, param_id: usize) anyerror!?Executor.TensorSlice,
            execute: *const fn (ctx: ?*anyopaque, cl_alloc: *cl.Alloc, executor: *Executor, input: Executor.Tensor) anyerror!Executor.Tensor,
            registerWeights: if (isTracing(Executor))
                *const fn (ctx: ?*anyopaque, optimizer: *Optimizer) anyerror!void
            else
                void,
        };

        pub fn getWeights(self: Self, param_id: usize) !?Executor.TensorSlice {
            return try self.vtable.getWeights(self.ctx, param_id);
        }

        pub fn execute(self: Self, alloc: *cl.Alloc, executor: *Executor, input: Executor.Tensor) !Executor.Tensor {
            return try self.vtable.execute(self.ctx, alloc, executor, input);
        }

        pub fn registerWeights(self: Self, optimizer: *Optimizer) !void {
            if (@TypeOf(self.vtable.registerWeights) != void) {
                try self.vtable.registerWeights(self.ctx, optimizer);
            }
        }
    };
}

pub fn HeInitializer(comptime Executor: type) type {
    return struct {
        executor: *Executor,
        rand_source: *math.RandSource,

        pub fn generate(self: @This(), cl_alloc: *cl.Alloc, fan_in: usize, dims: anytype) !Executor.Tensor {
            const stddev = std.math.sqrt(2 / @as(f32, @floatFromInt(fan_in)));
            return self.executor.randGaussian(cl_alloc, dims, stddev, self.rand_source);
        }
    };
}

pub fn ZeroInitializer(comptime Executor: type) type {
    return struct {
        executor: *Executor,

        pub fn generate(self: @This(), cl_alloc: *cl.Alloc, _: usize, dims: anytype) !Executor.Tensor {
            return try self.executor.createTensorFilled(cl_alloc, dims, 0.0);
        }
    };
}

pub fn FullyConnected(comptime Executor: type) type {
    return struct {
        weights: Executor.Tensor,
        biases: Executor.Tensor,
        name: []const u8,

        const Self = @This();

        pub fn init(cl_alloc: *cl.Alloc, weights_initializer: anytype, bias_initializer: anytype, inputs: u32, outputs: u32) !Self {
            const weights = try weights_initializer.generate(cl_alloc, inputs, &.{ inputs, outputs });
            const biases = try bias_initializer.generate(cl_alloc, inputs, &.{ 1, outputs });

            return .{
                .weights = weights,
                .name = try std.fmt.allocPrint(cl_alloc.heap(), "fc ({d} -> {d})", .{ inputs, outputs }),
                .biases = biases,
            };
        }

        const layer_vtable = Layer(Executor).VTable{
            .getWeights = getWeights,
            .execute = execute,
            .registerWeights = assignIfTracing(Executor, registerWeights),
        };

        fn getWeights(ctx: ?*anyopaque, param_id: usize) !?Executor.TensorSlice {
            const self: *const Self = @ptrCast(@alignCast(ctx));

            switch (param_id) {
                0 => return self.weights.asSlice(),
                1 => return self.biases.asSlice(),
                else => return null,
            }
        }

        pub fn layer(self: *Self) Layer(Executor) {
            return .{
                .vtable = &layer_vtable,
                .name = self.name,
                .ctx = self,
            };
        }

        fn execute(ctx: ?*anyopaque, cl_alloc: *cl.Alloc, executor: *Executor, input: Executor.Tensor) !Executor.Tensor {
            const self: *Self = @ptrCast(@alignCast(ctx));

            std.debug.assert(input.dims.len() == 2);
            const input_reshaped = try executor.reshape(cl_alloc, input, &.{ 1, input.dims.get(0), input.dims.get(1) });
            const mul_res = try executor.matmul(cl_alloc, self.weights, input_reshaped);
            const w_bias = try executor.addSplatOuter(cl_alloc, self.biases, mul_res);
            return try executor.reshape(cl_alloc, w_bias, &.{ self.weights.dims.get(1), input.dims.get(1) });
        }

        fn registerWeights(ctx: ?*anyopaque, optimizer: *Optimizer) !void {
            const self: *Self = @ptrCast(@alignCast(ctx));

            try optimizer.registerWeights(self.weights);
            try optimizer.registerWeights(self.biases);
        }
    };
}

pub fn Conv(comptime Executor: type) type {
    return struct {
        name: []const u8,
        kernel: Executor.Tensor,

        const Self = @This();

        pub fn init(cl_alloc: *cl.Alloc, weights_initializer: anytype, w: u32, h: u32, in_c: u32, out_c: u32) !Self {
            const fan_in = w * h * in_c;
            const kernel = try weights_initializer.generate(cl_alloc, fan_in, &.{ w, h, in_c, out_c });

            return .{
                .name = try std.fmt.allocPrint(cl_alloc.heap(), "conv {d}x{d}x{d}x{d}", .{ w, h, in_c, out_c }),
                .kernel = kernel,
            };
        }

        const layer_vtable = Layer(Executor).VTable{
            .getWeights = getWeights,
            .execute = execute,
            .registerWeights = assignIfTracing(Executor, registerWeights),
        };

        fn getWeights(ctx: ?*anyopaque, param_id: usize) !?Executor.TensorSlice {
            const self: *Self = @ptrCast(@alignCast(ctx));
            return try self.kernel.indexOuter(param_id);
        }

        fn execute(ctx: ?*anyopaque, cl_alloc: *cl.Alloc, executor: *Executor, input: Executor.Tensor) !Executor.Tensor {
            const self: *Self = @ptrCast(@alignCast(ctx));
            return try executor.convMany(cl_alloc, input, self.kernel);
        }

        fn registerWeights(ctx: ?*anyopaque, optimizer: *Optimizer) !void {
            const self: *Self = @ptrCast(@alignCast(ctx));

            try optimizer.registerWeights(self.kernel);
        }

        pub fn layer(self: *Self) Layer(Executor) {
            return .{
                .vtable = &layer_vtable,
                .name = self.name,
                .ctx = self,
            };
        }
    };
}

pub fn Sigmoid(comptime Executor: type) type {
    return struct {
        const Self = @This();

        const layer_vtable = Layer(Executor).VTable{
            .getWeights = nullGetWeights(Executor),
            .execute = execute,
            .registerWeights = assignIfTracing(Executor, nullRegisterWeights),
        };

        fn execute(_: ?*anyopaque, cl_alloc: *cl.Alloc, executor: *Executor, input: Executor.Tensor) !Executor.Tensor {
            return executor.sigmoid(cl_alloc, input);
        }
    };
}

fn nullGetWeights(comptime Executor: type) fn (?*anyopaque, usize) anyerror!?Executor.TensorSlice {
    return struct {
        fn f(_: ?*anyopaque, _: usize) anyerror!?Executor.TensorSlice {
            return null;
        }
    }.f;
}

fn nullRegisterWeights(_: ?*anyopaque, _: *Optimizer) !void {}

pub fn sigmoidLayer(comptime Executor: type) Layer(Executor) {
    return .{
        .vtable = &Sigmoid(Executor).layer_vtable,
        .name = "sigmoid",
        .ctx = null,
    };
}

pub fn Relu(comptime Executor: type) type {
    return struct {
        const Self = @This();

        const layer_vtable = Layer(Executor).VTable{
            .getWeights = nullGetWeights(Executor),
            .execute = execute,
            .registerWeights = assignIfTracing(Executor, nullRegisterWeights),
        };

        fn execute(_: ?*anyopaque, cl_alloc: *cl.Alloc, executor: *Executor, input: Executor.Tensor) !Executor.Tensor {
            return executor.relu(cl_alloc, input);
        }
    };
}

pub fn reluLayer(comptime Executor: type) Layer(Executor) {
    return .{
        .vtable = &Relu(Executor).layer_vtable,
        .name = "relu",
        .ctx = null,
    };
}

pub fn Reshape(comptime Executor: type) type {
    return struct {
        shape: []const u32,

        const Self = @This();

        pub fn init(shape: []const u32) !Self {
            return .{
                .shape = shape,
            };
        }

        const layer_vtable = Layer(Executor).VTable{
            .getWeights = nullGetWeights(Executor),
            .execute = execute,
            .registerWeights = assignIfTracing(Executor, nullRegisterWeights),
        };

        pub fn layer(self: *Self) Layer(Executor) {
            return .{
                .ctx = self,
                .name = "reshape",
                .vtable = &layer_vtable,
            };
        }

        fn execute(ctx: ?*anyopaque, cl_alloc: *cl.Alloc, executor: *Executor, input: Executor.Tensor) !Executor.Tensor {
            const self: *Self = @ptrCast(@alignCast(ctx));
            return executor.reshape(cl_alloc, input, self.shape);
        }
    };
}

pub const Optimizer = struct {
    weights: sphtud.util.RuntimeBoundedArray(math.TracingExecutor.Tensor),
    traced: *math.TracingExecutor,
    executor: math.Executor,
    lr: f32,

    pub fn registerWeights(self: *Optimizer, weights: math.TracingExecutor.Tensor) !void {
        try self.weights.append(weights);
    }

    pub fn optimize(self: Optimizer, cl_alloc: *cl.Alloc, gradients: math.TracingExecutor.Gradients, batch_size: usize) !void {
        const grad_mul = -self.lr / @as(f32, @floatFromInt(batch_size));

        for (self.weights.items) |item| {
            const item_grad = gradients.get(item.buf);
            const tensor = self.traced.getClTensor(item.buf);
            const weighted_grad = try self.executor.mulScalar(cl_alloc, item_grad, grad_mul);
            try self.executor.addAssign(tensor, weighted_grad);
        }
    }
};

pub fn Trainer(comptime Notifier: type) type {
    return struct {
        optimizer: Optimizer,
        cl_alloc: *cl.Alloc,
        tracing_executor: *math.TracingExecutor,
        layers: []const Layer(math.TracingExecutor),
        notifier: *Notifier,

        const Self = @This();

        pub fn step(self: *Self, batch: math.Executor.Tensor, correct: math.Executor.Tensor) !void {
            var results = try self.tracing_executor.appendNode(batch, .init);

            const layer_outputs = try self.cl_alloc.heap().alloc(math.TracingExecutor.Tensor, self.layers.len);

            for (self.layers, 0..) |layer, layer_id| {
                results = try layer.execute(self.cl_alloc, self.tracing_executor, results);
                layer_outputs[layer_id] = results;
            }
            try self.notifier.notifyLayerOutputs(layer_outputs);

            try self.notifier.predictionsQueued(results);

            const traced_correct = try self.tracing_executor.appendNode(correct, .init);

            const loss = try self.tracing_executor.squaredErr(self.cl_alloc, traced_correct, results);
            try self.notifier.notifyLoss(loss);

            const gradients = try self.tracing_executor.backprop(self.cl_alloc, loss.buf);
            try self.notifier.notifyGradients(self.layers, gradients);

            try self.optimizer.optimize(self.cl_alloc, gradients, batch.dims.get(batch.dims.len() - 1));
        }
    };
}
