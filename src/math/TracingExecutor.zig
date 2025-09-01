const std = @import("std");
const sphtud = @import("sphtud");
const math = @import("../math.zig");
const cl = @import("../cl.zig");

const TracingExecutor = @This();

pub const Tensor = math.Tensor(NodeId);
pub const TensorSlice = math.TensorSlice(NodeId);

inner: math.Executor,
graph: sphtud.util.RuntimeBoundedArray(Node),

const TwoParam = struct {
    a: NodeId,
    b: NodeId,
};

const Operation = union(enum) {
    init,
    gt: struct {
        in: NodeId,
        dim: usize,
    },
    maxpool: struct {
        in: NodeId,
        stride: u32,
    },
    reshape: NodeId,
    squared_err: TwoParam,
    sigmoid: NodeId,
    relu: NodeId,
    matmul: TwoParam,
    add_splat_outer: TwoParam,
    conv: TwoParam,
    transpose: NodeId,
};

const NodeId = struct { usize };

// FIXME: requires_grad
const Node = struct {
    tensor: math.Executor.Tensor,
    operation: Operation,
};

const TensorRes = struct {
    event: cl.Executor.Event,
    val: Tensor,
};

pub fn init(inner: math.Executor, alloc: std.mem.Allocator, max_size: usize) !TracingExecutor {
    return .{
        .inner = inner,
        .graph = try .init(alloc, max_size),
    };
}

pub fn checkpoint(self: TracingExecutor) usize {
    return self.graph.items.len;
}

pub fn restore(self: *TracingExecutor, restore_point: usize) void {
    self.graph.resize(restore_point) catch unreachable;
}

pub fn clExecutor(self: *TracingExecutor) *cl.Executor {
    return self.inner.executor;
}

pub fn createTensor(self: *TracingExecutor, cl_alloc: *cl.Alloc, initial_data: []const f32, dims_in: []const u32) !TensorRes {
    const res = try self.inner.createTensor(cl_alloc, initial_data, dims_in);
    const tensor = try self.appendNode(res.val, .init);
    return .{
        .event = res.event,
        .val = tensor,
    };
}

pub fn createTensorUntracked(self: *TracingExecutor, cl_alloc: *cl.Alloc, initial_data: []const f32, dims_in: []const u32) !Tensor {
    return try self.appendNode(
        try self.inner.createTensorUntracked(cl_alloc, initial_data, dims_in),
        .init,
    );
}

pub fn createTensorFilled(self: *TracingExecutor, cl_alloc: *cl.Alloc, dims_in: []const u32, val: f32) !Tensor {
    return try self.appendNode(
        try self.inner.createTensorFilled(cl_alloc, dims_in, val),
        .init,
    );
}

pub fn rand(self: *TracingExecutor, cl_alloc: *cl.Alloc, dims_in: anytype, source: *math.RandSource) !Tensor {
    return try self.appendNode(
        try self.inner.rand(cl_alloc, dims_in, source),
        .init,
    );
}

pub fn randGaussian(self: *TracingExecutor, cl_alloc: *cl.Alloc, dims_in: anytype, stddev: f32, source: *math.RandSource) !Tensor {
    return try self.appendNode(
        try self.inner.randGaussian(cl_alloc, dims_in, stddev, source),
        .init,
    );
}

pub fn sliceToCpuDeferred(self: *const TracingExecutor, data_alloc: std.mem.Allocator, event_alloc: *cl.Alloc, tensor: TensorSlice) !math.Executor.Deferred([]f32) {
    return self.inner.sliceToCpuDeferred(
        data_alloc,
        event_alloc,
        .{
            .buf = self.getClTensor(tensor.buf).buf,
            .elem_offs = tensor.elem_offs,
            .dims = tensor.dims,
        },
    );
}

pub fn toCpu(self: *TracingExecutor, alloc: std.mem.Allocator, scratch_cl: *cl.Alloc, tensor: Tensor) ![]f32 {
    return self.inner.toCpu(alloc, scratch_cl, self.getClTensor(tensor.buf));
}

pub fn reshape(self: *TracingExecutor, cl_alloc: *cl.Alloc, val: Tensor, new_dims_in: []const u32) !Tensor {
    return try self.appendNode(
        try self.inner.reshape(cl_alloc, self.getClTensor(val.buf), new_dims_in),
        .{ .reshape = val.buf },
    );
}

pub fn transpose(self: *TracingExecutor, cl_alloc: *cl.Alloc, val: Tensor) !Tensor {
    return try self.appendNode(
        try self.inner.transpose(cl_alloc, self.getClTensor(val.buf)),
        .{ .transpose = val.buf },
    );
}

pub fn gt(self: *TracingExecutor, cl_alloc: *cl.Alloc, in: Tensor, dim: u32) !Tensor {
    return try self.appendNode(
        try self.inner.gt(cl_alloc, self.getClTensor(in.buf), dim),
        .{
            .gt = .{
                .in = in.buf,
                .dim = dim,
            },
        },
    );
}

pub fn convMany(self: *TracingExecutor, cl_alloc: *cl.Alloc, in: Tensor, kernel: Tensor) !Tensor {
    return try self.appendNode(
        try self.inner.convMany(cl_alloc, self.getClTensor(in.buf), self.getClTensor(kernel.buf)),
        .{
            .conv = .{
                .a = in.buf,
                .b = kernel.buf,
            },
        },
    );
}

pub fn maxpool(self: *TracingExecutor, cl_alloc: *cl.Alloc, in: Tensor, stride: u32) !Tensor {
    return try self.appendNode(
        try self.inner.maxpool(cl_alloc, self.getClTensor(in.buf), stride),
        .{
            .maxpool = .{
                .in = in.buf,
                .stride = stride,
            },
        },
    );
}

pub fn squaredErr(self: *TracingExecutor, cl_alloc: *cl.Alloc, a: Tensor, b: Tensor) !Tensor {
    return try self.appendNode(
        try self.inner.squaredErr(cl_alloc, self.getClTensor(a.buf), self.getClTensor(b.buf)),
        .{
            .squared_err = .{
                .a = a.buf,
                .b = b.buf,
            },
        },
    );
}

pub fn addSplatOuter(self: *TracingExecutor, cl_alloc: *cl.Alloc, a: Tensor, b: Tensor) !Tensor {
    const a_inner = self.getClTensor(a.buf);
    const b_inner = self.getClTensor(b.buf);
    return try self.appendNode(
        try self.inner.addSplatOuter(cl_alloc, a_inner, b_inner),
        .{
            .add_splat_outer = .{
                .a = a.buf,
                .b = b.buf,
            },
        },
    );
}

pub fn matmul(self: *TracingExecutor, cl_alloc: *cl.Alloc, a: Tensor, b: Tensor) !Tensor {
    const a_inner = self.getClTensor(a.buf);
    const b_inner = self.getClTensor(b.buf);
    return try self.appendNode(
        try self.inner.matmul(cl_alloc, a_inner, b_inner),
        .{
            .matmul = .{
                .a = a.buf,
                .b = b.buf,
            },
        },
    );
}

pub fn sigmoid(self: *TracingExecutor, cl_alloc: *cl.Alloc, in: Tensor) !Tensor {
    return try self.appendNode(
        try self.inner.sigmoid(cl_alloc, self.getClTensor(in.buf)),
        .{
            .sigmoid = in.buf,
        },
    );
}

pub fn relu(self: *TracingExecutor, cl_alloc: *cl.Alloc, in: Tensor) !Tensor {
    return try self.appendNode(
        try self.inner.relu(cl_alloc, self.getClTensor(in.buf)),
        .{
            .relu = in.buf,
        },
    );
}

pub fn appendNode(self: *TracingExecutor, cl_tensor: math.Executor.Tensor, operation: Operation) !Tensor {
    const ret = self.graph.items.len;
    try self.graph.append(.{
        .tensor = cl_tensor,
        .operation = operation,
    });
    const node_id = NodeId{ret};

    return .{
        .buf = node_id,
        // Do not need to clone, because this is conceptually the same
        // thing, we are just a wrapper type
        .dims = cl_tensor.dims,
    };
}

pub fn printBackwards(self: TracingExecutor, id: NodeId, indent_level: usize) void {
    const node = self.getNode(id);

    for (0..indent_level) |_| {
        std.debug.print("    ", .{});
    }
    std.debug.print("{s}: {d} (", .{ @tagName(node.operation), id[0] });
    for (node.tensor.dims.inner) |v| {
        std.debug.print("{d},", .{v});
    }
    std.debug.print(")\n", .{});

    switch (node.operation) {
        .init => {},
        .gt => |params| {
            self.printBackwards(params.in, indent_level + 1);
        },
        .reshape, .sigmoid => |next| {
            self.printBackwards(next, indent_level + 1);
        },
        .squared_err, .add_splat_outer, .matmul => |params| {
            self.printBackwards(params.a, indent_level + 1);
            self.printBackwards(params.b, indent_level + 1);
        },
    }
}

pub fn getClTensor(self: TracingExecutor, id: NodeId) math.Executor.Tensor {
    return self.getNode(id).tensor;
}

fn getNode(self: TracingExecutor, id: NodeId) *Node {
    return &self.graph.items[id[0]];
}

pub const Gradients = struct {
    graph: []math.Executor.Tensor,

    pub fn get(self: Gradients, id: NodeId) math.Executor.Tensor {
        return self.graph[id[0]];
    }
};

pub fn backprop(self: TracingExecutor, cl_alloc: *cl.Alloc, id: NodeId) !Gradients {
    var ret = Gradients{
        .graph = try cl_alloc.heap().alloc(math.Executor.Tensor, self.graph.items.len),
    };

    for (self.graph.items, 0..) |node, idx| {
        const gradient_size = node.tensor.dims.byteSize();
        const buf = try self.inner.executor.createBuffer(cl_alloc, .read_write, gradient_size);

        try self.inner.executor.fillBuffer(buf, 0.0, gradient_size);

        ret.graph[idx] = .{
            .buf = buf,
            .dims = try node.tensor.dims.clone(cl_alloc.heap()),
        };
    }

    const in_node = self.getNode(id);
    const initial_grad_buf = try self.inner.executor.createBuffer(cl_alloc, .read_write, in_node.tensor.dims.byteSize());
    try self.inner.executor.fillBuffer(initial_grad_buf, 1.0, in_node.tensor.dims.byteSize());
    ret.graph[id[0]] = .{ .buf = initial_grad_buf, .dims = in_node.tensor.dims };
    try self.backpropInner(cl_alloc, id, .{ .buf = initial_grad_buf, .dims = in_node.tensor.dims }, &ret);
    return ret;
}

fn backpropInner(self: TracingExecutor, cl_alloc: *cl.Alloc, id: NodeId, downstream_gradients: math.Executor.Tensor, gradient_tree: *Gradients) !void {
    const node = self.getNode(id);
    switch (node.operation) {
        .squared_err => |params| {
            try self.backpropTwoParams(cl_alloc, params, downstream_gradients, gradient_tree, math.Executor.squaredErrGrad);
        },
        .gt => {
            // This is either infinite(? Maybe at the limit it's 1 or something for some reason) or 0
            // Assume 0 for now and do nothing
        },
        .maxpool => |params| {
            const inputs = self.getClTensor(params.in);
            const grads = try self.inner.maxpoolGrad(cl_alloc, downstream_gradients, inputs, params.stride);

            // FIXME: This only has to happen if it's a gradient we care about
            try self.inner.addAssign(cl_alloc, gradient_tree.get(params.in), grads);

            try self.backpropInner(cl_alloc, params.in, grads, gradient_tree);
        },
        .reshape => |source_id| {
            // Here we just need to take our gradient_tree and transform them to match our parent

            const dims = self.getClTensor(source_id).dims;
            const grads = try self.inner.reshape(cl_alloc, downstream_gradients, dims);
            try self.backpropInner(cl_alloc, source_id, grads, gradient_tree);
        },
        .sigmoid => |source_id| {
            const inputs = self.getClTensor(source_id);
            const grads = try self.inner.sigmoidGrad(cl_alloc, downstream_gradients, inputs);

            // FIXME: This only has to happen if it's a gradient we care about
            try self.inner.addAssign(cl_alloc, gradient_tree.get(source_id), grads);

            try self.backpropInner(cl_alloc, source_id, grads, gradient_tree);
        },
        .relu => |source_id| {
            const inputs = self.getClTensor(source_id);
            const grads = try self.inner.reluGrad(cl_alloc, downstream_gradients, inputs);

            // FIXME: This only has to happen if it's a gradient we care about
            try self.inner.addAssign(cl_alloc, gradient_tree.get(source_id), grads);

            try self.backpropInner(cl_alloc, source_id, grads, gradient_tree);
        },
        .add_splat_outer => |params| {
            try self.backpropTwoParams(
                cl_alloc,
                params,
                downstream_gradients,
                gradient_tree,
                math.Executor.addSplatOuterGrad,
            );
        },
        .init => {},
        .matmul => |params| {
            // FIXME: Wasting GPU compute calculating gradients that we
            // don't need. We probably only need ONE of a/b
            try self.backpropTwoParams(
                cl_alloc,
                params,
                downstream_gradients,
                gradient_tree,
                math.Executor.matmulGrad,
            );
        },
        .conv => |params| {
            try self.backpropTwoParams(
                cl_alloc,
                params,
                downstream_gradients,
                gradient_tree,
                math.Executor.convManyGrad,
            );
        },
        .transpose => |source_id| {
            const grads = try self.inner.transpose(cl_alloc, downstream_gradients);
            try self.backpropInner(cl_alloc, source_id, grads, gradient_tree);
        },
    }
}

fn backpropTwoParams(self: TracingExecutor, cl_alloc: *cl.Alloc, params: TwoParam, downstream_gradients: math.Executor.Tensor, gradient_tree: *Gradients, gradFn: *const fn (math: math.Executor, cl_alloc: *cl.Alloc, downstream_gradients: math.Executor.Tensor, a: math.Executor.Tensor, b: math.Executor.Tensor) anyerror![2]math.Executor.Tensor) anyerror!void {
    const a = self.getClTensor(params.a);
    const b = self.getClTensor(params.b);

    // FIXME: Add cl.Alloc checkpoint

    const a_grads, const b_grads = try gradFn(self.inner, cl_alloc, downstream_gradients, a, b);

    // FIXME: This only has to happen if it's a gradient we care about
    try self.inner.addAssign(cl_alloc, gradient_tree.get(params.a), a_grads);
    try self.inner.addAssign(cl_alloc, gradient_tree.get(params.b), b_grads);

    try self.backpropInner(cl_alloc, params.a, a_grads, gradient_tree);
    try self.backpropInner(cl_alloc, params.b, b_grads, gradient_tree);
}
