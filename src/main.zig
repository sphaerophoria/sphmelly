const std = @import("std");
const sphtud = @import("sphtud");
const cl = @import("cl.zig");
const math = @import("math.zig");
const sphalloc = sphtud.alloc;
const sphrender = sphtud.render;
const gl = sphrender.gl;
const sphwindow = sphtud.window;
const gui = sphtud.ui;
const BarcodeGen = @import("BarcodeGen.zig");
const tsv = @import("training_sample_view.zig");
const nn = @import("nn.zig");
const Optimizer = nn.Optimizer;
const AppAllocators = sphrender.AppAllocators(100);
const train_ui = @import("train_ui.zig");

fn OpenClLayerGen(comptime Executor: type) type {
    return struct {
        cl_alloc: *cl.Alloc,
        math_executor: *Executor,

        const Self = @This();

        fn fullyConnected(self: Self, alloc: std.mem.Allocator, weights_initializer: anytype, bias_initializer: anytype, inputs: u32, outputs: u32) !nn.Layer(Executor) {
            const fc = try alloc.create(nn.FullyConnected(Executor));
            fc.* = try nn.FullyConnected(Executor).init(
                self.cl_alloc,
                weights_initializer,
                bias_initializer,
                inputs,
                outputs,
            );
            return fc.layer();
        }

        fn conv(self: Self, alloc: std.mem.Allocator, weights_initializer: anytype, w: u32, h: u32, in_c: u32, out_c: u32) !nn.Layer(Executor) {
            const conv_layer = try alloc.create(nn.Conv(Executor));
            conv_layer.* = try nn.Conv(Executor).init(self.cl_alloc, weights_initializer, w, h, in_c, out_c);
            return conv_layer.layer();
        }

        fn reshape(_: Self, alloc: std.mem.Allocator, dims: []const u32) !nn.Layer(Executor) {
            const layer = try alloc.create(nn.Reshape(Executor));
            layer.* = try nn.Reshape(Executor).init(dims);
            return layer.layer();
        }

        fn sigmoid(_: Self) nn.Layer(Executor) {
            return nn.sigmoidLayer(Executor);
        }

        fn relu(_: Self) nn.Layer(Executor) {
            return nn.reluLayer(Executor);
        }
    };
}

const GuiDataReq = struct {
    alloc_buf: []u8,
    loss: bool,
    active_layer_id: u32,
    train_sample: ?u32 = null,
    gradient: ?struct {
        layer_idx: u32,
        param_idx: u32,
    } = null,
    weights: ?struct {
        layer_idx: u32,
        param_idx: u32,
    } = null,
    layer_output: ?struct {
        img_idx: u32,
        layer_id: u32,
    } = null,
};

const CpuTensor = math.Tensor([]f32);

const TrainRequest = union(enum) {
    gui_data: GuiDataReq,
    update_lr: f32,
    toggle_pause,
    shutdown,
};

const TrainResponse = struct {
    train_sample: ?struct {
        img: CpuTensor,
        orientation: []f32,
        prediction: []const f32,
        loss: f32,
    } = null,
    grads: ?CpuTensor = null,
    weights: ?CpuTensor = null,
    layer_output: ?CpuTensor = null,

    layer_name_update: []const u8 = &.{},
    num_layers: u32 = 0,

    loss: f32 = 0.0,
};

const CpuTensorReadRes = struct {
    const empty: @This() = .{
        .event = null,
        .elem = .{
            .buf = &.{},
            .dims = .empty,
        },
    };

    event: ?cl.Executor.Event,
    elem: CpuTensor,
};

const TrainResponseBuilder = struct {
    train_sample: ?struct {
        event: cl.Executor.Event,
        data: CpuTensor,
        orientation: []f32,
    } = null,
    prediction: ?struct {
        event: cl.Executor.Event,
        prediction: []const f32,
    } = null,
    loss: ?struct {
        event: cl.Executor.Event,
        loss: []const f32,
    } = null,
    gradients: ?CpuTensorReadRes = null,
    layer_output: ?struct {
        event: cl.Executor.Event,
        data: CpuTensor,
    } = null,

    fn finish(self: *TrainResponseBuilder, req: GuiDataReq, layers: anytype, weights: ?CpuTensor) !TrainResponse {
        defer self.* = .{};

        var res = TrainResponse{};

        if (self.train_sample) |s| {
            try s.event.wait();
            const prediction = self.prediction orelse return error.NoPrediction;
            try prediction.event.wait();

            var loss: f32 = -1;
            if (self.loss) |l| blk: {
                const sample_idx = req.train_sample orelse break :blk;
                try l.event.wait();
                loss = l.loss[sample_idx * 2] + l.loss[sample_idx * 2 + 1];
            }

            res.train_sample = .{
                .img = s.data,
                .orientation = s.orientation,
                .prediction = prediction.prediction,
                .loss = loss,
            };
        }

        if (self.loss) |l| {
            try l.event.wait();
            var total_loss: f32 = 0;
            for (l.loss) |v| {
                total_loss += v;
            }

            res.loss = total_loss;
        }

        if (self.gradients) |grads| blk: {
            const event = grads.event orelse break :blk;
            try event.wait();

            res.grads = grads.elem;
        }

        if (self.layer_output) |output| {
            try output.event.wait();

            res.layer_output = output.data;
        }

        if (req.active_layer_id < layers.len) {
            res.layer_name_update = layers[req.active_layer_id].name;
        }

        res.weights = weights;
        res.num_layers = @intCast(layers.len);

        return res;
    }
};

pub fn SharedChannel(comptime T: type, comptime size: usize) type {
    return struct {
        mutex: std.Thread.Mutex,
        protected: std.fifo.LinearFifo(T, .{ .Static = size }),

        const Self = @This();

        const init = Self{
            .mutex = .{},
            .protected = .init(),
        };

        fn send(self: *Self, val: T) !void {
            self.mutex.lock();
            defer self.mutex.unlock();

            try self.protected.writeItem(val);
        }

        fn poll(self: *Self) ?T {
            self.mutex.lock();
            defer self.mutex.unlock();

            return self.protected.readItem();
        }
    };
}

const SharedChannels = struct {
    gui_to_train: SharedChannel(TrainRequest, 1024) = .init,
    train_to_gui: SharedChannel(TrainResponse, 1024) = .init,
};

const DeferredRead = struct {
    data: []f32,
    event: cl.Executor.Event,
};

const TrainNotifier = struct {
    comms: *SharedChannels,
    current_data_request: ?GuiDataReq = null,
    alloc: ?sphtud.alloc.BufAllocator = null,
    builder: TrainResponseBuilder = .{},
    cl_alloc: *cl.Alloc,
    cl_executor: *cl.Executor,
    tracing_executor: *const math.TracingExecutor,
    cache: struct {
        batch: ?math.Executor.Tensor = null,
        loss: ?math.TracingExecutor.Tensor = null,
        orientations: ?math.Executor.Tensor = null,
        predictions: ?math.TracingExecutor.Tensor = null,
        gradients: ?math.TracingExecutor.Gradients = null,
        layer_outputs: ?[]math.TracingExecutor.Tensor = null,
    } = .{},

    fn init(cl_alloc: *cl.Alloc, comms: *SharedChannels, tracing_executor: *const math.TracingExecutor) TrainNotifier {
        return .{
            .comms = comms,
            .cl_alloc = cl_alloc,
            .tracing_executor = tracing_executor,
            .cl_executor = tracing_executor.inner.executor,
        };
    }

    fn batchGenerationQueued(self: *TrainNotifier, batch: math.Executor.Tensor, orientations: math.Executor.Tensor) !void {
        self.cache.batch = batch;
        self.cache.orientations = orientations;

        const req = self.current_data_request orelse return;
        const train_sample = req.train_sample orelse return;

        const output_alloc = (self.alloc orelse return).allocator();

        std.debug.assert(batch.dims.len() == 3);
        std.debug.assert(orientations.dims.len() == 2);

        const batch_slice = try batch.indexOuter(train_sample);
        const batch_read_res = try self.tracing_executor.inner.sliceToCpuDeferred(output_alloc, self.cl_alloc, batch_slice);

        const orientation_read_res = try self.tracing_executor.inner.sliceToCpuDeferred(output_alloc, self.cl_alloc, try orientations.indexOuter(train_sample));

        self.builder.train_sample = .{
            .event = orientation_read_res.event,
            .data = CpuTensor{
                .buf = batch_read_res.val,
                .dims = try batch_slice.dims.clone(output_alloc),
            },
            .orientation = orientation_read_res.val,
        };
    }

    pub fn predictionsQueued(self: *TrainNotifier, predictions: math.TracingExecutor.Tensor) !void {
        self.cache.predictions = predictions;

        const output_alloc = (self.alloc orelse return).allocator();

        const req = self.current_data_request orelse return;
        const train_sample = req.train_sample orelse return;

        const prediction_read_res = try self.tracing_executor.sliceToCpuDeferred(
            output_alloc,
            self.cl_alloc,
            try predictions.indexOuter(train_sample),
        );

        self.builder.prediction = .{
            .event = prediction_read_res.event,
            .prediction = prediction_read_res.val,
        };
    }

    pub fn notifyLayerOutputs(self: *TrainNotifier, layer_outputs: []math.TracingExecutor.Tensor) !void {
        self.cache.layer_outputs = layer_outputs;

        const req = self.current_data_request orelse return;
        const layer_req = req.layer_output orelse return;

        const output_alloc = (self.alloc orelse return).allocator();

        if (layer_req.layer_id >= layer_outputs.len) return;

        const layer_output = layer_outputs[layer_req.layer_id];

        if (layer_req.img_idx >= layer_output.dims.get(layer_output.dims.len() - 1)) return;

        const img_output_slice = try layer_output.indexOuter(layer_req.img_idx);
        const read_res = try self.tracing_executor.sliceToCpuDeferred(output_alloc, self.cl_alloc, img_output_slice);

        self.builder.layer_output = .{
            .event = read_res.event,
            .data = .{
                .buf = read_res.val,
                .dims = try img_output_slice.dims.clone(output_alloc),
            },
        };
    }

    pub fn notifyLoss(self: *TrainNotifier, loss: math.TracingExecutor.Tensor) !void {
        self.cache.loss = loss;

        const req = self.current_data_request orelse return;
        if (!req.loss) return;

        // NOTE: result of this loss does not get transfered to gl thread, so
        // we don't need to use the output allocator
        const read_res = try self.tracing_executor.sliceToCpuDeferred(
            self.cl_alloc.heap(),
            self.cl_alloc,
            loss.asSlice(),
        );

        self.builder.loss = .{
            .event = read_res.event,
            .loss = read_res.val,
        };
    }

    pub fn notifyGradients(self: *TrainNotifier, layers: []const nn.Layer(math.TracingExecutor), gradients: math.TracingExecutor.Gradients) !void {
        self.cache.gradients = gradients;

        const req = self.current_data_request orelse return;
        const gradient_req = req.gradient orelse return;

        const output_alloc = (self.alloc orelse return).allocator();

        if (gradient_req.layer_idx >= layers.len) {
            self.builder.gradients = .empty;
            return;
        }

        const weights = layers[gradient_req.layer_idx].getWeights(gradient_req.param_idx) catch {
            self.builder.gradients = .empty;
            return;
        } orelse {
            self.builder.gradients = .empty;
            return;
        };

        const weight_grads = gradients.get(weights.buf);
        const weight_grads_slice: math.Executor.TensorSlice = .{
            .buf = weight_grads.buf,
            .dims = weights.dims,
            .elem_offs = weights.elem_offs,
        };

        const read_res = try self.tracing_executor.inner.sliceToCpuDeferred(output_alloc, self.cl_alloc, weight_grads_slice);

        self.builder.gradients = .{
            .event = read_res.event,
            .elem = .{
                .buf = read_res.val,
                .dims = try weights.dims.clone(output_alloc),
            },
        };
    }

    fn extractWeights(self: *TrainNotifier, layers: []const nn.Layer(math.TracingExecutor)) !?CpuTensor {
        // FIXME: Heavy duplication with gradients
        const req = self.current_data_request orelse return null;
        const gradient_req = req.weights orelse return null;

        const output_alloc = (self.alloc orelse return null).allocator();

        if (gradient_req.layer_idx >= layers.len) {
            return null;
        }

        const weights = layers[gradient_req.layer_idx].getWeights(gradient_req.param_idx) catch {
            return null;
        } orelse {
            return null;
        };

        const read_res = try self.tracing_executor.sliceToCpuDeferred(output_alloc, self.cl_alloc, weights);
        try read_res.event.wait();

        return .{
            .buf = read_res.val,
            .dims = try weights.dims.clone(output_alloc),
        };
    }

    const NotifyAction = struct {
        shutdown: bool = false,
        toggle_pause: bool = false,
        update_lr: ?f32 = null,
    };

    fn buildFromCached(self: *TrainNotifier, layers: []const nn.Layer(math.TracingExecutor)) !void {
        if (self.cache.batch) |b| blk: {
            const o = self.cache.orientations orelse break :blk;
            try self.batchGenerationQueued(b, o);
        }

        if (self.cache.predictions) |p| {
            try self.predictionsQueued(p);
        }

        if (self.cache.loss) |l| {
            try self.notifyLoss(l);
        }

        if (self.cache.gradients) |g| {
            try self.notifyGradients(layers, g);
        }

        if (self.cache.layer_outputs) |o| {
            try self.notifyLayerOutputs(o);
        }
    }

    fn finish(self: *TrainNotifier, layers: anytype) !NotifyAction {
        if (self.current_data_request) |r| {
            const weights = try self.extractWeights(layers);
            const response = try self.builder.finish(r, layers, weights);
            self.alloc = null;
            self.current_data_request = null;
            try self.comms.train_to_gui.send(response);
        }

        var ret = NotifyAction{};
        var pause_count: u1 = 0;
        while (self.comms.gui_to_train.poll()) |next| {
            switch (next) {
                .gui_data => |d| {
                    self.alloc = sphtud.alloc.BufAllocator.init(d.alloc_buf);
                    self.current_data_request = d;
                },
                .update_lr => |lr| {
                    ret.update_lr = lr;
                },
                .toggle_pause => {
                    pause_count +%= 1;
                },
                .shutdown => {
                    ret.shutdown = true;
                    return ret;
                },
            }
        }

        ret.toggle_pause = pause_count > 0;

        return ret;
    }
};

const train_num_images = 200;
const default_lr = 0.05;

fn trainThread(channels: *SharedChannels) !void {
    const cl_alloc_buf = try std.heap.page_allocator.alloc(u8, 1 * 1024 * 1024);
    defer std.heap.page_allocator.free(cl_alloc_buf);

    var cl_alloc: cl.Alloc = undefined;
    try cl_alloc.initPinned(cl_alloc_buf);
    defer cl_alloc.deinit();

    const profiling_mode = cl.Executor.ProfilingMode.non_profiling;

    var cl_executor = try cl.Executor.init(cl_alloc.heap(), profiling_mode);
    defer cl_executor.deinit();

    const math_executor = try math.Executor.init(&cl_alloc, &cl_executor);
    var tracing_executor = try math.TracingExecutor.init(math_executor, cl_alloc.heap(), 100);

    var notifier = TrainNotifier.init(&cl_alloc, channels, &tracing_executor);

    const layer_gen = OpenClLayerGen(math.TracingExecutor){
        .math_executor = &tracing_executor,
        .cl_alloc = &cl_alloc,
    };

    var rand_source = math.RandSource{
        .ctr = 0,
        .seed = 1,
    };

    const barcode_size = 64;

    const he_initializer = nn.HeInitializer(math.TracingExecutor){
        .executor = &tracing_executor,
        .rand_source = &rand_source,
    };

    const zero_initializer = nn.ZeroInitializer(math.TracingExecutor){
        .executor = &tracing_executor,
    };

    const layers: []const nn.Layer(math.TracingExecutor) = &.{
        try layer_gen.conv(cl_alloc.heap(), he_initializer, 3, 3, 1, 4),
        layer_gen.relu(),
        try layer_gen.conv(cl_alloc.heap(), he_initializer, 3, 3, 4, 2),
        layer_gen.relu(),
        try layer_gen.reshape(cl_alloc.heap(), &.{ barcode_size * barcode_size * 2, train_num_images }),
        try layer_gen.fullyConnected(cl_alloc.heap(), he_initializer, zero_initializer, barcode_size * barcode_size * 2, 16),
        layer_gen.relu(),
        try layer_gen.fullyConnected(cl_alloc.heap(), he_initializer, zero_initializer, 16, 16),
        layer_gen.relu(),
        try layer_gen.fullyConnected(cl_alloc.heap(), he_initializer, zero_initializer, 16, 2),
    };

    var barcode_gen = try BarcodeGen.init(&cl_alloc, math_executor);
    const rand_params = BarcodeGen.RandomizationParams{
        // FIXME offset range should probably be a ratio of image size, not absolute pixels
        .x_offs_range = .{ -50, 50 },
        .y_offs_range = .{ -50, 50 },
        .x_scale_range = .{ 2.0, 3.0 },
        .aspect_range = .{ 1.0, 2.0 },
        .min_contrast = 0.2,
        .x_noise_multiplier_range = .{ 5.0, 10.1 },
        .y_noise_multiplier_range = .{ 5.0, 10.1 },
        .perlin_grid_size_range = .{ 10, 100 },
        .background_color_range = .{ 0.0, 1.0 },
        // FIXME Blur amount is in pixel space, maybe these should be
        // scaled by resolution of inputs
        .blur_stddev_range = .{ 0.0001, 3.0 },
    };

    var trainer = nn.Trainer(TrainNotifier){
        .optimizer = .{
            .lr = default_lr,
            .executor = math_executor,
            .traced = &tracing_executor,
            .weights = try .init(cl_alloc.heap(), 100),
        },
        .cl_alloc = &cl_alloc,
        .layers = layers,
        .tracing_executor = &tracing_executor,
        .notifier = &notifier,
    };

    for (layers) |*layer| {
        try layer.registerWeights(&trainer.optimizer);
    }

    var pause: union(enum) {
        paused: cl.Alloc.Checkpoint,
        unpaused,
    } = .unpaused;

    const math_checkpoint = tracing_executor.checkpoint();
    const cl_alloc_checkpoint = cl_alloc.checkpoint();

    while (true) {
        switch (pause) {
            .unpaused => {
                cl_executor.resetTimers();
                tracing_executor.restore(math_checkpoint);
                cl_alloc.reset(cl_alloc_checkpoint);

                const bars = try barcode_gen.makeBars(&cl_alloc, rand_params, &.{ barcode_size, barcode_size, train_num_images }, &rand_source);
                try notifier.batchGenerationQueued(bars.imgs, bars.orientations);

                const batch_cl_4d = try math_executor.reshape(&cl_alloc, bars.imgs, &.{ barcode_size, barcode_size, 1, train_num_images });

                switch (try trainer.step(batch_cl_4d, bars.orientations)) {
                    .nan => {
                        pause = .{ .paused = cl_alloc.checkpoint() };
                        continue;
                    },
                    .ok => {},
                }

                try cl_executor.finish();
            },
            .paused => |pause_cp| {
                cl_alloc.reset(pause_cp);
                try notifier.buildFromCached(layers);
                std.time.sleep(std.time.ns_per_ms * 30);
            },
        }

        const actions = try notifier.finish(layers);

        if (actions.shutdown) {
            return;
        }

        if (actions.toggle_pause) {
            pause = switch (pause) {
                .paused => .unpaused,
                .unpaused => .{ .paused = cl_alloc.checkpoint() },
            };
        }

        if (actions.update_lr) |new_lr| {
            trainer.optimizer.lr = new_lr;
        }

        try cl_executor.finish();
    }
}

const TrainingReqDoubleBuffer = struct {
    bufs: [2][]u8,
    idx: u1,

    fn init(alloc: std.mem.Allocator, buf_size: usize) !TrainingReqDoubleBuffer {
        return .{
            .bufs = .{
                try alloc.alloc(u8, buf_size),
                try alloc.alloc(u8, buf_size),
            },
            .idx = 0,
        };
    }

    fn next(self: TrainingReqDoubleBuffer) []u8 {
        return self.bufs[self.idx];
    }

    fn swap(self: *TrainingReqDoubleBuffer) void {
        self.idx +%= 1;
    }
};

pub fn main() !void {
    var allocators: AppAllocators = undefined;
    try allocators.initPinned(20 * 1024 * 1024);

    var ui: train_ui.Gui = undefined;
    try ui.initPinned(&allocators, default_lr, train_num_images);

    var comms = SharedChannels{};

    const train_thread = try std.Thread.spawn(.{}, trainThread, .{&comms});
    defer blk: {
        comms.gui_to_train.send(.shutdown) catch break :blk;
        train_thread.join();
    }

    var req_alloc_buf = try TrainingReqDoubleBuffer.init(
        allocators.scratch.allocator(),
        5 * 1024 * 1024,
    );

    var outstanding_req: usize = 0;
    var num_completed_reqs: usize = 0;

    const checkpoint = allocators.scratch.checkpoint();

    while (!ui.window.closed()) {
        allocators.scratch.restore(checkpoint);

        const width, const height = ui.window.getWindowSize();

        gl.glViewport(0, 0, @intCast(width), @intCast(height));
        gl.glScissor(0, 0, @intCast(width), @intCast(height));

        gl.glClear(gl.GL_COLOR_BUFFER_BIT);

        const action = try ui.step(@intCast(width), @intCast(height));
        if (action) |a| switch (a) {
            .pause => {
                try comms.gui_to_train.send(.toggle_pause);
            },
            .lr_update => {
                try comms.gui_to_train.send(.{
                    .update_lr = ui.params.lr,
                });
            },
            else => {},
        };

        if (outstanding_req == num_completed_reqs) {
            // FIXME: This should only happen when gui requests something new maybe?
            // It's spamming like crazy
            var req = GuiDataReq{
                .alloc_buf = req_alloc_buf.next(),
                .loss = true,
                .active_layer_id = ui.params.current_layer,
            };

            req.alloc_buf = req_alloc_buf.next();

            switch (ui.params.view_mode) {
                .training_sample => {
                    req.train_sample = ui.params.current_img;
                },
                .gradients => {
                    req.gradient = .{
                        .layer_idx = ui.params.current_layer,
                        .param_idx = ui.params.current_param,
                    };
                },
                .layer_out => {
                    req.layer_output = .{
                        .img_idx = ui.params.current_img,
                        .layer_id = ui.params.current_layer,
                    };
                },
                .weights => {
                    req.weights = .{
                        .layer_idx = ui.params.current_layer,
                        .param_idx = ui.params.current_param,
                    };
                },
            }

            try comms.gui_to_train.send(.{ .gui_data = req });
            outstanding_req += 1;
        }

        while (comms.train_to_gui.poll()) |response| {
            // Make sure nothing is referencing old data

            defer req_alloc_buf.swap();

            num_completed_reqs += 1;
            std.debug.assert(num_completed_reqs == outstanding_req);

            try ui.clear();

            ui.params.current_loss = response.loss;

            if (response.train_sample) |sample| {
                try ui.setImageGrayscale(allocators.scratch.allocator(), sample.img);

                ui.widgets.orientation.setOrientation(sample.orientation[0..2].*);
                ui.widgets.predicted_orientation.setOrientation(sample.prediction[0..2].*);

                ui.params.img_loss = sample.loss;
                ui.params.img_predicted = sample.prediction[0..2].*;
                ui.params.img_ground_truth = sample.orientation[0..2].*;
            } else {
                ui.params.img_loss = 0;
                ui.params.img_predicted = .{ 0, 0 };
                ui.params.img_ground_truth = .{ 0, 0 };
            }

            if (response.grads) |grads| {
                try ui.setImageHotCold(allocators.scratch.allocator(), grads);
            }

            if (response.layer_output) |sample| {
                try ui.setImageHotCold(allocators.scratch.allocator(), sample);
            }

            if (response.weights) |weights| {
                try ui.setImageHotCold(allocators.scratch.allocator(), weights);
            }

            ui.params.num_layers = response.num_layers;

            // FIXME: Need to make sure GUI goes out of scope before thread starts to shut down
            ui.params.layer_name = response.layer_name_update;
        }

        ui.window.swapBuffers();
    }
}

test {
    _ = std.testing.refAllDeclsRecursive(@This());
}
