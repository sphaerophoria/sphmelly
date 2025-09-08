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
        expected: CpuTensor,
        prediction: CpuTensor,
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
        mask: CpuTensor,
    } = null,
    prediction: ?struct {
        event: cl.Executor.Event,
        prediction: CpuTensor,
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
                .expected = s.mask,
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
        gt_mask: ?math.Executor.Tensor = null,
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

    fn batchGenerationQueued(self: *TrainNotifier, batch: math.Executor.Tensor, mask: math.Executor.Tensor) !void {
        self.cache.batch = batch;
        self.cache.gt_mask = mask;

        const req = self.current_data_request orelse return;
        const train_sample = req.train_sample orelse return;

        const output_alloc = (self.alloc orelse return).allocator();

        std.debug.assert(batch.dims.len() == 3);

        const batch_slice = try batch.indexOuter(train_sample);
        const batch_read_res = try self.tracing_executor.inner.sliceToCpuDeferred(output_alloc, self.cl_alloc, batch_slice);

        const mask_slice = try mask.indexOuter(train_sample);
        const mask_read_res = try self.tracing_executor.inner.sliceToCpuDeferred(output_alloc, self.cl_alloc, mask_slice);

        self.builder.train_sample = .{
            .event = mask_read_res.event,
            .data = CpuTensor{
                .buf = batch_read_res.val,
                .dims = try batch_slice.dims.clone(output_alloc),
            },
            .mask = .{
                .buf = mask_read_res.val,
                .dims = try mask_slice.dims.clone(output_alloc),
            },
        };
    }

    pub fn predictionsQueued(self: *TrainNotifier, predictions: math.TracingExecutor.Tensor) !void {
        self.cache.predictions = predictions;

        const output_alloc = (self.alloc orelse return).allocator();

        const req = self.current_data_request orelse return;
        const train_sample = req.train_sample orelse return;

        const prediction_slice = try predictions.indexOuter(train_sample);
        const prediction_read_res = try self.tracing_executor.sliceToCpuDeferred(
            output_alloc,
            self.cl_alloc,
            prediction_slice,
        );

        self.builder.prediction = .{
            .event = prediction_read_res.event,
            .prediction = .{
                .buf = prediction_read_res.val,
                .dims = try prediction_slice.dims.clone(output_alloc),
            },
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
            const m = self.cache.gt_mask orelse break :blk;
            try self.batchGenerationQueued(b, m);
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

const TrainingInput = struct {
    bars: BarcodeGen.Bars,
    input: math.Executor.Tensor,
    expected: math.Executor.Tensor,
};

fn generateTrainingInput(cl_alloc: *cl.Alloc, barcode_gen: *BarcodeGen, rand_params: BarcodeGen.RandomizationParams, math_executor: math.Executor, rand_source: *math.RandSource, _: *TrainNotifier, train_num_images: u32, barcode_size: u32, enable_backgrounds: bool) !TrainingInput {
    const bars = try barcode_gen.makeBars(cl_alloc, rand_params, enable_backgrounds, train_num_images, rand_source);

    const batch_cl_4d = try math_executor.reshape(cl_alloc, bars.imgs, &.{ barcode_size, barcode_size, 1, train_num_images });

    return .{
        .bars = bars,
        .input = batch_cl_4d,
        .expected = bars.bounding_boxes,
    };
}

const Config = struct {
    data: struct {
        batch_size: u32,
        img_size: u32,
        rand_params: BarcodeGen.RandomizationParams,
        enable_backgrounds: bool,
    },
    log_freq: u32,
    val_freq: u32,
    heal_orientations: bool,
    loss_multipliers: []f32,
    network: nn.Config,

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
};

fn trainThread(channels: *SharedChannels, background_dir: []const u8, config: Config, out_dir: std.fs.Dir) !void {
    const cl_alloc_buf = try std.heap.page_allocator.alloc(u8, 1 * 1024 * 1024);
    defer std.heap.page_allocator.free(cl_alloc_buf);

    var cl_alloc: cl.Alloc = undefined;
    try cl_alloc.initPinned(cl_alloc_buf);
    defer cl_alloc.deinit();

    const profiling_mode = cl.Executor.ProfilingMode.non_profiling;

    var cl_executor = try cl.Executor.init(cl_alloc.heap(), profiling_mode);
    defer cl_executor.deinit();

    const math_executor = try math.Executor.init(&cl_alloc, &cl_executor);
    var tracing_executor = try math.TracingExecutor.init(math_executor, cl_alloc.heap(), 1000);

    var notifier = TrainNotifier.init(&cl_alloc, channels, &tracing_executor);

    var rand_source = math.RandSource{
        .ctr = 0,
        .seed = 1,
    };

    const initializers = nn.makeInitializers(&tracing_executor, &rand_source);
    const layers = try nn.modelFromConfig(&cl_alloc, &tracing_executor, &initializers, config.network.layers);

    var barcode_gen = try BarcodeGen.init(
        // Probably a bit of a violation of separation, but we know that
        // everything that goes on cl_alloc will use the front half of the
        // buffer, so it's safe for us to use the back half for scratch space
        cl_alloc.buf_alloc.backLinear(),
        &cl_alloc,
        math_executor,
        background_dir,
        config.data.img_size,
    );

    var trainer = nn.Trainer(TrainNotifier){
        .optimizer = try .init(.{
            .lr = config.network.lr,
            .executor = &tracing_executor,
            .cl_alloc = &cl_alloc,
        }),
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

    const val_size = 200;
    const validation_set = try generateTrainingInput(
        &cl_alloc,
        &barcode_gen,
        config.data.rand_params,
        math_executor,
        &rand_source,
        &notifier,
        val_size,
        config.data.img_size,
        config.data.enable_backgrounds,
    );

    var iter: usize = 0;
    const start_time = try std.time.Instant.now();

    const log_file = try out_dir.createFile("log.csv", .{});
    defer log_file.close();

    var buf_writer = std.io.bufferedWriter(log_file.writer());
    defer buf_writer.flush() catch {};

    const log_writer = buf_writer.writer();

    const math_checkpoint = tracing_executor.checkpoint();
    const cl_alloc_checkpoint = cl_alloc.checkpoint();

    while (true) {
        switch (pause) {
            .unpaused => {
                cl_executor.resetTimers();
                tracing_executor.restore(math_checkpoint);
                cl_alloc.reset(cl_alloc_checkpoint);

                const train_input = try generateTrainingInput(
                    &cl_alloc,
                    &barcode_gen,
                    config.data.rand_params,
                    math_executor,
                    &rand_source,
                    &notifier,
                    config.data.batch_size,
                    config.data.img_size,
                    config.data.enable_backgrounds,
                );

                const results = try nn.runLayers(&cl_alloc, train_input.input, layers, &tracing_executor);
                try notifier.notifyLayerOutputs(results);
                try notifier.predictionsQueued(results[results.len - 1]);

                if (config.heal_orientations) {
                    try barcode_gen.healOrientations(&cl_alloc, train_input.expected, tracing_executor.getClTensor(results[results.len - 1].buf));
                }
                try notifier.batchGenerationQueued(train_input.bars.imgs, train_input.bars.bounding_boxes);
                const traced_expected = try tracing_executor.appendNode(train_input.expected, .init);
                const even_loss = try tracing_executor.squaredErr(&cl_alloc, results[results.len - 1], traced_expected);
                const loss_multipliers = try math_executor.createTensorUntracked(&cl_alloc, config.loss_multipliers, &.{6});
                const loss = try tracing_executor.elemMul(&cl_alloc, even_loss, loss_multipliers);
                try notifier.notifyLoss(loss);

                switch (try trainer.step(loss)) {
                    .nan => {
                        pause = .{ .paused = cl_alloc.checkpoint() };
                        continue;
                    },
                    .ok => {},
                }

                iter += 1;
                if (iter % config.log_freq == 0) {
                    const wall_time = (try std.time.Instant.now()).since(start_time);

                    try log_writer.print("step,{d},{d},{d}\n", .{ iter, wall_time, iter * config.data.batch_size });
                    const loss_cpu = try tracing_executor.toCpu(cl_alloc.heap(), &cl_alloc, loss);
                    var total_losses: [6]f32 = @splat(0);
                    for (0..loss_cpu.len / 6) |i| {
                        for (0..6) |j| {
                            total_losses[j] += loss_cpu[i * 6 + j];
                        }
                    }

                    const labels: []const []const u8 = &.{ "dx", "dy", "dw", "dh", "drx", "dry" };
                    for (total_losses, labels) |total_loss, label| {
                        try log_writer.print("{s} loss,{d}\n", .{ label, total_loss });
                    }
                    var total_loss: f32 = 0;
                    for (total_losses) |v| {
                        total_loss += v;
                    }
                    try log_writer.print("total loss,{d}\n", .{total_loss});

                    if (iter % config.val_freq == 0) {
                        const val_results = try nn.runLayers(&cl_alloc, validation_set.input, layers, &tracing_executor);

                        if (config.heal_orientations) {
                            try barcode_gen.healOrientations(&cl_alloc, validation_set.expected, tracing_executor.getClTensor(val_results[val_results.len - 1].buf));
                        }

                        const val_err = try math_executor.sub(&cl_alloc, tracing_executor.getClTensor(val_results[val_results.len - 1].buf), validation_set.expected);

                        const val_err_cpu = try math_executor.toCpu(cl_alloc.heap(), &cl_alloc, val_err);
                        var total_val_mse: f32 = 0;
                        for (val_err_cpu) |l| {
                            total_val_mse += l * l;
                        }

                        std.debug.assert(val_err.dims.get(0) == 6);
                        std.debug.assert(val_err.dims.len() == 2);
                        var totals: [6]f32 = @splat(0);
                        var sum_squares: [6]f32 = @splat(0);
                        for (0..val_err_cpu.len / 6) |i| {
                            for (0..6) |j| {
                                const val = val_err_cpu[i * 6 + j];
                                totals[j] += @abs(val);
                                sum_squares[j] += val * val;
                            }
                        }
                        for (&totals) |*t| {
                            t.* /= val_size;
                        }
                        for (&sum_squares) |*t| {
                            t.* /= val_size;
                        }

                        try log_writer.print("val mse,{d}\n", .{total_val_mse / val_size});

                        for (labels, totals, sum_squares) |label, average, mse| {
                            try log_writer.print("avg {s},{d}\n", .{ label, average });
                            try log_writer.print("mse {s},{d}\n", .{ label, mse });
                        }
                    }

                    try buf_writer.flush();
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

const Args = struct {
    background_dir: []const u8,
    config: []const u8,
    out_dir: []const u8,

    const Switch = enum {
        @"--background-dir",
        @"--config",
        @"--out-dir",
    };

    fn parse(alloc: std.mem.Allocator) !Args {
        var it = try std.process.argsWithAllocator(alloc);

        const process_name = it.next() orelse "sphmelly";

        var background_dir: ?[]const u8 = null;
        var config: ?[]const u8 = null;
        var out_dir: ?[]const u8 = null;

        while (it.next()) |arg| {
            const s = std.meta.stringToEnum(Switch, arg) orelse {
                std.log.err("{s} is not a valid argument", .{arg});
                help(process_name);
            };

            switch (s) {
                .@"--background-dir" => background_dir = it.next() orelse {
                    std.log.err("Missing background dir arg", .{});
                    help(process_name);
                },
                .@"--config" => config = it.next() orelse {
                    std.log.err("Missing config arg", .{});
                    help(process_name);
                },
                .@"--out-dir" => out_dir = it.next() orelse {
                    std.log.err("Missing out dir arg", .{});
                    help(process_name);
                },
            }
        }

        return .{
            .background_dir = background_dir orelse {
                std.log.err("background dir not provided", .{});
                help(process_name);
            },
            .config = config orelse {
                std.log.err("config not provided", .{});
                help(process_name);
            },
            .out_dir = out_dir orelse {
                std.log.err("out dir not provided", .{});
                help(process_name);
            },
        };
    }

    fn help(process_name: []const u8) noreturn {
        const stdout = std.io.getStdOut();

        stdout.writer().print(
            \\USAGE: {s} [ARGS]
            \\
            \\Required args:
            \\--background-dir: Where to load image backgrounds from
            \\--config: Training configuration path
            \\--out-dir: Where to save training output
            \\
        , .{process_name}) catch {};

        std.process.exit(1);
    }
};

pub fn main() !void {
    var allocators: AppAllocators = undefined;
    try allocators.initPinned(30 * 1024 * 1024);

    const args = try Args.parse(allocators.root.arena());

    try std.fs.cwd().makeDir(args.out_dir);

    var out_dir = try std.fs.cwd().openDir(args.out_dir, .{});
    defer out_dir.close();

    const config = try Config.parse(allocators.root.arena(), args.config);

    {
        const config_out = try out_dir.createFile("config.json", .{});
        defer config_out.close();

        try std.json.stringify(config, .{ .whitespace = .indent_2 }, config_out.writer());
    }

    var ui: train_ui.Gui = undefined;
    try ui.initPinned(&allocators, config.network.lr, config.data.batch_size);

    var comms = SharedChannels{};

    const train_thread = try std.Thread.spawn(.{}, trainThread, .{ &comms, args.background_dir, config, out_dir });
    defer blk: {
        comms.gui_to_train.send(.shutdown) catch break :blk;
        train_thread.join();
    }

    var req_alloc_buf = try TrainingReqDoubleBuffer.init(
        allocators.scratch.allocator(),
        10 * 1024 * 1024,
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
                try ui.renderBBoxOverlay(&allocators.scratch_gl, sample.expected, .{ 1, 0, 0 });
                try ui.renderBBoxOverlay(&allocators.scratch_gl, sample.prediction, .{ 0, 0, 1 });

                ui.params.img_loss = sample.loss;
            } else {
                ui.params.img_loss = 0;
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
