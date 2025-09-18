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
const nn_checkpoint = @import("nn/checkpoint.zig");
const Config = @import("Config.zig");
const training_stats = @import("training_stats.zig");

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

const TrainResponse = union(enum) {
    const GuiData = struct {
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

    gui_data: GuiData,
    shutdown,
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

    fn finish(self: *TrainResponseBuilder, out_alloc: std.mem.Allocator, req: GuiDataReq, layers: anytype, weights: ?CpuTensor) !TrainResponse {
        defer self.* = .{};

        var res = TrainResponse.GuiData{};

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
            res.layer_name_update = try out_alloc.dupe(u8, layers[req.active_layer_id].name);
        }

        res.weights = weights;
        res.num_layers = @intCast(layers.len);

        return .{ .gui_data = res };
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
            const response = try self.builder.finish(self.alloc.?.allocator(), r, layers, weights);
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

fn generateTrainingInput(barcode_gen: *BarcodeGen, make_bars_params: BarcodeGen.MakeBarsParams, math_executor: math.Executor, target: Config.TrainTarget, barcode_size: u32) !TrainingInput {
    const bars = try barcode_gen.makeBars(make_bars_params);

    const batch_cl_4d = try math_executor.reshape(make_bars_params.cl_alloc, bars.imgs, &.{ barcode_size, barcode_size, 1, make_bars_params.num_images });
    const expected = switch (target) {
        .bbox => bars.box_labels,
        .bars => bars.bars,
    };

    return .{
        .bars = bars,
        .input = batch_cl_4d,
        .expected = expected,
    };
}

fn logBboxLosses(logger: anytype, box_stats: training_stats.BboxLosses) !void {
    inline for (std.meta.fields(@TypeOf(box_stats))) |field| {
        const val_opt: ?f32 = @field(box_stats, field.name);
        if (val_opt) |val| {
            try logger.print("{s} loss,{d}\n", .{ field.name, val });
        }
    }
}

pub fn logBboxVal(logger: anytype, val_stats: training_stats.BboxValidationData) !void {
    inline for (std.meta.fields(@TypeOf(val_stats))) |field| {
        const segmented_stats: ?training_stats.SegmentedStats = @field(val_stats, field.name);
        if (segmented_stats) |s| {
            inline for (std.meta.fields(training_stats.SegmentedStats)) |stat_field| {
                const val: f32 = @field(s, stat_field.name);
                try logger.print("{s} {s},{d}\n", .{ field.name, stat_field.name, val });
            }
        }
    }
}

fn trainThread(channels: *SharedChannels, background_dir: []const u8, config: Config, out_dir: std.fs.Dir, initial_checkpoint_path: ?[]const u8) !void {
    defer {
        channels.train_to_gui.send(.shutdown) catch {
            std.log.err("Failed to notify thread death", .{});
        };
    }
    const cl_alloc_buf = try std.heap.page_allocator.alloc(u8, 1000 * 1024 * 1024);
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

    const checkpoint: ?[]nn.LayerWeights(math.TracingExecutor) = if (initial_checkpoint_path) |p|
        try nn_checkpoint.loadCheckpoint(&cl_alloc, &tracing_executor, p)
    else
        null;

    const layers = try nn.modelFromConfig(&cl_alloc, &tracing_executor, &initializers, config.network.layers, checkpoint);

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

    const validation_set = try generateTrainingInput(
        &barcode_gen,
        .{
            .cl_alloc = &cl_alloc,
            .rand_params = config.data.val_rand_params,
            .rand_source = &rand_source,
            .num_images = config.val_size,
            .label_in_frame = config.data.label_in_frame,
            .label_iou = config.data.label_iou,
            .enable_backgrounds = config.data.enable_backgrounds,
        },
        math_executor,
        config.train_target,
        config.data.img_size,
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
                    &barcode_gen,
                    .{
                        .cl_alloc = &cl_alloc,
                        .rand_params = config.data.rand_params,
                        .rand_source = &rand_source,
                        .num_images = config.data.batch_size,
                        .label_in_frame = config.data.label_in_frame,
                        .label_iou = config.data.label_iou,
                        .enable_backgrounds = config.data.enable_backgrounds,
                    },
                    math_executor,
                    config.train_target,
                    config.data.img_size,
                );

                const results = try nn.runLayers(&cl_alloc, train_input.input, layers, &tracing_executor);
                try notifier.notifyLayerOutputs(results);
                try notifier.predictionsQueued(results[results.len - 1]);

                if (config.train_target == .bbox) {
                    try barcode_gen.healBboxLabels(&cl_alloc, train_input.expected, tracing_executor.getClTensor(results[results.len - 1].buf), config.data.label_iou, config.disable_bbox_loss_if_out_of_frame);
                }
                try notifier.batchGenerationQueued(train_input.bars.imgs, train_input.expected);

                const traced_expected = try tracing_executor.appendNode(train_input.expected, .init);

                const loss = blk: switch (config.train_target) {
                    .bbox => {
                        const even_loss = try tracing_executor.squaredErr(&cl_alloc, results[results.len - 1], traced_expected);
                        const loss_multipliers = try math_executor.createTensorUntracked(&cl_alloc, config.loss_multipliers, &.{@intCast(config.loss_multipliers.len)});
                        break :blk try tracing_executor.elemMul(&cl_alloc, even_loss, loss_multipliers);
                    },
                    .bars => try tracing_executor.bceWithLogits(&cl_alloc, results[results.len - 1], traced_expected),
                };

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
                    switch (config.train_target) {
                        .bbox => {
                            const enabled_labels = training_stats.EnabledLabels{
                                .in_frame = config.data.label_in_frame,
                                .iou = config.data.label_iou,
                            };
                            const box_stats = try training_stats.extractBboxLosses(
                                math.TracingExecutor,
                                &cl_alloc,
                                enabled_labels,
                                &tracing_executor,
                                loss,
                            );

                            try logBboxLosses(log_writer, box_stats);

                            if (iter % config.val_freq == 0) {
                                const layer_outputs = try nn.runLayers(&cl_alloc, validation_set.input, layers, &tracing_executor);
                                const val_results = tracing_executor.getClTensor(layer_outputs[layer_outputs.len - 1].buf);
                                try barcode_gen.healBboxLabels(&cl_alloc, validation_set.expected, val_results, config.data.label_iou, config.disable_bbox_loss_if_out_of_frame);
                                const val_stats = try training_stats.calcBboxValidationData(
                                    &cl_alloc,
                                    enabled_labels,
                                    &barcode_gen,
                                    math_executor,
                                    val_results,
                                    validation_set.expected,
                                );
                                try logBboxVal(log_writer, val_stats);
                            }
                        },
                        .bars => {
                            var total_loss: f32 = 0;
                            for (loss_cpu) |v| {
                                total_loss += v;
                            }

                            try log_writer.print("loss,{d}\n", .{total_loss});

                            if (iter % config.val_freq == 0) {
                                const val_results = try nn.runLayers(&cl_alloc, validation_set.input, layers, &tracing_executor);
                                const val_result_cpu = try tracing_executor.toCpu(cl_alloc.heap(), &cl_alloc, val_results[val_results.len - 1]);
                                const expected_cpu = try math_executor.toCpu(cl_alloc.heap(), &cl_alloc, validation_set.expected);

                                var correct_bars: usize = 0;
                                var correct_codes: usize = 0;
                                const bars_per_code = val_results[val_results.len - 1].dims.get(0);
                                for (0..config.val_size) |img_idx| {
                                    var code_correct = true;
                                    for (0..bars_per_code) |bar_idx| {
                                        const predicted = val_result_cpu[bars_per_code * img_idx + bar_idx];
                                        const actual = expected_cpu[bars_per_code * img_idx + bar_idx];
                                        const predicted_true = predicted > 0;
                                        const actual_true = actual > 0.5;

                                        if (predicted_true == actual_true) {
                                            correct_bars += 1;
                                        } else {
                                            code_correct = false;
                                        }
                                    }

                                    if (code_correct) {
                                        correct_codes += 1;
                                    }
                                }

                                const bars_per_code_f: f32 = @floatFromInt(bars_per_code);
                                const val_size_f: f32 = @floatFromInt(config.val_size);
                                try log_writer.print("correct bars,{d}\n", .{correct_bars});
                                try log_writer.print("correct bars ratio,{d}\n", .{@as(f32, @floatFromInt(correct_bars)) / val_size_f / bars_per_code_f});

                                try log_writer.print("correct codes,{d}\n", .{correct_codes});
                                try log_writer.print("correct codes ratio,{d}\n", .{@as(f32, @floatFromInt(correct_codes)) / val_size_f});
                            }
                        },
                    }

                    try buf_writer.flush();
                }

                if (iter % config.checkpoint_freq == 0) {
                    const checkpoint_name = try std.fmt.allocPrint(cl_alloc.heap(), "checkpoint_{d}", .{iter});
                    try nn_checkpoint.write(&cl_alloc, &tracing_executor, out_dir, checkpoint_name, layers);
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
    checkpoint: ?[]const u8,

    const Switch = enum {
        @"--background-dir",
        @"--config",
        @"--out-dir",
        @"--checkpoint",
    };

    fn parse(alloc: std.mem.Allocator) !Args {
        var it = try std.process.argsWithAllocator(alloc);

        const process_name = it.next() orelse "sphmelly";

        var background_dir: ?[]const u8 = null;
        var config: ?[]const u8 = null;
        var out_dir: ?[]const u8 = null;
        var checkpoint: ?[]const u8 = null;

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
                .@"--checkpoint" => checkpoint = it.next() orelse {
                    std.log.err("Missing checkpoint arg", .{});
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
            .checkpoint = checkpoint,
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
            \\Optional args:
            \\--checkpoint: Initial weights to load
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
    try ui.initPinned(&allocators, config.network.lr, config.data.batch_size, config.train_target);

    var comms = SharedChannels{};

    const train_thread = try std.Thread.spawn(.{}, trainThread, .{ &comms, args.background_dir, config, out_dir, args.checkpoint });
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

        while (comms.train_to_gui.poll()) |response_in| {
            // Make sure nothing is referencing old data

            const response = switch (response_in) {
                .gui_data => |r| r,
                .shutdown => return,
            };

            defer req_alloc_buf.swap();

            num_completed_reqs += 1;
            std.debug.assert(num_completed_reqs == outstanding_req);

            try ui.clear();

            ui.params.current_loss = response.loss;

            if (response.train_sample) |sample| {
                try ui.setImageGrayscale(allocators.scratch.allocator(), sample.img);

                switch (config.train_target) {
                    .bbox => {
                        try ui.renderBBoxOverlay(&allocators.scratch_gl, sample.expected, .{ 1, 0, 0 });
                        try ui.renderBBoxOverlay(&allocators.scratch_gl, sample.prediction, .{ 0, 0, 1 });
                    },
                    .bars => {
                        try ui.renderBarComparison(&allocators.scratch_gl, sample.prediction, sample.expected);
                    },
                }

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
