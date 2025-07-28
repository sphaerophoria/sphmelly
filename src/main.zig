const std = @import("std");
const sphtud = @import("sphtud");
const cl = @cImport({
    @cInclude("CL/cl.h");
    @cInclude("CL/cl_ext.h");
});

const opencl_program = @embedFile("test.cl");
const sum_program_content = @embedFile("sum.cl");

const TwoParam = struct {
    in1: *TrackedF32,
    in2: *TrackedF32,
};
const Operation = union(enum) {
    init,
    add: TwoParam,
    sub: TwoParam,
    mul: TwoParam,
    sigmoid: *TrackedF32,
    pow: struct {
        in1: *TrackedF32,
        in2: f32,
    },
};

const TrackedF32 = struct {
    val: f32,
    op: Operation,
    gradient: f32 = 0.0,

    fn init(alloc: std.mem.Allocator, val: f32) !*TrackedF32 {
        const ret = try alloc.create(TrackedF32);
        ret.* = .{
            .val = val,
            .op = .init,
        };
        return ret;
    }

    fn add(alloc: std.mem.Allocator, a: *TrackedF32, b: *TrackedF32) !*TrackedF32 {
        const ret = try alloc.create(TrackedF32);
        ret.* = .{
            .val = a.val + b.val,
            .op = .{
                .add = .{
                    .in1 = a,
                    .in2 = b,
                },
            },
        };
        return ret;
    }

    fn sub(alloc: std.mem.Allocator, a: *TrackedF32, b: *TrackedF32) !*TrackedF32 {
        const ret = try alloc.create(TrackedF32);
        ret.* = .{
            .val = a.val - b.val,
            .op = .{
                .sub = .{
                    .in1 = a,
                    .in2 = b,
                },
            },
        };
        return ret;
    }

    fn mul(alloc: std.mem.Allocator, a: *TrackedF32, b: *TrackedF32) !*TrackedF32 {
        const ret = try alloc.create(TrackedF32);
        ret.* = .{
            .val = a.val * b.val,
            .op = .{
                .mul = .{
                    .in1 = a,
                    .in2 = b,
                },
            },
        };
        return ret;
    }

    fn sigmoid(alloc: std.mem.Allocator, a: *TrackedF32) !*TrackedF32 {
        const ret = try alloc.create(TrackedF32);
        ret.* = .{
            .val = 1.0 / (1.0 + std.math.exp(-a.val)),
            .op = .{ .sigmoid = a },
        };
        return ret;
    }

    fn pow(alloc: std.mem.Allocator, a: *TrackedF32, b: f32) !*TrackedF32 {
        const ret = try alloc.create(TrackedF32);
        ret.* = .{
            .val = std.math.pow(f32, a.val, b),
            .op = .{
                .pow = .{
                    .in1 = a,
                    .in2 = b,
                },
            },
        };
        return ret;
    }

    fn backprop(self: *TrackedF32, downstream_gradient: f32) void {
        switch (self.op) {
            .init => return,
            .add => |params| {
                const in1_grad = 1.0 * downstream_gradient;
                const in2_grad = 1.0 * downstream_gradient;

                params.in1.gradient += in1_grad;
                params.in2.gradient += in2_grad;

                params.in1.backprop(in1_grad);
                params.in2.backprop(in2_grad);
            },
            .mul => |params| {
                const in1_grad = params.in2.val * downstream_gradient;
                const in2_grad = params.in1.val * downstream_gradient;

                params.in1.gradient += in1_grad;
                params.in2.gradient += in2_grad;

                params.in1.backprop(in1_grad);
                params.in2.backprop(in2_grad);
            },
            .pow => |params| {
                const grad = params.in1.val * params.in2 * downstream_gradient;
                params.in1.gradient += grad;
                params.in1.backprop(grad);
            },
            .sub => |params| {
                const in1_grad = 1.0 * downstream_gradient;
                const in2_grad = -1.0 * downstream_gradient;

                params.in1.gradient += in1_grad;
                params.in2.gradient += in2_grad;

                params.in1.backprop(in1_grad);
                params.in2.backprop(in2_grad);
            },
            .sigmoid => |input| {
                const x = input.val;
                const enx = std.math.exp(-x);
                const enxp1 = (enx + 1);

                const grad = enx / enxp1 / enxp1 * downstream_gradient;
                input.gradient += grad;
                input.backprop(grad);
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

fn clError(ret: cl.cl_int) !void {
    switch (ret) {
        cl.CL_SUCCESS => return,
        cl.CL_PLATFORM_NOT_FOUND_KHR => return error.PlatformNotFound,
        cl.CL_INVALID_VALUE => return error.InvalidValue,
        cl.CL_MEM_OBJECT_ALLOCATION_FAILURE, cl.CL_OUT_OF_HOST_MEMORY => return error.OutOfMemory,
        cl.CL_INVALID_PLATFORM => return error.InvalidPlatform,
        cl.CL_INVALID_DEVICE_TYPE => return error.InvalidDeviceType,
        cl.CL_DEVICE_NOT_FOUND => return error.DeviceNotFound,
        cl.CL_OUT_OF_RESOURCES => return error.OutOfResources,
        cl.CL_INVALID_CONTEXT => return error.InvalidContext,
        cl.CL_INVALID_PROPERTY => return error.InvalidProperty,
        cl.CL_INVALID_DEVICE => return error.InvalidDevice,
        cl.CL_INVALID_BUFFER_SIZE => return error.InvalidBufferSize,
        cl.CL_INVALID_HOST_PTR => return error.InvalidHostPointer,
        cl.CL_INVALID_OPERATION => return error.InvalidOperation,
        else => return error.UnknownError,
    }
}

const ClAlloc = struct {
    mem: sphtud.util.RuntimeSegmentedList(cl.cl_mem),
    events: sphtud.util.RuntimeSegmentedList(cl.cl_event),
    programs: sphtud.util.RuntimeSegmentedList(cl.cl_program),
    kernels: sphtud.util.RuntimeSegmentedList(cl.cl_kernel),

    fn init(alloc: std.mem.Allocator) !ClAlloc {
        const small_size = 100;
        const max_size = 1000000;

        // FIXME: No way to force no deallocations on RuntimeSegmentedList
        return .{
            .mem = try .init(alloc, alloc, small_size, max_size),
            .events = try .init(alloc, alloc, small_size, max_size),
            .programs = try .init(alloc, alloc, small_size, max_size),
            .kernels = try .init(alloc, alloc, small_size, max_size),
        };
    }

    fn reset(self: *ClAlloc) void {
        self.resetMem();
        self.resetEvents();
        self.resetPrograms();
        self.resetKernels();
    }

    fn resetMem(self: *ClAlloc) void {
        var it = self.mem.iter();
        while (it.next()) |obj| {
            _ = cl.clReleaseMemObject(obj.*);
        }
        self.mem.shrink(0);
    }

    fn resetEvents(self: *ClAlloc) void {
        var it = self.events.iter();
        while (it.next()) |obj| {
            _ = cl.clReleaseEvent(obj.*);
        }
        self.events.shrink(0);
    }

    fn resetPrograms(self: *ClAlloc) void {
        var it = self.programs.iter();
        while (it.next()) |obj| {
            _ = cl.clReleaseProgram(obj.*);
        }
        self.programs.shrink(0);
    }

    fn resetKernels(self: *ClAlloc) void {
        var it = self.kernels.iter();
        while (it.next()) |obj| {
            _ = cl.clReleaseKernel(obj.*);
        }
        self.kernels.shrink(0);
    }
};

const OpenClExecutor = struct {
    context: cl.cl_context,
    device: cl.cl_device_id,
    queue: cl.cl_command_queue,

    const Buffer = struct {
        buf: cl.cl_mem,
    };

    const Event = struct {
        event: cl.cl_event,

        fn wait(self: Event) !void {
            try clError(cl.clWaitForEvents(1, &self.event));
        }
    };

    fn init() !OpenClExecutor {
        var platform: cl.cl_platform_id = undefined;
        try clError(cl.clGetPlatformIDs(1, &platform, null));

        var platform_name: [100]u8 = undefined;
        try clError(cl.clGetPlatformInfo(platform, cl.CL_PLATFORM_VENDOR, 100, &platform_name, null));

        var device_id: cl.cl_device_id = undefined;
        try clError(cl.clGetDeviceIDs(platform, cl.CL_DEVICE_TYPE_GPU, 1, &device_id, null));

        var compute_units: cl.cl_uint = undefined;
        try clError(cl.clGetDeviceInfo(device_id, cl.CL_DEVICE_MAX_COMPUTE_UNITS, 4, &compute_units, null));

        var cl_err_out: cl.cl_int = undefined;
        const context = cl.clCreateContext(null, 1, &device_id, null, null, &cl_err_out);
        try clError(cl_err_out);

        const queue = cl.clCreateCommandQueueWithProperties(
            context,
            device_id,
            null,
            &cl_err_out,
        );

        try clError(cl_err_out);

        return .{
            .context = context,
            .device = device_id,
            .queue = queue,
        };
    }

    fn deinit(self: OpenClExecutor) void {
        _ = cl.clReleaseContext(self.context);
    }

    const Kernel = struct {
        kernel: cl.cl_kernel,

        const Arg = union(enum) {
            local_mem: u32,
            buf: Buffer,
            uint: u32,

            fn size(self: Arg) cl.cl_uint {
                switch (self) {
                    .buf => return @sizeOf(cl.cl_mem),
                    .uint => return @sizeOf(u32),
                    .local_mem => |s| return s,
                }
            }

            fn ptr(self: *const Arg) ?*const anyopaque {
                switch (self.*) {
                    .buf => |*b| return @ptrCast(&b.buf),
                    .uint => |*u| return u,
                    .local_mem => return null,
                }
            }
        };

        fn setArg(self: Kernel, idx: cl.cl_uint, arg: Arg) !void {
            const ret = cl.clSetKernelArg(self.kernel, idx, arg.size(), arg.ptr());
            try clError(ret);
        }
    };

    fn createBuffer(self: OpenClExecutor, alloc: *ClAlloc, hint: BufferHint, size: usize) !Buffer {
        var ret: cl.cl_int = undefined;
        const buf = cl.clCreateBuffer(self.context, hint.toCl(), size, null, &ret);
        try clError(ret);

        errdefer _ = cl.clReleaseMemObject(buf);
        try alloc.mem.append(buf);

        return .{
            .buf = buf,
        };
    }

    const Program = struct {
        program: cl.cl_program,

        fn createKernel(self: Program, alloc: *ClAlloc, name: [:0]const u8) !Kernel {
            var ret: cl.cl_int = undefined;
            const kernel = cl.clCreateKernel(self.program, name.ptr, &ret);
            try clError(ret);

            errdefer _ = cl.clReleaseKernel(kernel);
            try alloc.kernels.append(kernel);

            return .{
                .kernel = kernel,
            };
        }
    };

    fn createProgram(self: OpenClExecutor, alloc: *ClAlloc, source: []const u8) !Program {
        var ret: cl.cl_int = undefined;

        const program = cl.clCreateProgramWithSource(self.context, 1, @ptrCast(@constCast(&source.ptr)), &source.len, &ret);

        try clError(ret);

        clError(cl.clBuildProgram(program, 0, null, null, null, null)) catch |e| {
            var log: [4096]u8 = undefined;
            var log_len: usize = 0;
            _ = cl.clGetProgramBuildInfo(program, self.device, cl.CL_PROGRAM_BUILD_LOG, log.len, &log, &log_len);
            std.debug.print("{s}\n", .{log[0..log_len]});

            return e;
        };

        errdefer _ = cl.clReleaseProgram(program);

        try alloc.programs.append(program);

        return .{
            .program = program,
        };
    }

    const BufferHint = enum {
        read_only,
        read_write,
        write_only,

        fn toCl(self: BufferHint) cl.cl_mem_flags {
            switch (self) {
                .read_only => return cl.CL_MEM_READ_ONLY,
                .write_only => return cl.CL_MEM_WRITE_ONLY,
                .read_write => return cl.CL_MEM_READ_WRITE,
            }
        }
    };

    fn writeBufferUntracked(self: OpenClExecutor, buf: Buffer, content: []const u8) !void {
        try clError(cl.clEnqueueWriteBuffer(
            self.queue,
            buf.buf,
            cl.CL_FALSE,
            0,
            content.len,
            content.ptr,
            0,
            null,
            null,
        ));
    }

    fn writeBuffer(self: OpenClExecutor, alloc: *ClAlloc, buf: Buffer, content: []const u8) !Event {
        var event: cl.cl_event = undefined;
        try clError(cl.clEnqueueWriteBuffer(
            self.queue,
            buf.buf,
            cl.CL_FALSE,
            0,
            content.len,
            content.ptr,
            0,
            null,
            &event,
        ));

        errdefer cl.clReleaseEvent(event);
        try alloc.events.append(event);

        return .{
            .event = event,
        };
    }

    fn readBufferUntracked(self: OpenClExecutor, buf: Buffer, out: []u8) !void {
        try clError(cl.clEnqueueReadBuffer(
            self.queue,
            buf.buf,
            cl.CL_FALSE,
            0,
            out.len,
            out.ptr,
            0,
            null,
            null,
        ));
    }

    fn readBuffer(self: OpenClExecutor, alloc: *ClAlloc, buf: Buffer, out: []u8) !Event {
        var event: cl.cl_event = undefined;
        std.debug.print("{any}\n", .{out.ptr});
        try clError(cl.clEnqueueReadBuffer(
            self.queue,
            buf.buf,
            cl.CL_FALSE,
            0,
            out.len,
            out.ptr,
            0,
            null,
            &event,
        ));

        errdefer _ = cl.clReleaseEvent(event);
        try alloc.events.append(event);

        return .{
            .event = event,
        };
    }

    fn executeKernelUntracked(self: OpenClExecutor, kernel: Kernel, num_elems: usize) !void {
        const params = try self.getKernelParams(kernel, num_elems);

        std.debug.print("local size: {d}\n", .{params.local_size});
        try clError(cl.clEnqueueNDRangeKernel(
            self.queue,
            kernel.kernel,
            params.work_dim,
            params.offset,
            &params.global_size,
            &params.local_size,
            params.num_events_in_wait_list,
            params.wait_event_list,
            null,
        ));
    }

    fn executeKernel(self: OpenClExecutor, alloc: *ClAlloc, kernel: Kernel, num_elems: usize) !Event {
        const params = try self.getKernelParams(kernel, num_elems);

        var event: cl.cl_event = undefined;
        try clError(cl.clEnqueueNDRangeKernel(
            self.queue,
            kernel.kernel,
            params.work_dim,
            params.offset,
            &params.global_size,
            &params.local_size,
            params.num_events_in_wait_list,
            params.wait_event_list,
            &event,
        ));

        errdefer cl.clReleaseEvent(event);
        try alloc.events.append(event);

        return .{
            .event = event,
        };
    }

    const KernelParams = struct {
        global_size: usize,
        local_size: usize,
        work_dim: cl.cl_uint = 1,
        offset: cl.cl_uint = 0,
        num_events_in_wait_list: cl.cl_uint = 0,
        wait_event_list: [*c]const cl.cl_event = null,
    };

    fn getKernelParams(self: OpenClExecutor, kernel: Kernel, num_elems: usize) !KernelParams {
        const local_size = try self.getKernelLocalSize(kernel);

        const global_size = roundUp(usize, num_elems, local_size);
        return .{
            .global_size = global_size,
            .local_size = local_size,
        };
    }

    fn getKernelLocalSize(self: OpenClExecutor, kernel: Kernel) !usize {
        var local_size: usize = undefined;
        const ret = cl.clGetKernelWorkGroupInfo(
            kernel.kernel,
            self.device,
            cl.CL_KERNEL_WORK_GROUP_SIZE,
            @sizeOf(usize),
            &local_size,
            null,
        );
        try clError(ret);
        return local_size;
    }
};

const OpenClMath = struct {
    sum_kernel: OpenClExecutor.Kernel,
    executor: OpenClExecutor,

    fn init(cl_alloc: *ClAlloc, executor: OpenClExecutor) !OpenClMath {
        const sum_program = try executor.createProgram(cl_alloc, sum_program_content);
        const sum_kernel = try sum_program.createKernel(cl_alloc, "sum");

        return .{
            .sum_kernel = sum_kernel,
            .executor = executor,
        };
    }

    fn sum(self: OpenClMath, scratch: std.mem.Allocator, a: OpenClExecutor.Buffer, size: u32, log: bool) !f32 {
        var scratch_cl = try ClAlloc.init(scratch);
        defer scratch_cl.reset();

        const output = try self.executor.createBuffer(&scratch_cl, .write_only, 2 * @sizeOf(f32));
        const local_size = try self.executor.getKernelLocalSize(self.sum_kernel);
        try self.sum_kernel.setArg(0, .{ .buf = a });
        try self.sum_kernel.setArg(1, .{ .buf = output });
        try self.sum_kernel.setArg(2, .{ .local_mem = @intCast(local_size * @sizeOf(f32)) });
        try self.sum_kernel.setArg(3, .{ .uint = size });
        try self.sum_kernel.setArg(4, .{ .uint = if (log) 1 else 0 });

        try self.executor.executeKernelUntracked(self.sum_kernel, roundUp(u32, size, 2) / 2);

        const out_cpu = try scratch.alloc(f32, 2);
        const event = try self.executor.readBuffer(&scratch_cl, output, std.mem.sliceAsBytes(out_cpu));
        try event.wait();

        std.debug.print("{any}\n", .{out_cpu});
        return out_cpu[0];
    }
};

fn roundUp(comptime T: type, num: T, multiple: T) T {
    return num + ((multiple - (num % multiple)) % multiple);
}

test "roundUp" {
    try std.testing.expectEqual(roundUp(u32, 4, 4), 4);
    try std.testing.expectEqual(roundUp(u32, 7, 4), 8);
    try std.testing.expectEqual(roundUp(u32, 8, 4), 8);
}

fn moveToClAndSum(scratch: std.mem.Allocator, executor: OpenClExecutor, cl_math: OpenClMath, nums: []const f32, log: bool) !f32 {
    var scratch_cl = try ClAlloc.init(scratch);
    defer scratch_cl.reset();

    const a = try executor.createBuffer(&scratch_cl, .read_only, nums.len * @sizeOf(f32));

    try executor.writeBufferUntracked(a, std.mem.sliceAsBytes(nums));
    return try cl_math.sum(scratch, a, @intCast(nums.len), log);
}

test "opencl sum" {
    var arena = sphtud.alloc.BufAllocator.init(try std.heap.page_allocator.alloc(u8, 10 * 1024 * 1024));
    const executor = try OpenClExecutor.init();
    defer executor.deinit();

    var cl_alloc = try ClAlloc.init(arena.allocator());
    defer cl_alloc.reset();

    const cl_math = try OpenClMath.init(&cl_alloc, executor);
    const local_size = try executor.getKernelLocalSize(cl_math.sum_kernel);

    const list_sizes: []const struct { usize, bool } = &.{
        .{ 1, false },
        .{ 3, false },
        .{ 4, false },
        .{ local_size, false },
        .{ local_size * 2, false },
        .{ local_size * 2 - 7, false },
        .{ 9, true },
        .{ local_size * 2 - 1, false },
        //.{ local_size * 2 + 1, false },
    };
    for (list_sizes) |params| {
        const size = params[0];
        const log = params[1];
        const vals = try arena.allocator().alloc(f32, size);
        var expected_u: usize = 0;
        for (0..vals.len) |i| {
            expected_u += i;
            vals[i] = @floatFromInt(i);
        }
        const expected: f32 = @floatFromInt(expected_u);

        try std.testing.expectApproxEqAbs(expected, try moveToClAndSum(arena.allocator(), executor, cl_math, vals, log), 0.001);
    }
}

pub fn main() !void {
    var arena = sphtud.alloc.BufAllocator.init(try std.heap.page_allocator.alloc(u8, 10 * 1024 * 1024));

    const executor = try OpenClExecutor.init();
    defer executor.deinit();

    var cl_alloc = try ClAlloc.init(arena.allocator());
    defer cl_alloc.reset();

    //const program = try executor.createProgram(&cl_alloc, opencl_program);
    //const kernel = try program.createKernel(&cl_alloc, "add");

    const cl_math = try OpenClMath.init(&cl_alloc, executor);

    {
        const list_size = 4;
        const a = try executor.createBuffer(&cl_alloc, .read_only, list_size * @sizeOf(f32));
        //const b = try executor.createBuffer(&cl_alloc, .read_only, 4000);
        //const out_buf = try executor.createBuffer(&cl_alloc, .write_only, 4000);

        var a_cpu: [list_size]f32 = undefined;
        for (&a_cpu, 0..) |*out, i| {
            out.* = @floatFromInt(i);
        }

        try executor.writeBufferUntracked(a, std.mem.sliceAsBytes(&a_cpu));
        _ = try cl_math.sum(arena.allocator(), a, list_size, false);

        //var b_cpu: [1000]f32 = undefined;
        //for (&b_cpu, 0..) |*out, i| {
        //    out.* = @floatFromInt(i * 2);
        //}

        //try executor.writeBufferUntracked(b, std.mem.sliceAsBytes(&b_cpu));

        //try kernel.setArg(0, .{ .buf = a });
        //try kernel.setArg(1, .{ .buf = b });
        //try kernel.setArg(2, .{ .buf = out_buf });
        //try kernel.setArg(3, .{ .uint = @intCast(a_cpu.len) });

        //try executor.executeKernelUntracked(kernel, 1000);

        //var out_cpu: [1000]f32 = undefined;
        //const read_buffeer_event = try executor.readBuffer(&cl_alloc, out_buf, std.mem.sliceAsBytes(&out_cpu));
        //try read_buffeer_event.wait();

        //for (&out_cpu) |val| {
        //    std.debug.print("{d}\n", .{val});
        //}
    }

    //executor.writeBuffer(a, cpu_a);
    //executor.writeBuffer(b, cpu_b);

    //executor.add(a, b);

    if (true) return;

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

test {
    _ = std.testing.refAllDeclsRecursive(@This());
}
