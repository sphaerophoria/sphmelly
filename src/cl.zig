const std = @import("std");
const sphtud = @import("sphtud");
const math = @import("math.zig");

const cl = @cImport({
    @cInclude("CL/cl.h");
    @cInclude("CL/cl_ext.h");
});

pub const Alloc = struct {
    buf_alloc: sphtud.alloc.BufAllocator,

    mem: sphtud.util.RuntimeSegmentedList(cl.cl_mem),
    events: sphtud.util.RuntimeSegmentedList(cl.cl_event),
    programs: sphtud.util.RuntimeSegmentedList(cl.cl_program),
    kernels: sphtud.util.RuntimeSegmentedList(cl.cl_kernel),

    pub fn initPinned(self: *Alloc, buf: []u8) !void {
        const small_size = 100;
        const max_size = 1000000;

        self.buf_alloc = sphtud.alloc.BufAllocator.init(buf);
        const alloc = self.buf_alloc.allocator();

        self.mem = try .init(alloc, alloc, small_size, max_size);
        self.events = try .init(alloc, alloc, small_size, max_size);
        self.programs = try .init(alloc, alloc, small_size, max_size);
        self.kernels = try .init(alloc, alloc, small_size, max_size);
    }

    pub fn deinit(self: *Alloc) void {
        self.resetMem(0);
        self.resetEvents(0);
        self.resetPrograms(0);
        self.resetKernels(0);
    }

    pub const Checkpoint = struct {
        heap: sphtud.alloc.BufAllocator.Checkpoint,
        mem: usize,
        events: usize,
        programs: usize,
        kernels: usize,
    };

    pub fn checkpoint(self: Alloc) Checkpoint {
        return .{
            .heap = self.buf_alloc.checkpoint(),
            .mem = self.mem.len,
            .events = self.events.len,
            .programs = self.programs.len,
            .kernels = self.kernels.len,
        };
    }

    pub fn reset(self: *Alloc, restore_point: Checkpoint) void {
        self.resetMem(restore_point.mem);
        self.resetEvents(restore_point.events);
        self.resetPrograms(restore_point.programs);
        self.resetKernels(restore_point.kernels);
        self.buf_alloc.restore(restore_point.heap);
    }

    pub fn heap(self: *Alloc) std.mem.Allocator {
        return self.buf_alloc.allocator();
    }

    fn resetMem(self: *Alloc, restore_point: usize) void {
        var it = self.mem.iterFrom(restore_point);
        while (it.next()) |obj| {
            _ = cl.clReleaseMemObject(obj.*);
        }
        self.mem.shrink(restore_point);
    }

    fn resetEvents(self: *Alloc, restore_point: usize) void {
        var it = self.events.iterFrom(restore_point);
        while (it.next()) |obj| {
            _ = cl.clReleaseEvent(obj.*);
        }
        self.events.shrink(restore_point);
    }

    fn resetPrograms(self: *Alloc, restore_point: usize) void {
        var it = self.programs.iterFrom(restore_point);
        while (it.next()) |obj| {
            _ = cl.clReleaseProgram(obj.*);
        }
        self.programs.shrink(restore_point);
    }

    fn resetKernels(self: *Alloc, restore_point: usize) void {
        var it = self.kernels.iterFrom(restore_point);
        while (it.next()) |obj| {
            _ = cl.clReleaseKernel(obj.*);
        }
        self.kernels.shrink(restore_point);
    }
};

const KernelTimers = struct {
    items: sphtud.util.RuntimeBoundedArray(Item),

    const Item = struct {
        kernel_name: []const u8,
        event: Executor.Event,
    };

    pub fn init(alloc: std.mem.Allocator, max_elems: usize) !KernelTimers {
        return .{
            .items = try .init(alloc, max_elems),
        };
    }

    pub fn append(self: *KernelTimers, kernel_name: []const u8, event: Executor.Event) !void {
        try self.items.append(.{
            .kernel_name = kernel_name,
            .event = event,
        });
    }

    pub fn reset(self: *KernelTimers) void {
        self.items.resize(0) catch unreachable;
    }

    pub fn times(self: *const KernelTimers, alloc: std.mem.Allocator) !std.StringHashMapUnmanaged(u64) {
        var ret = std.StringHashMapUnmanaged(u64){};

        for (self.items.items) |item| {
            var start: cl.cl_ulong = 0;
            var end: cl.cl_ulong = 0;
            try clError(cl.clGetEventProfilingInfo(
                item.event.event,
                cl.CL_PROFILING_COMMAND_START,
                @sizeOf(cl.cl_ulong),
                &start,
                null,
            ));
            try clError(cl.clGetEventProfilingInfo(
                item.event.event,
                cl.CL_PROFILING_COMMAND_END,
                @sizeOf(cl.cl_ulong),
                &end,
                null,
            ));

            const gop = try ret.getOrPut(alloc, item.kernel_name);
            if (!gop.found_existing) {
                gop.key_ptr.* = item.kernel_name;
                gop.value_ptr.* = 0;
            }

            gop.value_ptr.* += end - start;
        }

        return ret;
    }
};

pub const Executor = struct {
    context: cl.cl_context,
    device: cl.cl_device_id,
    queue: cl.cl_command_queue,
    profiling_info: ProfilingInfo,

    const ProfilingInfo = union(enum) {
        profiling: KernelTimers,
        non_profiling,
    };

    pub const Buffer = struct {
        buf: cl.cl_mem,
    };

    pub const Event = struct {
        event: cl.cl_event,

        pub fn wait(self: Event) !void {
            try clError(cl.clWaitForEvents(1, &self.event));
        }
    };

    pub const ProfilingMode = enum {
        non_profiling,
        profiling,
    };

    pub fn init(alloc: std.mem.Allocator, profiling_mode: ProfilingMode) !Executor {
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

        const profiling_properties: []const cl.cl_queue_properties = &.{
            cl.CL_QUEUE_PROPERTIES, cl.CL_QUEUE_PROFILING_ENABLE,
            0,
        };

        const properties = switch (profiling_mode) {
            .profiling => profiling_properties.ptr,
            .non_profiling => null,
        };

        const queue = cl.clCreateCommandQueueWithProperties(
            context,
            device_id,
            properties,
            &cl_err_out,
        );

        try clError(cl_err_out);

        const profiling_info: ProfilingInfo = switch (profiling_mode) {
            .profiling => .{ .profiling = try .init(alloc, 100) },
            .non_profiling => .non_profiling,
        };
        return .{
            .context = context,
            .device = device_id,
            .queue = queue,
            .profiling_info = profiling_info,
        };
    }

    pub fn deinit(self: Executor) void {
        _ = cl.clReleaseContext(self.context);
    }

    pub fn resetTimers(self: *Executor) void {
        switch (self.profiling_info) {
            .profiling => |*timers| timers.reset(),
            .non_profiling => {},
        }
    }

    pub fn getProfilingInfo(self: *Executor, alloc: std.mem.Allocator) !std.StringHashMapUnmanaged(u64) {
        switch (self.profiling_info) {
            .profiling => |t| return try t.times(alloc),
            .non_profiling => return .{},
        }
    }

    pub const Kernel = struct {
        kernel: cl.cl_kernel,
        name: []const u8,

        pub const Arg = union(enum) {
            local_mem: u32,
            buf: Buffer,
            uint: u32,
            ulong: u64,
            float: f32,

            pub fn size(self: Arg) cl.cl_uint {
                switch (self) {
                    .buf => return @sizeOf(cl.cl_mem),
                    .uint => return @sizeOf(u32),
                    .ulong => return @sizeOf(u64),
                    .float => return @sizeOf(f32),
                    .local_mem => |s| return s,
                }
            }

            pub fn ptr(self: *const Arg) ?*const anyopaque {
                switch (self.*) {
                    .buf => |*b| return @ptrCast(&b.buf),
                    .uint => |*u| return u,
                    .ulong => |*u| return u,
                    .float => |*f| return f,
                    .local_mem => return null,
                }
            }
        };

        pub fn setArg(self: Kernel, idx: cl.cl_uint, arg: Arg) !void {
            const ret = cl.clSetKernelArg(self.kernel, idx, arg.size(), arg.ptr());
            try clError(ret);
        }
    };

    pub fn createBuffer(self: Executor, alloc: *Alloc, hint: BufferHint, size: usize) !Buffer {
        var ret: cl.cl_int = undefined;
        const buf = cl.clCreateBuffer(self.context, hint.toCl(), size, null, &ret);
        try clError(ret);

        errdefer _ = cl.clReleaseMemObject(buf);
        try alloc.mem.append(buf);

        return .{
            .buf = buf,
        };
    }

    pub fn fillBuffer(self: Executor, buf: Executor.Buffer, val: f32, size: usize) !void {
        comptime std.debug.assert(cl.cl_float == f32);

        try clError(
            cl.clEnqueueFillBuffer(
                self.queue,
                buf.buf,
                @ptrCast(&val),
                @sizeOf(f32),
                0,
                size,
                0,
                null,
                null,
            ),
        );
    }

    pub const Program = struct {
        program: cl.cl_program,

        pub fn createKernel(self: Program, alloc: *Alloc, name: [:0]const u8) !Kernel {
            var ret: cl.cl_int = undefined;
            const kernel = cl.clCreateKernel(self.program, name.ptr, &ret);
            try clError(ret);

            errdefer _ = cl.clReleaseKernel(kernel);
            try alloc.kernels.append(kernel);

            return .{
                .kernel = kernel,
                .name = name,
            };
        }
    };

    pub fn createProgram(self: Executor, alloc: *Alloc, source: []const u8) !Program {
        var ret: cl.cl_int = undefined;

        const program = cl.clCreateProgramWithSource(self.context, 1, @ptrCast(@constCast(&source.ptr)), &source.len, &ret);

        try clError(ret);

        clError(cl.clBuildProgram(program, 0, null, "-cl-std=CL2.0", null, null)) catch |e| {
            var log: [4096]u8 = undefined;
            var log_len: usize = 0;
            _ = cl.clGetProgramBuildInfo(program, self.device, cl.CL_PROGRAM_BUILD_LOG, log.len, &log, &log_len);
            std.debug.print("{s}\n", .{log[0..@min(log_len, log.len)]});

            return e;
        };

        errdefer _ = cl.clReleaseProgram(program);

        try alloc.programs.append(program);

        return .{
            .program = program,
        };
    }

    pub const BufferHint = enum {
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

    pub fn writeBufferUntracked(self: Executor, buf: Buffer, offset: usize, content: []const u8) !void {
        try clError(cl.clEnqueueWriteBuffer(
            self.queue,
            buf.buf,
            cl.CL_FALSE,
            offset,
            content.len,
            content.ptr,
            0,
            null,
            null,
        ));
    }

    pub fn writeBuffer(self: Executor, alloc: *Alloc, buf: Buffer, offset: usize, content: []const u8) !Event {
        var event: cl.cl_event = undefined;
        try clError(cl.clEnqueueWriteBuffer(
            self.queue,
            buf.buf,
            cl.CL_FALSE,
            offset,
            content.len,
            content.ptr,
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

    pub fn readBufferUntracked(self: Executor, buf: Buffer, offset: usize, out: []u8) !void {
        try clError(cl.clEnqueueReadBuffer(
            self.queue,
            buf.buf,
            cl.CL_FALSE,
            offset,
            out.len,
            out.ptr,
            0,
            null,
            null,
        ));
    }

    pub fn readBuffer(self: Executor, alloc: *Alloc, buf: Buffer, offset: usize, out: []u8) !Event {
        var event: cl.cl_event = undefined;
        try clError(cl.clEnqueueReadBuffer(
            self.queue,
            buf.buf,
            cl.CL_FALSE,
            offset,
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

    pub fn executeKernelUntracked(self: *Executor, alloc: *Alloc, kernel: Kernel, num_elems: usize, args: []const Kernel.Arg) !void {
        const params = try self.prepareKernel(kernel, num_elems, args);

        var event: cl.cl_event = undefined;
        const event_ptr: ?*cl.cl_event = switch (self.profiling_info) {
            .profiling => &event,
            .non_profiling => null,
        };

        try clError(cl.clEnqueueNDRangeKernel(
            self.queue,
            kernel.kernel,
            params.work_dim,
            params.offset,
            &params.global_size,
            &params.local_size,
            params.num_events_in_wait_list,
            params.wait_event_list,
            event_ptr,
        ));

        if (event_ptr) |p| {
            errdefer _ = cl.clReleaseEvent(event);
            try alloc.events.append(p.*);
        }

        switch (self.profiling_info) {
            .profiling => |*timers| {
                try timers.append(kernel.name, .{ .event = event_ptr.?.* });
            },
            .non_profiling => {},
        }
    }

    pub fn executeKernel(self: Executor, alloc: *Alloc, kernel: Kernel, num_elems: usize, args: []const Kernel.Arg) !Event {
        const params = try self.prepareKernel(kernel, num_elems, args);

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

        errdefer _ = cl.clReleaseEvent(event);
        try alloc.events.append(event);

        switch (self.profiling_info) {
            .profiling => |*timers| {
                timers.append(kernel.name, event);
            },
            .non_profiling => {},
        }

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

    fn prepareKernel(self: Executor, kernel: Kernel, num_elems: usize, args: []const Kernel.Arg) !KernelParams {
        const local_size = try self.getKernelLocalSize(kernel);

        const global_size = math.roundUp(usize, num_elems, local_size);

        for (args, 0..) |arg, i| {
            try kernel.setArg(@intCast(i), arg);
        }

        return .{
            .global_size = global_size,
            .local_size = local_size,
        };
    }

    pub fn getKernelLocalSize(self: Executor, kernel: Kernel) !usize {
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

    pub fn finish(self: Executor) !void {
        try clError(cl.clFinish(self.queue));
    }
};

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
        // Oops, we got a little lazy here
        else => return error.UnknownError,
    }
}
