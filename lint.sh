#!/usr/bin/env bash

set -ex

zig fmt --check src build.zig build.zig.zon
zig build -Dextras
./zig-out/bin/test
