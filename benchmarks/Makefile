# =============================================================================
# AppleSiliconFFT
# Copyright (c) 2026 Mohamed Amine Bergach <mbergach@illumina.com>
# Licensed under the MIT License. See LICENSE file in the project root.
# =============================================================================

SWIFT = swiftc
SWIFTFLAGS = -O -framework Metal -framework Foundation -framework Accelerate
SRCDIR = Sources
BUILDDIR = build
SOURCES = $(SRCDIR)/memory_bandwidth_bench.swift $(SRCDIR)/utils.swift $(SRCDIR)/main.swift
VDSP_SRC = $(SRCDIR)/vdsp_baseline_bench.swift

# Include vDSP benchmark if it exists
ifneq (,$(wildcard $(VDSP_SRC)))
    SOURCES += $(VDSP_SRC)
endif

TARGET = $(BUILDDIR)/fft_microbench

.PHONY: all clean run

all: $(TARGET)
	@cp $(SRCDIR)/memory_bandwidth.metal $(BUILDDIR)/

$(BUILDDIR):
	@mkdir -p $(BUILDDIR)

$(TARGET): $(SOURCES) $(SRCDIR)/memory_bandwidth.metal | $(BUILDDIR)
	$(SWIFT) $(SWIFTFLAGS) -o $@ $(SOURCES)
	@cp $(SRCDIR)/memory_bandwidth.metal $(BUILDDIR)/

run: $(TARGET)
	@$(TARGET) $(ARGS)

clean:
	rm -rf $(BUILDDIR)

# Individual benchmark targets
bench-tgmem: $(TARGET)
	@$(TARGET) 1

bench-shuffle: $(TARGET)
	@$(TARGET) 2

bench-regcopy: $(TARGET)
	@$(TARGET) 3

bench-occupancy: $(TARGET)
	@$(TARGET) 4

bench-threadcount: $(TARGET)
	@$(TARGET) 5

bench-vdsp: $(TARGET)
	@$(TARGET) 6
