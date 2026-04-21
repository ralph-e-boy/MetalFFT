DOCS_DIR  = docs
BIN_DIR   = bin
SWIFTC    = swiftc -O -framework Metal -framework Accelerate
SWIFTC_LIB = $(SWIFTC) -parse-as-library

.PHONY: docs docs-llm test clean-docs demo demo-fft demo-ct demo-batched demo-multisize demo-radar

docs:
	@mkdir -p $(DOCS_DIR)
	swift package --allow-writing-to-directory $(DOCS_DIR) \
		generate-documentation \
		--target MetalFFT \
		--output-path $(DOCS_DIR)/MetalFFT.doccarchive

docs-llm:
	@mkdir -p $(DOCS_DIR)
	swift build --target MetalFFT \
		-Xswiftc -enable-library-evolution \
		-Xswiftc -emit-module-interface-path \
		-Xswiftc $(PWD)/$(DOCS_DIR)/MetalFFT.swiftinterface \
		-Xswiftc -no-verify-emitted-module-interface
	@echo "Swift interface written to $(DOCS_DIR)/MetalFFT.swiftinterface"

test:
	swift test

clean-docs:
	rm -rf $(DOCS_DIR)

# ---------------------------------------------------------------------------
# Demos (src/ research kernels)
# Binaries go in bin/; Metal sources are copied alongside each binary since
# the host programs locate .metal files relative to their executable path.
# ---------------------------------------------------------------------------

$(BIN_DIR):
	@mkdir -p $(BIN_DIR) $(BIN_DIR)/radar

$(BIN_DIR)/fft_host: src/fft_host.swift | $(BIN_DIR)
	$(SWIFTC_LIB) -o $@ $<
	@cp src/fft_stockham_4096.metal $(BIN_DIR)/
	@cp src/fft_4096_radix8.metal $(BIN_DIR)/fft_4096_mma.metal

$(BIN_DIR)/fft_ct_host: src/fft_ct_mma_host.swift | $(BIN_DIR)
	$(SWIFTC_LIB) -o $@ $<
	@cp src/fft_4096_ct_mma.metal $(BIN_DIR)/

$(BIN_DIR)/fft_batched_host: src/fft_batched_host.swift | $(BIN_DIR)
	$(SWIFTC_LIB) -o $@ $<
	@cp src/fft_4096_batched.metal $(BIN_DIR)/

$(BIN_DIR)/fft_multi_host: src/fft_multisize_host.swift | $(BIN_DIR)
	$(SWIFTC_LIB) -o $@ $<
	@cp src/fft_multisize.metal $(BIN_DIR)/

$(BIN_DIR)/radar/sar: src/radar/main.swift src/radar/sar_simulator.swift \
                      src/radar/rda_pipeline.swift src/radar/rda_fused_pipeline.swift \
                      src/radar/radar_metrics.swift src/radar/precision_comparison.swift | $(BIN_DIR)
	$(SWIFTC) -o $@ $^
	@cp src/radar/rda_kernels.metal src/fft_sar_fused.metal src/fft_multisize.metal $(BIN_DIR)/radar/

demo-fft: $(BIN_DIR)/fft_host
	@echo "\n=== Radix-4 Stockham FFT (113.6 GFLOPS target) ==="
	$(BIN_DIR)/fft_host

demo-ct: $(BIN_DIR)/fft_ct_host
	@echo "\n=== CT DIF + simdgroup MMA (128 GFLOPS target) ==="
	$(BIN_DIR)/fft_ct_host

demo-batched: $(BIN_DIR)/fft_batched_host
	@echo "\n=== Batched FFT ==="
	$(BIN_DIR)/fft_batched_host

demo-multisize: $(BIN_DIR)/fft_multi_host
	@echo "\n=== Multi-size FFT (N=256–16384) ==="
	$(BIN_DIR)/fft_multi_host

demo-radar: $(BIN_DIR)/radar/sar
	@echo "\n=== SAR Range-Doppler (fused vs unfused) ==="
	$(BIN_DIR)/radar/sar 4096 --fused

demo: demo-fft demo-ct demo-batched demo-multisize demo-radar
