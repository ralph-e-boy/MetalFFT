DOCS_DIR = docs

.PHONY: docs docs-llm test clean-docs

docs:
	swift package generate-documentation \
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
