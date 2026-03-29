// =============================================================================
// AppleSiliconFFT
// Copyright (c) 2026 Mohamed Amine Bergach <mbergach@illumina.com>
// Licensed under the MIT License. See LICENSE file in the project root.
// =============================================================================

import Foundation

setbuf(stdout, nil)
setbuf(stderr, nil)

do {
    try benchmarkMain()
} catch {
    fputs("FATAL: \(error)\n", stderr)
    exit(1)
}
