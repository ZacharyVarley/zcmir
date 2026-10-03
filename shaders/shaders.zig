//! WGSL sources, embedded at compile time. The same files are served to the browser app.

/// SMI moments, maps and pose scores (+ the copula scores it calls), one module.
pub const smi = @embedFile("smi.wgsl") ++ "\n" ++ @embedFile("copula.wgsl");
pub const gls_mift = @embedFile("gls_mift.wgsl");
pub const pos_gift = @embedFile("pos_gift.wgsl");
pub const ncc = @embedFile("ncc.wgsl");
pub const ffd = @embedFile("ffd.wgsl");
/// Refine on the GPU: the climb's decisions between the score kernels.
pub const climb = @embedFile("climb.wgsl");
/// SIM(2) sweep (+ the copula scores its combine calls).
pub const sweep = @embedFile("sweep.wgsl") ++ "\n" ++ @embedFile("copula.wgsl");
