// Shannon-calibrated per-shift scores on normal-score features [z, z², z³, z⁴]
// (js/copula.js has the host twins and the derivation notes). Appended to smi.wgsl
// and sweep.wgsl. m: the 45 moment sums divided by n (m[0] unused), layout of smi.wgsl.
// Both return MI in nats, or COP_BAD where undefined.

fn cop_raw(m: ptr<function, array<f32, 45>>, p: u32, q: u32) -> f32 {
    if (p == 0u && q == 0u) { return 1.0; }
    if (q == 0u) { return (*m)[p]; }
    if (p == 0u) { return (*m)[4u + q]; }
    return (*m)[9u + (p - 1u) * 4u + (q - 1u)];
}

fn cop_binom(n: u32, k: u32) -> f32 {
    var t = array<f32, 15>(1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 3.0, 3.0, 1.0, 1.0, 4.0, 6.0, 4.0, 1.0);
    return t[n * (n + 1u) / 2u + k];
}

fn cop_pow(x: f32, k: u32) -> f32 {
    var r = 1.0;
    for (var i = 0u; i < k; i++) { r = r * x; }
    return r;
}

// Central moment μ_pq (p + q ≤ 4) from raw moments.
fn cop_mu(m: ptr<function, array<f32, 45>>, p: u32, q: u32, mx: f32, my: f32) -> f32 {
    var v = 0.0;
    for (var i = 0u; i <= p; i++) {
        for (var j = 0u; j <= q; j++) {
            v = v + cop_binom(p, i) * cop_binom(q, j) * cop_raw(m, i, j) * cop_pow(-mx, p - i) * cop_pow(-my, q - j);
        }
    }
    return v;
}

fn cop_marg_j(l3: f32, l4: f32) -> f32 {
    return l3 * l3 / 12.0 + l4 * l4 / 48.0 - l3 * l3 * l4 / 8.0 + 7.0 * l3 * l3 * l3 * l3 / 48.0;
}

fn cop_joint_j(w30: f32, w21: f32, w12: f32, w03: f32, w40: f32, w31: f32, w22: f32, w13: f32, w04: f32) -> f32 {
    let a = w03 * w03; let b = w12 * w12; let c = w21 * w21; let d = w30 * w30;
    return 0.1458333333 * a * a - 0.125 * a * w04 + 0.875 * a * b + 0.125 * a * c + a / 12.0
        + 1.5 * w03 * b * w21 - 0.5 * w03 * w12 * w13 + 0.25 * w03 * w12 * w21 * w30
        + w03 * w21 * c / 3.0 - 0.25 * w03 * w21 * w22 + w04 * w04 / 48.0 - 0.125 * w04 * b + 0.5625 * b * b
        + w12 * b * w30 / 3.0 + 2.0 * b * c - 0.5 * b * w22 + 0.125 * b * d + 0.25 * b
        - 0.5 * w12 * w13 * w21 + 1.5 * w12 * c * w30 - 0.5 * w12 * w21 * w31 - 0.25 * w12 * w22 * w30
        + w13 * w13 / 12.0 + 0.5625 * c * c - 0.5 * c * w22 + 0.875 * c * d - 0.125 * c * w40
        + 0.25 * c - 0.5 * w21 * w30 * w31 + 0.125 * w22 * w22 + 0.1458333333 * d * d
        - 0.125 * d * w40 + d / 12.0 + w31 * w31 / 12.0 + w40 * w40 / 48.0;
}

// Marks an undefined overlap (singular covariance); never a score.
const COP_BAD: f32 = -3.0e38;

// Consistent-E4 Edgeworth MI of the normal scores: needs moments 1..24 only.
fn copula_e4(m: ptr<function, array<f32, 45>>) -> f32 {
    let mx = (*m)[1];
    let my = (*m)[5];
    let k20 = cop_mu(m, 2u, 0u, mx, my);
    let k02 = cop_mu(m, 0u, 2u, mx, my);
    let k11 = cop_mu(m, 1u, 1u, mx, my);
    if (!(k20 > 1e-8) || !(k02 > 1e-8)) { return COP_BAD; }
    let sx = sqrt(k20);
    let sy = sqrt(k02);
    let rho = k11 / (sx * sy);
    let det = 1.0 - rho * rho;
    if (!(det > 1e-5)) { return COP_BAD; }
    var k: array<f32, 9>;    // 30 21 12 03 40 31 22 13 04, standardized
    k[0] = cop_mu(m, 3u, 0u, mx, my) / (sx * sx * sx);
    k[1] = cop_mu(m, 2u, 1u, mx, my) / (sx * sx * sy);
    k[2] = cop_mu(m, 1u, 2u, mx, my) / (sx * sy * sy);
    k[3] = cop_mu(m, 0u, 3u, mx, my) / (sy * sy * sy);
    k[4] = (cop_mu(m, 4u, 0u, mx, my) - 3.0 * k20 * k20) / (k20 * k20);
    k[5] = (cop_mu(m, 3u, 1u, mx, my) - 3.0 * k20 * k11) / (sx * sx * sx * sy);
    k[6] = (cop_mu(m, 2u, 2u, mx, my) - k20 * k02 - 2.0 * k11 * k11) / (k20 * k02);
    k[7] = (cop_mu(m, 1u, 3u, mx, my) - 3.0 * k02 * k11) / (sx * sy * sy * sy);
    k[8] = (cop_mu(m, 0u, 4u, mx, my) - 3.0 * k02 * k02) / (k02 * k02);
    // Whiten the joint: Z2 = (Y − ρX)/√(1−ρ²); w_pq = Σ_j C(q,j) (−ρ)^(q−j) g_(p+q−j, j) / d^q.
    let d = sqrt(det);
    let r = -rho;
    let w30 = k[0];
    let w21 = (k[1] + r * k[0]) / d;
    let w12 = (k[2] + 2.0 * r * k[1] + r * r * k[0]) / (d * d);
    let w03 = (k[3] + 3.0 * r * k[2] + 3.0 * r * r * k[1] + r * r * r * k[0]) / (d * d * d);
    let w40 = k[4];
    let w31 = (k[5] + r * k[4]) / d;
    let w22 = (k[6] + 2.0 * r * k[5] + r * r * k[4]) / (d * d);
    let w13 = (k[7] + 3.0 * r * k[6] + 3.0 * r * r * k[5] + r * r * r * k[4]) / (d * d * d);
    let w04 = (k[8] + 4.0 * r * k[7] + 6.0 * r * r * k[6] + 4.0 * r * r * r * k[5] + r * r * r * r * k[4]) / (d * d * d * d);
    let jj = cop_joint_j(w30, w21, w12, w03, w40, w31, w22, w13, w04);
    return -0.5 * log(det) + jj - cop_marg_j(k[0], k[4]) - cop_marg_j(k[3], k[8]);
}

fn cop_chol3(g: ptr<function, array<f32, 9>>) -> bool {
    for (var i = 0u; i < 3u; i++) {
        for (var j = 0u; j <= i; j++) {
            var s = (*g)[i * 3u + j];
            for (var k = 0u; k < j; k++) { s = s - (*g)[i * 3u + k] * (*g)[j * 3u + k]; }
            if (i == j) {
                if (!(s > 0.0)) { return false; }
                (*g)[i * 4u] = sqrt(s);
            } else {
                (*g)[i * 3u + j] = s / (*g)[j * 4u];
            }
        }
    }
    return true;
}

fn cop_tri(i: u32, j: u32) -> u32 {
    let a = min(i, j);
    let b = max(i, j);
    return a * 4u - (a * (a + 1u)) / 2u + b;
}

// λ²max's parts: the moving side's covariance factor L_A (lower triangle of ga, cop_chol3) and
// S = M Mᵀ, M = L_A⁻¹ C L_B⁻ᵀ the whitened cross-covariance (degree-3 features of each side,
// exact per-overlap whitening); false where undefined.
fn cop_lmax_parts(m: ptr<function, array<f32, 45>>, ridge: f32, ga: ptr<function, array<f32, 9>>, S: ptr<function, array<f32, 9>>) -> bool {
    var gb: array<f32, 9>;
    var cr: array<f32, 9>;
    for (var i = 0u; i < 3u; i++) {
        for (var j = 0u; j < 3u; j++) {
            (*ga)[i * 3u + j] = (*m)[25u + cop_tri(i, j)] - (*m)[1u + i] * (*m)[1u + j];
            gb[i * 3u + j] = (*m)[35u + cop_tri(i, j)] - (*m)[5u + i] * (*m)[5u + j];
            cr[i * 3u + j] = (*m)[9u + i * 4u + j] - (*m)[1u + i] * (*m)[5u + j];
        }
    }
    // Relative ridge per diagonal: the raw powers z, z², z³ have very different scales.
    for (var i = 0u; i < 3u; i++) { (*ga)[i * 4u] = (*ga)[i * 4u] * (1.0 + ridge); gb[i * 4u] = gb[i * 4u] * (1.0 + ridge); }
    if (!cop_chol3(ga) || !cop_chol3(&gb)) { return false; }
    var x: array<f32, 9>;
    for (var j = 0u; j < 3u; j++) {
        for (var i = 0u; i < 3u; i++) {
            var s = cr[i * 3u + j];
            for (var k = 0u; k < i; k++) { s = s - (*ga)[i * 3u + k] * x[k * 3u + j]; }
            x[i * 3u + j] = s / (*ga)[i * 4u];
        }
    }
    var mm: array<f32, 9>;
    for (var r = 0u; r < 3u; r++) {
        for (var i = 0u; i < 3u; i++) {
            var s = x[r * 3u + i];
            for (var k = 0u; k < i; k++) { s = s - gb[i * 3u + k] * mm[r * 3u + k]; }
            mm[r * 3u + i] = s / gb[i * 4u];
        }
    }
    for (var i = 0u; i < 3u; i++) {
        for (var j = 0u; j < 3u; j++) {
            (*S)[i * 3u + j] = mm[i * 3u] * mm[j * 3u] + mm[i * 3u + 1u] * mm[j * 3u + 1u] + mm[i * 3u + 2u] * mm[j * 3u + 2u];
        }
    }
    return true;
}

// The largest eigenvalue of a symmetric 3×3 (closed form), clamped to [0, 1 − 1e-6] as λ².
fn cop_top_eig3(S: array<f32, 9>) -> f32 {
    let p1 = S[1] * S[1] + S[2] * S[2] + S[5] * S[5];
    let q = (S[0] + S[4] + S[8]) / 3.0;
    let p2 = (S[0] - q) * (S[0] - q) + (S[4] - q) * (S[4] - q) + (S[8] - q) * (S[8] - q) + 2.0 * p1;
    var l2 = q;
    if (p2 > 1e-20) {
        let p = sqrt(p2 / 6.0);
        let b0 = (S[0] - q) / p; let b4 = (S[4] - q) / p; let b8 = (S[8] - q) / p;
        let b1 = S[1] / p; let b2 = S[2] / p; let b5 = S[5] / p;
        let detb = b0 * (b4 * b8 - b5 * b5) - b1 * (b1 * b8 - b5 * b2) + b2 * (b1 * b5 - b4 * b2);
        l2 = q + 2.0 * p * cos(acos(clamp(detb * 0.5, -1.0, 1.0)) / 3.0);
    }
    return clamp(l2, 0.0, 1.0 - 1e-6);
}

// −½ log(1 − λ²max), degree-3 features of each side, exact per-overlap whitening.
fn copula_lmax(m: ptr<function, array<f32, 45>>, ridge: f32) -> f32 {
    var ga: array<f32, 9>;
    var S: array<f32, 9>;
    if (!cop_lmax_parts(m, ridge, &ga, &S)) { return COP_BAD; }
    return -0.5 * log(1.0 - cop_top_eig3(S));
}

// λmax's Gauss–Newton weight on the moving features (smi.wgsl Q0, 4 × 4, the fourth feature
// unused): the score is −½ log(1 − λ²) of the top canonical pair alone, so its Gauss–Newton
// matrix is SMI's for that pair's one-dimensional features — λ² a aᵀ, a = L_A⁻ᵀ u the moving
// side's canonical direction (u: S's top eigenvector) — times d/dλ² of the score, 1 / (2 (1 −
// λ²)). Zero where undefined.
fn copula_lmax_q(m: ptr<function, array<f32, 45>>, ridge: f32) -> array<f32, 16> {
    var Q: array<f32, 16>;
    var ga: array<f32, 9>;
    var S: array<f32, 9>;
    if (!cop_lmax_parts(m, ridge, &ga, &S)) { return Q; }
    let l2 = cop_top_eig3(S);
    // u: the cross product of two rows of S − λ² I (the largest of the three), else power
    // iteration (a repeated top eigenvalue: any vector of its space will do)
    let r0 = vec3f(S[0] - l2, S[1], S[2]);
    let r1 = vec3f(S[3], S[4] - l2, S[5]);
    let r2 = vec3f(S[6], S[7], S[8] - l2);
    let c01 = cross(r0, r1);
    let c02 = cross(r0, r2);
    let c12 = cross(r1, r2);
    var uv = c01;
    if (dot(c02, c02) > dot(uv, uv)) { uv = c02; }
    if (dot(c12, c12) > dot(uv, uv)) { uv = c12; }
    let sc = max(abs(S[0]) + abs(S[4]) + abs(S[8]), 1e-30);
    if (!(dot(uv, uv) > 1e-10 * sc * sc * sc * sc)) {
        uv = vec3f(1.0, 1.0, 1.0);
        for (var it = 0u; it < 32u; it++) {
            uv = vec3f(S[0] * uv.x + S[1] * uv.y + S[2] * uv.z, S[3] * uv.x + S[4] * uv.y + S[5] * uv.z, S[6] * uv.x + S[7] * uv.y + S[8] * uv.z);
            uv = uv / max(length(uv), 1e-30);
        }
    }
    uv = uv / max(length(uv), 1e-30);
    // a = L_A⁻ᵀ u (L_Aᵀ a = u, back substitution)
    var a: array<f32, 3>;
    for (var r = 0u; r < 3u; r++) {
        let i = 2u - r;
        var s = uv[i];
        for (var k = i + 1u; k < 3u; k++) { s = s - ga[k * 3u + i] * a[k]; }
        a[i] = s / ga[i * 4u];
    }
    let c = l2 / (2.0 * (1.0 - l2));
    for (var i = 0u; i < 3u; i++) {
        for (var j = 0u; j < 3u; j++) { Q[i * 4u + j] = c * a[i] * a[j]; }
    }
    for (var i = 0u; i < 16u; i++) { if (!(abs(Q[i]) < 1e30)) { return array<f32, 16>(); } }
    return Q;
}

// mode 2: E4, 3: λmax. Undefined → 0 (so it never wins a map).
fn copula_score(m: ptr<function, array<f32, 45>>, mode: u32, ridge: f32) -> f32 {
    var v = 0.0;
    if (mode == 2u) { v = copula_e4(m); } else { v = copula_lmax(m, ridge); }
    if (!(v == v) || v <= COP_BAD) { return 0.0; }
    return v;
}
