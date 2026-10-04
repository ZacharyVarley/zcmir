"""ctypes bindings to the zcmir shared library (built from Zig; see src/exports.zig, src/native_api.zig)."""
import ctypes
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

U32, I32, I64, F64, SZ, P = ctypes.c_uint32, ctypes.c_int32, ctypes.c_int64, ctypes.c_double, ctypes.c_size_t, ctypes.c_void_p
f32p, f64p, u8p, u32p = ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_uint32)
Mat = ctypes.c_double * 9


class ShiftResult(ctypes.Structure):
    _fields_ = [
        ("n", U32), ("cw", U32), ("ch", U32), ("canvas_w", U32), ("canvas_h", U32),
        ("ox", I32), ("oy", I32), ("peak_index", U32),
        ("dx", F64), ("dy", F64), ("peak", F64), ("zero", F64), ("corr_area_px", F64),
        ("exact", U32), ("calibrated", U32), ("n_y", U32), ("_pad", U32),
    ]


class RsResult(ctypes.Structure):
    _fields_ = [
        ("n", U32), ("n_th", U32), ("n_lam", U32), ("peak_index", U32),
        ("dlam", F64), ("r0", F64), ("r1", F64), ("cx", F64), ("cy", F64), ("peak", F64), ("zero", F64),
        ("dth", F64), ("dsg", F64), ("symmetric", U32), ("calibrated", U32),
    ]


class GradResult(ctypes.Structure):
    _fields_ = [
        ("score", F64), ("fwd", F64), ("inv", F64), ("n", F64),
        ("grad", F64 * 8), ("hess", F64 * 64), ("has_hess", U32), ("nk", U32),
    ]


class ClimbResult(ctypes.Structure):
    _fields_ = [("H", Mat), ("score", F64), ("iterations", U32), ("moved", U32)]


class DetectResult(ctypes.Structure):
    _fields_ = [("n1", U32), ("n2", U32), ("levels1", U32), ("levels2", U32)]


class MatchResult(ctypes.Structure):
    _fields_ = [
        ("H", Mat), ("ninl", U32), ("n_corr", U32), ("ninl_aff", U32), ("pos_n", U32), ("pos_inl", U32),
        ("has_rot", U32), ("rot_deg", F64), ("pos_kept", U32), ("_pad", U32),
    ]


class SearchResult(ctypes.Structure):
    _fields_ = [("H", Mat), ("score", F64), ("prev", F64), ("improved", U32), ("count_a", U32), ("count_b", U32), ("_pad", U32)]


class Canvas(ctypes.Structure):
    _fields_ = [("ox", I32), ("oy", I32), ("ow", U32), ("oh", U32), ("Hc", Mat)]


def _names():
    if sys.platform == "win32":
        return "zcmir.dll", "wgpu_native.dll"
    if sys.platform == "darwin":
        return "libzcmir.dylib", "libwgpu_native.dylib"
    return "libzcmir.so", "libwgpu_native.so"


def _find(env, name, dev_dirs):
    """ZCMIR_LIB / ZCMIR_WGPU, then the package's _lib/ (wheels), then a source checkout."""
    if os.environ.get(env):
        return Path(os.environ[env])
    for d in [HERE / "_lib", *dev_dirs]:
        if (d / name).exists():
            return d / name
    raise FileNotFoundError(f"{name} not found (set {env}, or build: zig build)")


_SIGS = {
    # name: (argtypes, restype)
    "zc_version": ([], ctypes.c_char_p),
    "zc_create": ([ctypes.c_char_p], P),
    "zc_create_error": ([ctypes.c_char_p, SZ], SZ),
    "zc_destroy": ([P], None),
    "zc_adapter": ([P, ctypes.c_char_p, SZ], SZ),
    "zc_error": ([P, ctypes.c_char_p, SZ], SZ),
    "zc_configure": ([P, ctypes.c_char_p, SZ], I32),
    "zc_settings": ([P, ctypes.c_char_p, SZ], SZ),
    "zc_set_image": ([P, U32, f32p, U32, U32], I32),
    "zc_set_image_gray": ([P, U32, f32p, U32, U32, U32], I32),
    "zc_set_image_rgba": ([P, U32, u8p, U32, U32], I32),
    "zc_work_image": ([P, U32, f32p, SZ], I32),
    "zc_features": ([P, U32, f32p, SZ], I32),
    "zc_compile_wgsl": ([P, ctypes.c_char_p, SZ, ctypes.c_char_p, SZ, ctypes.c_char_p, SZ], I32),
    "zc_moments": ([P, f64p, f64p], I32),
    "zc_pg_info": ([P, f64p], I32),
    "zc_pg_maps": ([P, U32, f32p, SZ], I32),
    "zc_pg_descriptors": ([P, U32, f32p, SZ], I32),
    "zc_image_size": ([P, U32, u32p], None),
    "zc_cancel": ([P], None),
    "zc_profile": ([P, U32], I32),
    "zc_bench_moments": ([P, U32, U32], I32),
    "zc_tune_parts": ([U32, U32, U32, U32], None),
    "zc_gpu_counts": ([P, u32p], None),
    "zc_profile_report": ([P, ctypes.c_char_p, SZ], SZ),
    "zc_swap": ([P], None),
    "zc_match_scores": ([P, f32p, SZ], I32),
    "zc_detect": ([P, ctypes.POINTER(DetectResult)], I32),
    "zc_match": ([P, ctypes.POINTER(MatchResult)], I32),
    "zc_search": ([P, ctypes.POINTER(SearchResult)], I32),
    "zc_auto": ([P, ctypes.POINTER(ClimbResult)], I32),
    "zc_climb": ([P, ctypes.POINTER(ClimbResult)], I32),
    "zc_keypoints": ([P, U32, f32p, SZ], I32),
    "zc_keypoint_frames": ([P, U32, f32p, SZ], I32),
    "zc_matches": ([P, u32p, SZ], I32),
    "zc_set_ffd": ([P, f32p, SZ], I32),
    "zc_get_ffd": ([P, f32p, SZ], SZ),
    "zc_set_pose": ([P, f64p], None),
    "zc_fit_check": ([P, f64p], U32),
    "zc_get_pose": ([P, f64p], None),
    "zc_events": ([P, ctypes.c_char_p, SZ], SZ),
    "zc_score": ([P, f64p, f64p], I32),
    "zc_gradient": ([P, f64p, U32, ctypes.POINTER(GradResult)], I32),
    "zc_ffd_gradient": ([P, f64p, f64p, f64p, SZ], I32),
    "zc_shift_map": ([P, f64p, U32, U32, ctypes.POINTER(ShiftResult), f32p, SZ], I32),
    "zc_shift_map_set": ([P, f64p, ctypes.POINTER(ShiftResult), f32p, SZ], I32),
    "zc_rs_map": ([P, f64p, f64p, U32, ctypes.POINTER(RsResult), f32p, f32p, f32p, SZ], I32),
    "zc_canvas": ([P, f64p, U32, ctypes.POINTER(Canvas)], I32),
    "zc_overlay": ([P, f64p, U32, ctypes.POINTER(Canvas), f32p, f32p, SZ], I32),
    "zc_tile_heat": ([P, f64p, U32, U32, f64p, f32p, f32p, SZ], I32),
    "zc_export_warp": ([P, ctypes.c_char_p, SZ, P, SZ], I64),
    "zc_displacement": ([P, f32p, SZ], I32),
}


def load():
    lib_name, wgpu_name = _names()
    repo = HERE.parent.parent  # source checkout: <repo>/python/zcmir
    lib_path = _find("ZCMIR_LIB", lib_name, [repo / "zig-out" / "bin", repo / "zig-out" / "lib"])
    wgpu_dirs = sorted((repo / ".deps" / "wgpu-native").glob("*/lib")) if (repo / ".deps").exists() else []
    wgpu_path = _find("ZCMIR_WGPU", wgpu_name, wgpu_dirs)
    lib = ctypes.CDLL(str(lib_path))
    for name, (args, res) in _SIGS.items():
        fn = getattr(lib, name)
        fn.argtypes, fn.restype = args, res
    return lib, str(wgpu_path)
