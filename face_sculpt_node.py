# -*- coding: utf-8 -*-
# FaceSculpt：人脸参数化微调 ComfyUI 节点
# 路线：MediaPipe 478 关键点 → 语义区域加权位移 → 三角剖分仿射/TPS 扭曲原图
# 纯像素位移，不生成新像素 → 身份 100% 保留（医美微调甜区 ≤8%）
import os
import numpy as np
import cv2
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_LANDMARKER_PATH = os.path.join(_HERE, "models", "face_landmarker.task")

# ---------------- MediaPipe 关键点提取（单例懒加载） ----------------
_LANDMARKER = None

def _get_landmarker():
    global _LANDMARKER
    if _LANDMARKER is None:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
        opts = mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=_LANDMARKER_PATH),
            running_mode=mp_vision.RunningMode.IMAGE,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1)
        _LANDMARKER = mp_vision.FaceLandmarker.create_from_options(opts)
    return _LANDMARKER


def detect_landmarks(img_rgb):
    """img_rgb: HxWx3 uint8 RGB → (478,2) 像素坐标 or None"""
    import mediapipe as mp
    lm = _get_landmarker()
    mp_img = mp.Image(mp.ImageFormat.SRGB, img_rgb)
    res = lm.detect(mp_img)
    if not res.face_landmarks:
        return None
    pts = np.array([[p.x * img_rgb.shape[1], p.y * img_rgb.shape[0]]
                    for p in res.face_landmarks[0]], dtype=np.float32)
    return pts


# ---------------- 语义区域定义（MediaPipe 478 索引组） ----------------
# 参考 MediaPipe Face Mesh 标准拓扑索引
REGIONS = {
    # 鼻梁（山根到鼻尖）纵向
    "nose_bridge": [6, 197, 195, 5, 4, 1, 2, 98, 327],
    # 鼻尖
    "nose_tip": [1, 2, 4, 19, 94, 168, 175, 5],
    # 鼻翼（左右）
    "nose_ala": [129, 358, 102, 331, 118, 117, 47, 277, 219, 439],
    # 内眼角
    "eye_inner": [362, 133, 398, 382, 384, 463, 33, 245, 243, 173, 155, 157],
    # 眼睛大小（上下眼睑）
    "eye_lid": [159, 145, 158, 153, 386, 374, 385, 380, 387, 373],
    # 嘴角
    "mouth_corner": [61, 291, 91, 321],
    # 唇厚度（上唇13/14，下唇17/0 唇珠唇谷）
    "lip_full": [13, 14, 17, 0, 82, 312, 84, 314, 87, 317, 78, 308],
    # 下颌/双下巴 轮廓线
    "jaw_line": [152, 148, 176, 149, 150, 136, 172, 58, 288, 379, 397, 400,
                 234, 454, 93, 323, 132, 361, 58, 288],
    # 脸颊（瘦脸）
    "cheek": [116, 117, 118, 119, 100, 47, 345, 340, 346, 347, 348, 329],
    # 眉毛
    "brow": [105, 334, 107, 336, 70, 300, 63, 293],
    # 下巴尖
    "chin": [152, 175, 199, 200],
}

# 整张脸的参考锚点（不随编辑动的外圈+轮廓外环，防背景错位）
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
             397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
             172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]


def _region_center(pts, idx_list):
    idx = [i for i in idx_list if 0 <= i < len(pts)]
    return pts[idx].mean(axis=0) if idx else None


def _apply_edit(deltas, pts, idx_list, vec, sigma):
    """对区域关键点施加高斯衰减位移：越靠区域中心位移越大"""
    idx = [i for i in idx_list if 0 <= i < len(pts)]
    if not idx:
        return
    center = pts[idx].mean(axis=0)
    all_pts = np.asarray(idx)
    d = np.linalg.norm(pts[all_pts] - center, axis=1)
    w = np.exp(-(d ** 2) / (2 * sigma ** 2))
    # 对区域外点也给轻微衰减位移（平滑过渡），用全体点距离衰减
    d_all = np.linalg.norm(pts - center, axis=1)
    w_all = np.exp(-(d_all ** 2) / (2 * (sigma * 2.5) ** 2))
    w_all = np.clip(w_all, 0, 0.35)  # 外围最多 35% 跟随，防止全脸漂移
    w_final = np.zeros(len(pts), dtype=np.float32)
    w_final[all_pts] = w
    w_final = np.maximum(w_final, w_all)
    deltas += vec[None, :] * w_final[:, None]


def build_edit_deltas(pts, edits):
    """edits: dict[str,float] ∈[-1,1] → 每个关键点 2D 位移
    幅度标定：以双眼间距为基准长度，+1.0 ≈ 移动 6% 眼距（医美微调甜区）"""
    deltas = np.zeros_like(pts)
    eye_dist = np.linalg.norm(pts[362] - pts[33]) + 1e-6
    unit = eye_dist * 0.06
    sigma = eye_dist * 0.35

    nose_c = _region_center(pts, REGIONS["nose_bridge"])
    up = np.array([0.0, -1.0], dtype=np.float32)

    if edits.get("nose_lift"):
        _apply_edit(deltas, pts, REGIONS["nose_bridge"], up * edits["nose_lift"] * unit, sigma)
    if edits.get("nose_tip_up"):
        _apply_edit(deltas, pts, REGIONS["nose_tip"], up * edits["nose_tip_up"] * unit, sigma * 0.8)
    if edits.get("nose_shrink"):
        # 鼻翼向内收：左右翼朝鼻梁中心线收
        c = _region_center(pts, REGIONS["nose_ala"])
        for i in REGIONS["nose_ala"]:
            if i >= len(pts):
                continue
            dirv = np.array([np.sign(c[0] - pts[i][0]), 0.0], dtype=np.float32)
            _apply_edit(deltas, pts, [i], dirv * abs(edits["nose_shrink"]) * unit * 0.7, sigma * 0.5)
    if edits.get("eye_inner"):
        # 开内眼角：左右内眼角相向
        for i in REGIONS["eye_inner"]:
            if i >= len(pts):
                continue
            side = 1.0 if pts[i][0] < pts[:, 0].mean() else -1.0
            _apply_edit(deltas, pts, [i], np.array([-side, 0.0], np.float32) * edits["eye_inner"] * unit * 0.6, sigma * 0.5)
    if edits.get("eye_big"):
        # 眼睛变大：上眼睑上抬+下眼睑下拉
        for i in REGIONS["eye_lid"]:
            if i >= len(pts):
                continue
            sign = -1.0 if i in (159, 158, 386, 385) else 1.0
            _apply_edit(deltas, pts, [i], np.array([0.0, sign], np.float32) * edits["eye_big"] * unit * 0.8, sigma * 0.45)
    if edits.get("lip_full"):
        # 丰唇：唇缘点沿唇中线外扩
        mc = _region_center(pts, REGIONS["lip_full"])
        for i in REGIONS["lip_full"]:
            if i >= len(pts):
                continue
            sign = -1.0 if pts[i][1] < mc[1] else 1.0
            _apply_edit(deltas, pts, [i], np.array([0.0, sign], np.float32) * edits["lip_full"] * unit * 0.5, sigma * 0.4)
    if edits.get("mouth_width"):
        # 嘴角宽度：左右嘴角点外扩/内收
        mc = _region_center(pts, REGIONS["mouth_corner"])
        for i in REGIONS["mouth_corner"]:
            if i >= len(pts):
                continue
            side = -1.0 if pts[i][0] < mc[0] else 1.0
            _apply_edit(deltas, pts, [i], np.array([side, 0.0], np.float32) * edits["mouth_width"] * unit * 0.6, sigma * 0.4)
    if edits.get("jaw_slim"):
        # 收下颌：下颌轮廓点朝面部中线收
        fc = pts[:, 0].mean()
        for i in REGIONS["jaw_line"]:
            if i >= len(pts):
                continue
            side = 1.0 if pts[i][0] < fc else -1.0
            _apply_edit(deltas, pts, [i], np.array([-side, 0.0], np.float32) * edits["jaw_slim"] * unit, sigma * 0.6)
    if edits.get("cheek_slim"):
        fc = pts[:, 0].mean()
        for i in REGIONS["cheek"]:
            if i >= len(pts):
                continue
            side = 1.0 if pts[i][0] < fc else -1.0
            _apply_edit(deltas, pts, [i], np.array([-side, 0.0], np.float32) * edits["cheek_slim"] * unit * 0.8, sigma * 0.6)
    if edits.get("brow_lift"):
        _apply_edit(deltas, pts, REGIONS["brow"], up * edits["brow_lift"] * unit, sigma * 0.7)
    if edits.get("chin_down"):
        _apply_edit(deltas, pts, REGIONS["chin"], -up * edits["chin_down"] * unit, sigma * 0.5)
    if edits.get("chin_shrink"):
        _apply_edit(deltas, pts, REGIONS["chin"], up * edits["chin_shrink"] * unit, sigma * 0.5)
    return deltas


def piecewise_affine_warp(img, src_pts, dst_pts, extra_boundary=True):
    """三角剖分仿射扭曲：src->dst 逐三角 warpAffine，O(三角形数) 快速精确"""
    h, w = img.shape[:2]
    src = np.asarray(src_pts, dtype=np.float32)
    dst = np.asarray(dst_pts, dtype=np.float32)
    if extra_boundary:
        # 图像四角+边中点作为固定锚点，防画面边缘被拖走
        border = np.array([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1],
                           [w // 2, 0], [w // 2, h - 1], [0, h // 2], [w - 1, h // 2]],
                          dtype=np.float32)
        src = np.vstack([src, border])
        dst = np.vstack([dst, border])
    from scipy.spatial import Delaunay
    tri = Delaunay(dst)
    out = np.zeros_like(img)
    for simplex in tri.simplices:
        tri_dst = np.ascontiguousarray(dst[simplex], dtype=np.float32)
        tri_src = np.ascontiguousarray(src[simplex], dtype=np.float32)
        r1 = cv2.boundingRect(tri_dst)
        r2 = cv2.boundingRect(tri_src)
        if r1[2] <= 0 or r1[3] <= 0 or r2[2] <= 0 or r2[3] <= 0:
            continue
        tri1 = np.ascontiguousarray(tri_dst - [r1[0], r1[1]], dtype=np.float32)
        tri2 = np.ascontiguousarray(tri_src - [r2[0], r2[1]], dtype=np.float32)
        mask = np.zeros((r1[3], r1[2], 3), dtype=np.float32)
        cv2.fillConvexPoly(mask, np.int32(tri1), (1.0, 1.0, 1.0), 16, 0)
        patch = img[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]
        if patch.size == 0:
            continue
        M, _ = cv2.estimateAffine2D(tri2, tri1, method=cv2.LMEDS)
        if M is None or M.shape[0] != 2:
            # LMEDS 三点全等时可能失败，退回标准仿射
            M = cv2.getAffineTransform(tri2, tri1)
        warped = cv2.warpAffine(patch, M, (r1[2], r1[3]),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REFLECT_101)
        roi = out[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
        roi[:] = roi * (1.0 - mask) + warped * mask
    return out


def feather_blend(base, warped, pts, feather_ratio=0.08):
    """以关键点凸包+外扩做人脸区域羽化融合，接缝无痕"""
    h, w = base.shape[:2]
    hull = cv2.convexHull(np.asarray(pts, dtype=np.float32))
    mask = np.zeros((h, w), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(hull), 1.0)
    # 外扩+羽化
    k = max(3, int(min(h, w) * feather_ratio))
    mask = cv2.dilate(mask, np.ones((k, k), np.uint8), iterations=1)
    mask = cv2.GaussianBlur(mask, (0, 0), k / 2.5)
    mask = np.clip(mask, 0, 1)[..., None]
    return (warped * mask + base * (1.0 - mask)).astype(np.uint8)


# ---------------- FaceVerse 3DMM 重建（骨相引擎） ----------------
_FVR = None


def _find_fv_dir():
    """FaceVerse_v4 仓库目录多布局查找：包内 > custom_nodes 同级 > 环境变量"""
    cands = [
        os.path.join(_HERE, "FaceVerse_v4"),
        os.path.join(os.path.dirname(_HERE), "FaceVerse_v4"),
        os.environ.get("FACE_VERSE_DIR", ""),
    ]
    for c in cands:
        # 官方仓库内 faceversev4 是子包目录，兼容文件/目录两种形态
        if c and (os.path.isfile(os.path.join(c, "faceversev4.py"))
                  or os.path.isdir(os.path.join(c, "faceversev4"))):
            return c
    return None


def _get_fvr(device):
    """懒加载 FaceVerseRecon（PyTorch ResNet50 回归 621 维参数）"""
    global _FVR
    if _FVR is None:
        fv_dir = _find_fv_dir()
        if fv_dir is None:
            raise RuntimeError(
                "未找到 FaceVerse_v4 仓库，请运行 install.py 或手动克隆 "
                "https://github.com/LizhenWangT/FaceVerse_v4 到本节点目录/同级目录，"
                "并下载 data/faceverse_v4_2.npy + data/faceverse_resnet50.pth")
        import sys
        if fv_dir not in sys.path:
            sys.path.insert(0, fv_dir)
        from faceversev4 import FaceVerseRecon
        _FVR = FaceVerseRecon(
            os.path.join(fv_dir, "data", "faceverse_v4_2.npy"),
            os.path.join(fv_dir, "data", "faceverse_resnet50.pth"),
            device)
    return _FVR


def _detect_bbox(img_rgb):
    """与 run.py detect_lms 同逻辑：mediapipe 478 → [bbox, eye_params]"""
    pts = detect_landmarks(img_rgb)
    if pts is None:
        return None, None
    def _norm(v):
        return v / (np.linalg.norm(v) + 1e-9)
    def _dist(a, b):
        return float(np.linalg.norm(a - b))
    vec = _norm(pts[362] - pts[263])
    leyex = np.dot(pts[473] - (pts[263] + pts[362]) / 2, vec) / _dist(pts[362], pts[263]) * 3
    leyey = np.dot(pts[473] - (pts[263] + pts[362]) / 2, vec[[1, 0]]) / _dist(pts[362], pts[263]) * -1.5
    vec2 = _norm(pts[33] - pts[133])
    reyex = np.dot(pts[468] - (pts[33] + pts[133]) / 2, vec2) / _dist(pts[33], pts[133]) * 3
    reyey = np.dot(pts[468] - (pts[33] + pts[133]) / 2, vec2[[1, 0]]) / _dist(pts[33], pts[133]) * -1.5
    bbox = [float(pts[:, 0].min()), float(pts[:, 1].min()),
            float(pts[:, 0].max()), float(pts[:, 1].max())]
    return [bbox, [float(leyey), float(leyex), float(reyey), float(reyex)]], pts


# ---------------- ComfyUI 节点 ----------------

def _to_uint8(img_tensor):
    t = img_tensor[0].detach().cpu().numpy()
    return np.clip(t * 255.0, 0, 255).astype(np.uint8)


def _to_tensor(img_uint8):
    return torch.from_numpy(img_uint8.astype(np.float32) / 255.0).unsqueeze(0)


class FaceSculptLandmarks:
    """MediaPipe 478 人脸关键点检测：输出 LANDMARKS(JSON) + 可视化叠加图"""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",),
                             "draw": ("BOOLEAN", {"default": True})}}

    RETURN_TYPES = ("LANDMARKS", "IMAGE")
    RETURN_NAMES = ("landmarks", "overlay")
    FUNCTION = "run"
    CATEGORY = "face/sculpt"

    def run(self, image, draw=True):
        import json
        img_rgb = _to_uint8(image)  # ComfyUI IMAGE 即 RGB
        pts = detect_landmarks(img_rgb)
        if pts is None:
            raise RuntimeError("未检测到人脸")
        vis = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        if draw:
            for p in pts.astype(np.int32):
                cv2.circle(vis, tuple(p), 1, (0, 255, 0), -1)
        vis = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        return (json.dumps({"pts": pts.tolist(), "w": img_rgb.shape[1], "h": img_rgb.shape[0]}),
                _to_tensor(vis))


class FaceSculptEdit:
    """语义微调编辑：对 landmarks 施加区域位移并仿射扭曲原图"""
    @classmethod
    def INPUT_TYPES(cls):
        fl = lambda d: ("FLOAT", {"default": d, "min": -1.0, "max": 1.0, "step": 0.05})
        return {"required": {
            "image": ("IMAGE",),
            "landmarks": ("LANDMARKS",),
            "nose_lift": fl(0.0), "nose_tip_up": fl(0.0), "nose_shrink": fl(0.0),
            "eye_inner": fl(0.0), "eye_big": fl(0.0),
            "lip_full": fl(0.0), "mouth_width": fl(0.0),
            "jaw_slim": fl(0.0), "cheek_slim": fl(0.0),
            "brow_lift": fl(0.0), "chin_down": fl(0.0), "chin_shrink": fl(0.0),
            "blend_feather": ("FLOAT", {"default": 0.08, "min": 0.01, "max": 0.3, "step": 0.01}),
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "face/sculpt"

    def run(self, image, landmarks, blend_feather=0.08, **edits):
        import json
        img_rgb = _to_uint8(image)
        meta = json.loads(landmarks)
        pts = np.array(meta["pts"], dtype=np.float32)
        # landmarks 坐标系与当前图一致时直接用；不一致则按比例缩放
        if meta["w"] != img_rgb.shape[1] or meta["h"] != img_rgb.shape[0]:
            sx = img_rgb.shape[1] / meta["w"]
            sy = img_rgb.shape[0] / meta["h"]
            pts = pts * [sx, sy]
        deltas = build_edit_deltas(pts, edits)
        dst = pts + deltas
        warped = piecewise_affine_warp(img_rgb, pts, dst)
        out = feather_blend(img_rgb, warped, pts, blend_feather)
        return (_to_tensor(out),)


class FaceVerseReconNode:
    """FaceVerse V4 重建：image → FVDATA(coeffs+投影顶点+三角网格) + 3D 渲染预览"""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",),
                             "device": (["cuda", "cpu"], {"default": "cuda"}),
                             "render": ("BOOLEAN", {"default": True})}}

    RETURN_TYPES = ("FVDATA", "IMAGE")
    RETURN_NAMES = ("fvdata", "mesh_preview")
    FUNCTION = "run"
    CATEGORY = "face/sculpt"

    def run(self, image, device="cuda", render=True):
        import json
        img_rgb = _to_uint8(image)
        h, w = img_rgb.shape[:2]
        det, _ = _detect_bbox(img_rgb)
        if det is None:
            raise RuntimeError("未检测到人脸")
        fvr = _get_fvr(device)
        lms_in = np.array([det[0]], dtype=np.float32)  # [1,4] bbox
        coeffs, bbox_list = fvr.process_imgs(img_rgb, lms_in)
        coeffs[:, -4:] = torch.from_numpy(
            np.array([det[1]], dtype=np.float32)).to(coeffs.device)
        # from_coeffs 返回 numpy：vs(3D), vs_proj(原图坐标2D), normals, colors_illumin
        vs, vs_proj, normals, colors = fvr.from_coeffs(coeffs, bbox_list)
        data = {
            "coeffs": coeffs[0].detach().cpu().tolist(),
            "bbox_list": np.asarray(bbox_list).tolist(),
            "vs_proj": np.asarray(vs_proj)[0].tolist(),
            "device": device, "w": w, "h": h,
        }
        vis = img_rgb
        if render:
            import sys
            fv_dir = _find_fv_dir()
            if fv_dir and fv_dir not in sys.path:
                sys.path.insert(0, fv_dir)
            from Sim3DR.renderer import render_fvr
            rgb, _ = render_fvr(img_rgb, np.asarray(vs_proj)[0],
                                fvr.fvd["tri"], np.asarray(normals)[0],
                                np.asarray(colors)[0])
            vis = rgb
        return (json.dumps(data), _to_tensor(np.clip(vis, 0, 255).astype(np.uint8)))


class FaceVerseSculpt:
    """FaceVerse 语义形变：改 id_coeff(shape) → 投影顶点差 → 三角网格仿射扭曲原图
    照片纹理零损失，只挪几何——骨相形变但人物还是本人"""
    @classmethod
    def INPUT_TYPES(cls):
        fl = lambda d: ("FLOAT", {"default": d, "min": -1.0, "max": 1.0, "step": 0.05})
        return {"required": {
            "image": ("IMAGE",),
            "fvdata": ("FVDATA",),
            "shape_scale": fl(0.0),      # 整体脸型缩放（正=饱满，负=瘦削）
            "nose_lift": fl(0.0),        # 鼻梁抬高（顶点组上移）
            "jaw_slim": fl(0.0),         # 下颌内收
            "chin_length": fl(0.0),      # 下巴长度
            "shape_dim": ("INT", {"default": -1, "min": -1, "max": 155}),
            "shape_delta": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.5}),
            "blend_feather": ("FLOAT", {"default": 0.08, "min": 0.01, "max": 0.3, "step": 0.01}),
        }}

    RETURN_TYPES = ("IMAGE", "LANDMARKS")
    RETURN_NAMES = ("image", "mesh_after")
    FUNCTION = "run"
    CATEGORY = "face/sculpt"

    # FaceVerse 28632 顶点中选取语义子集（与 mediapipe 关键点对应的顶点索引）
    # 用 fvd['keypoints_mediapipe'] 的顶点索引做锚点 → 邻域半径圈选
    def run(self, image, fvdata, blend_feather=0.08, shape_dim=-1, shape_delta=0.0,
            **edits):
        import json
        img_rgb = _to_uint8(image)
        meta = json.loads(fvdata)
        device = meta.get("device", "cuda")
        fvr = _get_fvr(device)
        coeffs = torch.tensor([meta["coeffs"]], dtype=torch.float32,
                              device=fvr.device)
        bbox_list = np.array(meta["bbox_list"], dtype=np.int32)

        # id_coeff(shape) 修改：shape_scale 整体缩放 / shape_dim 单维调试
        id_dims = getattr(fvr, "id_dims", 156)
        if edits.get("shape_scale"):
            coeffs[:, :id_dims] = coeffs[:, :id_dims] * (1.0 + 0.5 * edits["shape_scale"])
        if shape_dim >= 0 and shape_delta:
            coeffs[:, shape_dim] = coeffs[:, shape_dim] + shape_delta

        vs, vs_proj, normals, colors = fvr.from_coeffs(coeffs, bbox_list)
        new_proj = np.asarray(vs_proj)[0].astype(np.float32)[:, :2]
        old_proj = np.array(meta["vs_proj"], dtype=np.float32)[:, :2]

        # 2D 顶点级语义微调（不依赖 PCA 轴语义）：近邻锚点圈选区域后加权平移
        deltas3d = np.zeros_like(new_proj)
        fvd = fvr.fvd
        kp_idx = np.asarray(fvd.get("keypoints_mediapipe", []), dtype=np.int64).flatten()
        # kp_idx[i] = 顶点索引（对应 mediapipe 点 i）；语义组→顶点组
        def vgroup(mp_ids, radius):
            empty = (np.array([], dtype=np.int64), None)
            if not kp_idx.size:
                return empty
            ids = np.asarray(mp_ids, dtype=np.int64)
            ids = ids[(ids >= 0) & (ids < len(kp_idx))]
            if not ids.size:
                return empty
            vi = kp_idx[ids]
            vi = vi[(vi >= 0) & (vi < len(new_proj))]
            if not vi.size:
                return empty
            c = new_proj[vi].mean(axis=0)
            d = np.linalg.norm(new_proj - c, axis=1)
            return np.where(d < radius)[0], c
        unit = 20.0  # 像素基准
        def _weighted(sel, c, vec, radius):
            """组内高斯衰减位移：稠密场插值下硬边界会产生折痕"""
            if not len(sel) or c is None:
                return
            d = np.linalg.norm(new_proj[sel] - c, axis=1)
            wts = np.exp(-(d ** 2) / (2 * (radius * 0.55) ** 2))
            deltas3d[sel] += np.asarray(vec, dtype=np.float32) * wts[:, None]
        if edits.get("nose_lift"):
            sel, c = vgroup(REGIONS["nose_bridge"] + REGIONS["nose_tip"], 60)
            _weighted(sel, c, np.array([0.0, -edits["nose_lift"] * unit], np.float32), 60)
        if edits.get("jaw_slim"):
            sel, c = vgroup(REGIONS["jaw_line"], 140)
            if len(sel):
                cx = new_proj[:, 0].mean()
                side = np.where(new_proj[sel, 0] < cx, 1.0, -1.0)
                vec = np.stack([-side * edits["jaw_slim"] * unit,
                                np.zeros(len(sel), np.float32)], axis=1)
                _weighted(sel, c, vec, 140)
        if edits.get("chin_length"):
            sel, c = vgroup(REGIONS["chin"], 60)
            _weighted(sel, c, np.array([0.0, edits["chin_length"] * unit], np.float32), 60)

        dst = new_proj + deltas3d
        w, h = meta["w"], meta["h"]
        # 全量顶点 delta 场 + FaceVerse 固定三角拓扑 → 稠密插值 + 一次 remap
        # 精度=全网格无三角线性化损失，速度=向量化秒级
        delta_total = dst - old_proj
        tri_fv = np.asarray(fvd["tri"], dtype=np.int64)
        out = _dense_field_warp(img_rgb, delta_total, tri_fv, dst)
        out = feather_blend(img_rgb, out,
                            old_proj[:: max(1, len(old_proj) // 300)], blend_feather)
        mesh_json = json.dumps({"pts": dst[:: max(1, len(dst) // 480)].tolist(),
                                "w": w, "h": h})
        return (_to_tensor(out), mesh_json)


def _dense_field_warp(img, delta, tri, dst_pts):
    """顶点 delta 场经模型三角拓扑稠密插值 → cv2.remap 一次成型。
    全量顶点参与（无锚点抽稀的信息损失），向量化实现秒级完成。"""
    import matplotlib.tri as mtri
    h, w = img.shape[:2]
    # 输出域建在 dst 顶点位置上：输出像素 p 的采样源 = p - delta(p)
    tri_xy = mtri.Triangulation(dst_pts[:, 0], dst_pts[:, 1], tri)
    # matplotlib 自带三网线插值器直接吃 fvd 拓扑，免 28k 点 Delaunay
    fx = mtri.LinearTriInterpolator(tri_xy, np.asarray(delta[:, 0], dtype=np.float64))
    fy = mtri.LinearTriInterpolator(tri_xy, np.asarray(delta[:, 1], dtype=np.float64))
    # 仅在 dst 外扩包围盒内评估，盒外恒等映射
    x0, y0 = np.floor(dst_pts.min(axis=0)).astype(int) - 4
    x1, y1 = np.ceil(dst_pts.max(axis=0)).astype(int) + 4
    x0, y0 = max(0, int(x0)), max(0, int(y0))
    x1, y1 = min(w, int(x1)), min(h, int(y1))
    gx, gy = np.meshgrid(np.arange(x0, x1, dtype=np.float64),
                         np.arange(y0, y1, dtype=np.float64))
    # LinearTriInterpolator 返回 MaskedArray，凸包外置 NaN 后按恒等处理
    dx = np.ma.filled(fx(gx, gy), np.nan)
    dy = np.ma.filled(fy(gx, gy), np.nan)
    sx = (gx - dx).astype(np.float32)
    sy = (gy - dy).astype(np.float32)
    bad = ~np.isfinite(sx) | ~np.isfinite(sy)
    sx[bad] = gx[bad]
    sy[bad] = gy[bad]
    # 全场映射：框外恒等
    mx = np.tile(np.arange(w, dtype=np.float32), (h, 1))
    my = np.tile(np.arange(h, dtype=np.float32)[:, None], (1, w))
    mx[y0:y1, x0:x1] = sx
    my[y0:y1, x0:x1] = sy
    return cv2.remap(img, mx, my, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REFLECT_101)


def _mesh_affine_warp(img, src_pts, dst_pts, tri):
    """FaceVerse 三角网格逐三角仿射（拓扑由模型给出，免剖分）"""
    out = np.zeros_like(img)
    h, w = img.shape[:2]
    for simplex in tri:
        tri_dst = np.ascontiguousarray(dst_pts[simplex], dtype=np.float32)
        tri_src = np.ascontiguousarray(src_pts[simplex], dtype=np.float32)
        r1 = cv2.boundingRect(tri_dst)
        r2 = cv2.boundingRect(tri_src)
        if r1[2] <= 0 or r1[3] <= 0 or r2[2] <= 0 or r2[3] <= 0:
            continue
        t1 = np.ascontiguousarray(tri_dst - [r1[0], r1[1]], dtype=np.float32)
        t2 = np.ascontiguousarray(tri_src - [r2[0], r2[1]], dtype=np.float32)
        if t1.max() > 4000 or t2.max() > 4000:
            continue
        mask = np.zeros((r1[3], r1[2], 3), dtype=np.float32)
        cv2.fillConvexPoly(mask, np.int32(t1), (1.0, 1.0, 1.0), 16, 0)
        patch = img[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]
        if patch.size == 0:
            continue
        M, _ = cv2.estimateAffine2D(t2, t1, method=cv2.LMEDS)
        if M is None or M.shape[0] != 2:
            M = cv2.getAffineTransform(t2, t1)
        warped = cv2.warpAffine(patch, M, (r1[2], r1[3]),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REFLECT_101)
        roi = out[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
        roi[:] = roi * (1.0 - mask) + warped * mask
    # 网格未覆盖区域用原图
    face_region = (out.sum(axis=2) == 0)
    out[face_region] = img[face_region]
    return out


NODE_CLASS_MAPPINGS = {
    "FaceSculptLandmarks": FaceSculptLandmarks,
    "FaceSculptEdit": FaceSculptEdit,
    "FaceVerseRecon": FaceVerseReconNode,
    "FaceVerseSculpt": FaceVerseSculpt,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "FaceSculptLandmarks": "人脸关键点检测 (FaceSculpt)",
    "FaceSculptEdit": "人脸参数微调 (FaceSculpt)",
    "FaceVerseRecon": "FaceVerse 3D人脸重建",
    "FaceVerseSculpt": "FaceVerse 骨相微调",
}
