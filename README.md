# ComfyUI-FaceSculpt

人脸参数化微调 ComfyUI 自定义节点包（医美微调 / Parametric facial micro-editing）。

两条引擎：

- **TPS 微调引擎**（`FaceSculptLandmarks` + `FaceSculptEdit`）：MediaPipe 478 关键点 → 语义区域加权位移 → Delaunay 三角仿射扭曲 + 凸包羽化融合。**纯像素位移、不生成新像素**，身份 100% 保留，适合 ≤8% 幅度的眼/鼻/唇/下颌微调（美图类产品的同款底层原理）。
- **FaceVerse 骨相引擎**（`FaceVerseRecon` + `FaceVerseSculpt`）：FaceVerse V4（东亚人脸专项 3DMM）ResNet50 单图回归 621 维参数 → 形变后 3D 顶点投影回 2D → 投影锚点网格扭曲**原照片**。纹理/光影/发型零损失，支撑更大幅度的脸型/鼻梁/下巴调整。

## 安装

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/<your-org>/ComfyUI-FaceSculpt.git
cd ComfyUI-FaceSculpt
python install.py        # 装依赖 + 下 face_landmarker.task + 克隆 FaceVerse_v4 + 下权重 + 编译 Sim3DR
# 重启 ComfyUI
```

手动安装等价步骤：

```bash
pip install -r requirements.txt
# 1) face_landmarker.task (3.7MB) → models/face_landmarker.task
#    https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
# 2) FaceVerse_v4（仅骨相节点需要）：
git clone https://github.com/LizhenWangT/FaceVerse_v4
#    data/faceverse_v4_2.npy + data/faceverse_resnet50.pth
#    官方 README 内 OneDrive/Google Drive，或 https://github.com/Mrkomiljon/faceverse-onnx/releases
# 3) 可选：cd FaceVerse_v4/Sim3DR && python setup.py build_ext --inplace
```

`FaceVerse_v4` 可放在本包目录内、`custom_nodes/` 同级，或 `export FACE_VERSE_DIR=/path/to/FaceVerse_v4`。

## 节点

| 节点 | 输入 | 输出 | 说明 |
|---|---|---|---|
| FaceSculptLandmarks | image, draw | LANDMARKS, IMAGE | MediaPipe 478 点检测，输出 JSON 坐标；`未检测到人脸` 时抛错 |
| FaceSculptEdit | image, landmarks, 12 参数 | IMAGE | 语义微调，幅度 [-1,1]，推荐 ≤0.6 |
| FaceVerseRecon | image, device, render | FVDATA, IMAGE | 3DMM 重建；render 输出网格预览图（需 Sim3DR） |
| FaceVerseSculpt | image, fvdata, 参数 | IMAGE, LANDMARKS | 3D 参数形变→锚点网格扭曲原图 |

### FaceSculptEdit 参数（幅度均为 -1.0~1.0，步进 0.05，推荐甜区 ≤0.6）

`nose_lift` 鼻梁抬高 · `nose_tip_up` 鼻尖上翘 · `nose_shrink` 鼻翼缩小 · `eye_inner` 开内眼角 · `eye_big` 大眼 · `lip_full` 丰唇 · `mouth_width` 嘴角宽度 · `jaw_slim` 收下颌 · `cheek_slim` 瘦脸 · `brow_lift` 提眉 · `chin_down` 下巴延长 · `chin_shrink` 下巴缩短 · `blend_feather` 羽化

### FaceVerseSculpt 参数

`shape_scale` 脸型饱满度(-1瘦削~+1饱满) · `nose_lift` 鼻梁抬高 · `jaw_slim` 收下颌 · `chin_length` 下巴长度 · `shape_dim`/`shape_delta` 原始 156 维 shape PCA 轴调试口 · `blend_feather` 羽化

## 示例工作流

`workflow/face_sculpt_api.json`（2D 微调，~1.5s）· `workflow/face_sculpt_3d_api.json`（3D 骨相，首跑含模型加载 ~40s）

## 性能与边界

- TPS 路径：CPU 级速度，身份无损；>10% 幅度出现拉伸感属算法上限
- FaceVerse 路径：3D 重建驱动的 2D 网格扭曲（非纹理重渲染），无光影重构；`shape_dim` 语义轴需自行标定
- 多人脸：当前 `num_faces=1`，取画面中最大人脸

## License

代码 MIT。依赖各自许可：MediaPipe（Apache-2.0）、FaceVerse_v4（学术研究许可，商用需与作者确认）、OpenCV（Apache-2.0）、SciPy（BSD）。
