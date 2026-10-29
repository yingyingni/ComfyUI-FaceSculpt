# -*- coding: utf-8 -*-
# ComfyUI-Manager install hook / 手动安装脚本
# 用法: python install.py
# 1. pip 安装依赖
# 2. 下载 mediapipe face_landmarker.task（FaceSculptLandmarks 必需，3.7MB）
# 3. 克隆 FaceVerse_v4 + 下载权重（FaceVerseRecon/FaceVerseSculpt 需要，~280MB）
# 4. 编译 Sim3DR（可选，仅 FaceVerseRecon render=True 用；失败不影响骨相扭曲）
import os
import subprocess
import sys
import urllib.request
import json

HERE = os.path.dirname(os.path.abspath(__file__))
LANDMARKER_URL = ("https://storage.googleapis.com/mediapipe-models/"
                  "face_landmarker/face_landmarker/float16/1/face_landmarker.task")
FV_REPO = "https://github.com/LizhenWangT/FaceVerse_v4"
FV_RELEASE_API = "https://api.github.com/repos/Mrkomiljon/faceverse-onnx/releases/latest"
FV_FILES = ["faceverse_v4_2.npy", "faceverse_resnet50.pth"]


def sh(cmd, cwd=None, check=True):
    print(f"[install] $ {cmd}", flush=True)
    return subprocess.run(cmd, shell=True, cwd=cwd, check=check)


def pip_install():
    req = os.path.join(HERE, "requirements.txt")
    sh(f'"{sys.executable}" -m pip install -r "{req}"', check=False)


def download(url, dst):
    if os.path.isfile(dst) and os.path.getsize(dst) > 0:
        print(f"[install] 已存在，跳过: {dst}")
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    print(f"[install] 下载 {url} -> {dst}", flush=True)
    urllib.request.urlretrieve(url, dst)


def install_landmarker():
    download(LANDMARKER_URL, os.path.join(HERE, "models", "face_landmarker.task"))


def install_faceverse():
    fv_dir = os.path.join(HERE, "FaceVerse_v4")
    if not os.path.isdir(fv_dir):
        sh(f"git clone --depth 1 {FV_REPO} \"{fv_dir}\"")
    data_dir = os.path.join(fv_dir, "data")
    missing = [f for f in FV_FILES if not os.path.isfile(os.path.join(data_dir, f))]
    if missing:
        try:
            with urllib.request.urlopen(FV_RELEASE_API) as r:
                rel = json.load(r)
            assets = {a["name"]: a["browser_download_url"] for a in rel.get("assets", [])}
            for f in missing:
                if f in assets:
                    download(assets[f], os.path.join(data_dir, f))
                else:
                    print(f"[install] ⚠ release 无资产 {f}，请手动下载后放入 {data_dir}")
        except Exception as e:
            print(f"[install] ⚠ 自动下载失败({e})，请手动获取 FaceVerse 权重放入 {data_dir}:")
            print(f"  {FV_REPO} 仓库 README 内 OneDrive/Google Drive 链接，或")
            print(f"  https://github.com/Mrkomiljon/faceverse-onnx/releases")
    # Sim3DR Cython 编译（可选）
    sim3dr = os.path.join(fv_dir, "Sim3DR")
    if os.path.isdir(sim3dr):
        r = sh(f'"{sys.executable}" setup.py build_ext --inplace', cwd=sim3dr, check=False)
        if r.returncode != 0:
            print("[install] ⚠ Sim3DR 编译失败：FaceVerseSculpt/扭曲功能不受影响，"
                  "仅 FaceVerseRecon render=True 的网格预览不可用（需 Cython+编译链）")


if __name__ == "__main__":
    pip_install()
    install_landmarker()
    install_faceverse()
    print("[install] 完成。重启 ComfyUI 后节点生效。")
