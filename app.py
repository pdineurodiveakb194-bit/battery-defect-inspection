
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from itertools import product
import io
import tempfile
import zipfile
import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from torchvision import transforms
import matplotlib
from matplotlib import colormaps
import matplotlib.cm as cm
import os
from pathlib import Path
import gdown

# ============================================================
# 基本設定
# ============================================================

st.set_page_config(
    page_title="電極コーティング欠陥検査",
    page_icon="🔍",
    layout="wide"
)

CLASSES = [
    "割れ",
    "剥離",
    "ピンホール",
    "その他の欠陥"
]

CLASS_KEYS = [
    "crack",
    "delamination",
    "pinhole",
    "other"
]

# ============================================================
# Session State 初期化
# ============================================================
def count_defects(results):

    counts = {
        "crack": 0,
        "delamination": 0,
        "pinhole": 0,
        "other": 0,
    }

    for result in results:

        pattern = result["pattern"]

        for key in CLASS_KEYS:

            if pattern[key]:
                counts[key] += 1

    counts["total"] = len(results)

    return counts

def initialize_state():

    defaults = {
        "page": "home",

        # 入力画像
        "images": [],

        # 判定結果
        "results": [],

        # 閾値
        "thresholds": {
            "crack": 0.395,
            "delamination": 0.395,
            "pinhole": 0.635,
            "other": 0.41,
        },

        # 表示条件
        "selected_patterns": [],

        # 現在表示している画像
        "current_index": 0,

        # 判定済みか
        "prediction_done": False,

        # トップページへ戻る確認
        "show_home_confirm": False,

        # 表示対象
        "filtered_results": [],
        "gradcam_zip_data": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

initialize_state()

# ============================================================
# 16パターン生成
# ============================================================

def generate_patterns():

    patterns = []

    for values in product([False, True], repeat=4):

        pattern = {
            "crack": values[0],
            "delamination": values[1],
            "pinhole": values[2],
            "other": values[3],
        }

        patterns.append(pattern)

    return patterns

PATTERNS = generate_patterns()

def pattern_label(pattern):

    labels = []

    for key, name in zip(CLASS_KEYS, CLASSES):

        if pattern[key]:
            labels.append(f":red[{name}あり]")
        else:
            labels.append(f"{name}なし")

    return "　".join(labels)

# ============================================================
# デバイス
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ============================================================
# モデルの読み込み
# ============================================================
def create_convnext_model():
    model = models.convnext_tiny(weights=None)

    model.classifier[2] = nn.Linear(
        model.classifier[2].in_features,
        4
    )

    return model

# Google Driveからモデルをダウンロードする保存先
MODEL_DIR = Path("/tmp/battery_defect_models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

@st.cache_resource
def download_model(file_id, filename):
    """Google Driveからモデルをダウンロードする"""
    model_path = MODEL_DIR / filename

    if not model_path.exists() or model_path.stat().st_size == 0:
        if model_path.exists():
            model_path.unlink()

        result = gdown.download(
            id=file_id,
            output=str(model_path),
            quiet=False
        )

        if result is None or not model_path.exists():
            raise RuntimeError(
                f"モデルをダウンロードできませんでした: {filename}。"
                "Google Driveの共有設定を確認してください。"
            )

    return str(model_path)


def load_models(device_type):
    # 300×300用モデル
    model_300_path = download_model(
        "1RLUv0yegdmIxavAoYZPz2VF5unEal6Hv",
        "convnext_tiny_a1_model.pth"
    )

    model_300 = create_convnext_model()
    checkpoint_300 = torch.load(
        model_300_path,
        map_location=device_type
    )
    model_300.load_state_dict(
        checkpoint_300["model_state_dict"]
    )
    model_300.to(device_type)
    model_300.eval()

    # 224×224用モデル
    model_224_path = download_model(
        "1ySyDV0EbiYVr6ydDFRlaBq8_fR5CRWUr",
        "convnext_tiny_b2_model.pth"
    )

    model_224 = create_convnext_model()
    checkpoint_224 = torch.load(
        model_224_path,
        map_location=device_type
    )
    model_224.load_state_dict(
        checkpoint_224["model_state_dict"]
    )
    model_224.to(device_type)
    model_224.eval()

    return model_300, model_224


model_300, model_224 = load_models(str(device))

# ============================================================
# 画像の前処理
# ============================================================

transform_300 = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

transform_224 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ============================================================
# クラスごとの混合比率
# クラスの順番：
# [Surface_Crack, Delamination, Pinhole, unclassified]
# ============================================================

WEIGHTS_300 = torch.tensor(
    [0.6, 0.5, 0.4, 0.4],
    dtype=torch.float32,
    device=device
)

WEIGHTS_224 = torch.tensor(
    [0.4, 0.5, 0.6, 0.6],
    dtype=torch.float32,
    device=device
)

# ============================================================
# 実際の推論処理
# ConvNeXt-Tiny 2モデルの混合予測
# ============================================================

def predict_image(image):

    # PIL画像をRGBに変換
    image = image.convert("RGB")

    # --------------------------------------------------------
    # 300×300用の前処理
    # --------------------------------------------------------

    input_300 = transform_300(image)

    # バッチ次元を追加
    # [3, 300, 300] → [1, 3, 300, 300]
    input_300 = input_300.unsqueeze(0)

    # 使用デバイスへ移動
    input_300 = input_300.to(device)


    # --------------------------------------------------------
    # 224×224用の前処理
    # --------------------------------------------------------

    input_224 = transform_224(image)

    # バッチ次元を追加
    # [3, 224, 224] → [1, 3, 224, 224]
    input_224 = input_224.unsqueeze(0)

    # 使用デバイスへ移動
    input_224 = input_224.to(device)


    # --------------------------------------------------------
    # 推論
    # --------------------------------------------------------

    with torch.no_grad():

        # 各モデルの出力（logits）
        logits_300 = model_300(input_300)
        logits_224 = model_224(input_224)

        # sigmoidでクラスごとの確率に変換
        probs_300 = torch.sigmoid(logits_300)
        probs_224 = torch.sigmoid(logits_224)

        # クラスごとの重みで混合
        final_probs = (
            probs_300 * WEIGHTS_300
            + probs_224 * WEIGHTS_224
        )

    # --------------------------------------------------------
    # 結果を辞書形式で返す
    # --------------------------------------------------------

    # バッチ次元を削除
    final_probs = final_probs.squeeze(0)

    return {
        "crack": float(final_probs[0].item()),
        "delamination": float(final_probs[1].item()),
        "pinhole": float(final_probs[2].item()),
        "other": float(final_probs[3].item()),
    }

# ============================================================
# 閾値による「あり / なし」判定
# ============================================================

def get_pattern(probabilities, thresholds):

    pattern = {}

    for key in CLASS_KEYS:
        pattern[key] = (
            probabilities[key] >= thresholds[key]
        )

    return pattern

# ============================================================
# 判定結果のフィルタリング
# ============================================================

def filter_results(results, selected_patterns):

    if not selected_patterns:
        return []

    filtered = []

    for result in results:

        pattern = result["pattern"]

        for selected in selected_patterns:

            if pattern == selected:
                filtered.append(result)
                break

    return filtered

# ============================================================
# トップページ
# ============================================================

def home_page():

    st.title("🔍 電極コーティング欠陥検査システム")

    st.markdown(
        """
        電極コーティング画像をAIで解析し、
        割れ・剥離・ピンホール・その他の欠陥を判定します。
        """
    )

    st.divider()

    # ========================================================
    # ① 判定対象画像の選択
    # ========================================================

    st.subheader("① 判定対象画像の選択")

    uploaded_files = st.file_uploader(
        "画像を選択してください",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if uploaded_files:

        st.success(
            f"{len(uploaded_files)}枚の画像が選択されています。"
        )

        st.session_state.images = uploaded_files

        st.write("選択された画像")

        for file in uploaded_files:
            st.write(f"- {file.name}")

    st.divider()

    # ========================================================
    # ② 閾値設定（トップページに移動）
    # ========================================================

    st.subheader("② 欠陥判定の閾値設定")

    col1, col2 = st.columns(2)

    with col1:

        crack = st.select_slider(
            "割れ",
            options=[0.105, 0.395, 0.955],
            value=0.395,
            format_func=lambda x: {
                0.105: "見逃し少なめ(0.105)",
                0.395: "標準(0.395)",
                0.955: "誤認少なめ(0.955)"
            }[x]
        )

        pinhole = st.select_slider(
            "ピンホール",
            options=[0.415, 0.635, 0.900],
            value=0.635,
            format_func=lambda x: {
                0.415: "見逃し少なめ(0.415)",
                0.635: "標準(0.635)",
                0.9: "誤認少なめ(0.900)"
            }[x]
        )

    with col2:

        delamination = st.select_slider(
            "剥離",
            options=[0.125, 0.395, 0.665],
            value=0.395,
            format_func=lambda x: {
                0.125: "見逃し少なめ(0.125)",
                0.395: "標準(0.395)",
                0.665: "誤認少なめ(0.665)"
            }[x]
        )

        other = st.select_slider(
            "その他の欠陥",
            options=[0.205, 0.41, 0.8],
            value=0.41,
            format_func=lambda x: {
                0.205: "見逃し少なめ(0.205)",
                0.41: "標準(0.410)",
                0.8: "誤認少なめ(0.800)"
            }[x]
        )

    # 閾値を保存
    st.session_state.thresholds = {
        "crack": crack,
        "delamination": delamination,
        "pinhole": pinhole,
        "other": other,
    }

    st.divider()

    # ========================================================
    # ③ 判定実行
    # ========================================================

    st.subheader("③ 判定")

    if st.button(
        "判定実行",
        type="primary",
        use_container_width=True
    ):

        if not uploaded_files:

            st.warning(
                "判定する画像を選択してください。"
            )

            return

        with st.spinner("判定中です。しばらくお待ちください。"):

            results = []

            for file in uploaded_files:

                image = Image.open(file).convert("RGB")

                probabilities = predict_image(image)

                # トップページで設定した閾値を使用
                pattern = get_pattern(
                    probabilities,
                    st.session_state.thresholds
                )

                results.append(
                    {
                        "filename": file.name,
                        "image": image,
                        "probabilities": probabilities,
                        "pattern": pattern,

                        "gradcam": {
                            "crack": None,
                            "delamination": None,
                            "pinhole": None,
                            "other": None,
                        },
                        "gradcam_images": None
                    }
                )

        st.session_state.results = results
        st.session_state.filtered_results = []
        st.session_state.gradcam_zip_data = None
        st.session_state.prediction_done = True

        st.session_state.page = "settings"

        st.rerun()

# ============================================================
# 判定後ページ
# ============================================================

def settings_page():

    st.title("判定結果の設定")

    # ========================================================
    # ① 判定結果の集計表
    # ========================================================

    st.subheader("① 判定結果の集計")

    counts = count_defects(
        st.session_state.results
    )

    summary_df = pd.DataFrame(
        {
            "項目": [
                "割れあり",
                "剥離あり",
                "ピンホールあり",
                "その他の欠陥あり",
                "全画像数"
            ],

            "画像数": [
                counts["crack"],
                counts["delamination"],
                counts["pinhole"],
                counts["other"],
                counts["total"]
            ]
        }
    )

    st.table(summary_df)

    st.divider()

    # ========================================================
    # ② 表示する画像の条件
    # ========================================================

    st.subheader("② 表示する画像の条件")

    # --------------------------------------------------------
    # すべて表示
    # --------------------------------------------------------

    show_all = st.checkbox(
        "すべて表示",
        key="show_all"
    )

    st.divider()

    # --------------------------------------------------------
    # 一括選択ボタン
    # --------------------------------------------------------

    st.write("欠陥ありのパターンを一括選択")

    col1, col2, col3, col4 = st.columns(4)

    if col1.button("割れありをすべて選択"):
        selected = [
            p for p in PATTERNS if p["crack"]
        ]

        st.session_state.selected_patterns = selected

        for i, pattern in enumerate(PATTERNS):
            st.session_state[f"pattern_{i}"] = (
                pattern in selected
            )

        st.rerun()

    if col2.button("剥離ありをすべて選択"):
        selected = [
            p for p in PATTERNS if p["delamination"]
        ]

        st.session_state.selected_patterns = selected

        for i, pattern in enumerate(PATTERNS):
            st.session_state[f"pattern_{i}"] = (
                pattern in selected
            )

        st.rerun()

    if col3.button("ピンホールありをすべて選択"):
        selected = [
            p for p in PATTERNS if p["pinhole"]
        ]

        st.session_state.selected_patterns = selected

        for i, pattern in enumerate(PATTERNS):
            st.session_state[f"pattern_{i}"] = (
                pattern in selected
            )

        st.rerun()

    if col4.button("その他の欠陥ありをすべて選択"):
        selected = [
            p for p in PATTERNS if p["other"]
        ]

        st.session_state.selected_patterns = selected

        for i, pattern in enumerate(PATTERNS):
            st.session_state[f"pattern_{i}"] = (
                pattern in selected
            )

        st.rerun()

    if st.button("全ての選択を解除"):
        st.session_state.selected_patterns = []

        for i in range(len(PATTERNS)):
            st.session_state[f"pattern_{i}"] = False

        st.rerun()

    st.divider()

    # --------------------------------------------------------
    # 16パターン
    # --------------------------------------------------------

    st.write("表示する判定パターン")

    selected_patterns = []

    col1, col2 = st.columns(2)

    for i, pattern in enumerate(PATTERNS):

        label = pattern_label(pattern)

        column = col1 if i % 2 == 0 else col2

        with column:

            checked = pattern in st.session_state.selected_patterns

            value = st.checkbox(
                label,
                value=checked,
                key=f"pattern_{i}"
            )

            if value:
                selected_patterns.append(pattern)

    st.session_state.selected_patterns = selected_patterns

    st.divider()

    # --------------------------------------------------------
    # 表示対象数
    # --------------------------------------------------------

    if show_all:

        filtered_results = st.session_state.results

    else:

        filtered_results = filter_results(
            st.session_state.results,
            st.session_state.selected_patterns
        )

    st.info(
        f"表示対象：{len(filtered_results)} / "
        f"{len(st.session_state.results)} 枚"
    )

    st.divider()

    # --------------------------------------------------------
    # ボタン
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        if st.button(
            "表示ページへ",
            type="primary",
            use_container_width=True
        ):

            if show_all:

                st.session_state.filtered_results = (
                    st.session_state.results
                )

            else:

                st.session_state.filtered_results = (
                    filtered_results
                )

            st.session_state.current_index = 0
            st.session_state.page = "display"

            st.rerun()

    with col2:

        if st.button(
            "トップページへ戻る",
            use_container_width=True
        ):

            st.session_state.show_home_confirm = True

    # --------------------------------------------------------
    # 戻る確認
    # --------------------------------------------------------

    if st.session_state.show_home_confirm:

        st.warning(
            "トップページに戻ると判定結果は破棄されます。"
            "本当に戻りますか？"
        )

        col1, col2 = st.columns(2)

        with col1:

            if st.button("はい"):

                # 判定関連データを破棄
                st.session_state.results = []
                st.session_state.images = []
                st.session_state.filtered_results = []
                st.session_state.selected_patterns = []
                st.session_state.prediction_done = False
                st.session_state.current_index = 0
                st.session_state.show_home_confirm = False

                st.session_state.page = "home"

                st.rerun()

        with col2:

            if st.button("いいえ"):

                st.session_state.show_home_confirm = False

                st.rerun()

# ============================================================
# Grad-CAM
# ============================================================

def generate_gradcam(
    model,
    input_tensor,
    target_class,
    target_layer
):
    """
    1つのモデルからGrad-CAMを生成する。

    Parameters
    ----------
    model : PyTorch model
        ConvNeXt-Tinyモデル

    input_tensor : Tensor
        前処理済み画像
        shape: [1, 3, H, W]

    target_class : int
        Grad-CAMの対象クラス
        0: Surface_Crack
        1: Delamination
        2: Pinhole
        3: unclassified

    target_layer : nn.Module
        Grad-CAM対象の畳み込み特徴抽出層

    Returns
    -------
    cam : ndarray
        正規化済みGrad-CAM
        shape: [H, W]
    """

    activations = []
    gradients = []

    # --------------------------------------------------------
    # フック関数
    # --------------------------------------------------------

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    # フックを登録
    forward_handle = target_layer.register_forward_hook(
        forward_hook
    )

    backward_handle = target_layer.register_full_backward_hook(
        backward_hook
    )

    try:

        # 勾配を初期化
        model.zero_grad(set_to_none=True)

        # 推論
        logits = model(input_tensor)

        # 対象クラスのlogitを取得
        target_score = logits[0, target_class]

        # 対象クラスに対して逆伝播
        target_score.backward()

        # 特徴マップと勾配を取得
        activation = activations[0]
        gradient = gradients[0]

        # ----------------------------------------------------
        # ConvNeXtの特徴マップを確認
        # 通常は [B, C, H, W]
        # ----------------------------------------------------

        # チャネル方向の平均
        weights = gradient.mean(
            dim=(2, 3),
            keepdim=True
        )

        # 重み付き和
        cam = (weights * activation).sum(
            dim=1,
            keepdim=True
        )

        # ReLU
        cam = F.relu(cam)

        # 入力画像サイズに拡大
        cam = F.interpolate(
            cam,
            size=input_tensor.shape[2:],
            mode="bilinear",
            align_corners=False
        )

        # バッチ・チャネル次元を削除
        cam = cam.squeeze().detach().cpu().numpy()

        # 0～1に正規化
        cam_min = cam.min()
        cam_max = cam.max()

        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam

    finally:

        # フックを解除
        forward_handle.remove()
        backward_handle.remove()

target_layer_300 = model_300.features[-1]
target_layer_224 = model_224.features[-1]

# ============================================================
# Grad-CAMを元画像に重ねる
# ============================================================

def overlay_gradcam(image, cam, alpha=0.45):
    """
    元画像とGrad-CAMを合成する。

    Parameters
    ----------
    image : PIL.Image
        元画像

    cam : ndarray
        0～1に正規化されたGrad-CAM

    alpha : float
        ヒートマップの透明度

    Returns
    -------
    PIL.Image
        Grad-CAM合成画像
    """

    # 元画像をRGBに変換
    image = image.convert("RGB")

    # 元画像サイズ
    width, height = image.size

    # Grad-CAMを元画像サイズに変更
    cam_image = Image.fromarray(
        np.uint8(cam * 255)
    ).resize(
        (width, height),
        Image.Resampling.BILINEAR
    )

    # 0～1の配列に変換
    cam_array = np.asarray(cam_image).astype(
        np.float32
    ) / 255.0

    # Jetカラーマップを適用
    heatmap = colormaps["jet"](cam_array)

    # RGBのみ取り出して0～255に変換
    heatmap = np.uint8(
        heatmap[:, :, :3] * 255
    )

    heatmap_image = Image.fromarray(
        heatmap
    ).convert("RGB")

    # 元画像と合成
    result = Image.blend(
        image,
        heatmap_image,
        alpha=alpha
    )

    return result

# ============================================================
# 実際のGrad-CAM処理
# model_300を使用
# ============================================================

def create_gradcam(image, target_class):

    # RGB画像に変換
    image = image.convert("RGB")

    # 300×300用の前処理
    input_tensor = transform_300(image)

    # バッチ次元を追加
    input_tensor = input_tensor.unsqueeze(0)

    # デバイスへ移動
    input_tensor = input_tensor.to(device)

    # 勾配計算を有効にする
    input_tensor.requires_grad_(True)

    # 対象層
    target_layer = model_300.features[-1]

    # Grad-CAM生成
    cam = generate_gradcam(
        model=model_300,
        input_tensor=input_tensor,
        target_class=target_class,
        target_layer=target_layer
    )

    # 元画像に重ねる
    result = overlay_gradcam(
        image=image,
        cam=cam,
        alpha=0.45
    )

    return result

# ============================================================
# Grad-CAM画像のキャッシュ
# ============================================================

def get_gradcam_images(result):
    if result.get("gradcam_images") is not None:
        return result["gradcam_images"]

    image = result["image"]
    gradcam_images = [
        create_gradcam(image, target_class=0),
        create_gradcam(image, target_class=1),
        create_gradcam(image, target_class=2),
        create_gradcam(image, target_class=3),
    ]
    result["gradcam_images"] = gradcam_images
    return gradcam_images


# ============================================================
# 表示ページ
# ============================================================

def display_page():

    st.title("判定結果")

    results = st.session_state.filtered_results

    if not results:

        st.warning(
            "表示対象の画像がありません。"
        )

        if st.button("判定後ページへ"):

            st.session_state.page = "settings"
            st.rerun()

        return

    # --------------------------------------------------------
    # 現在の画像
    # --------------------------------------------------------

    index = st.session_state.current_index

    current = results[index]

    image = current["image"]

    st.subheader(
        f"{current['filename']}"
    )

    # --------------------------------------------------------
    # 元画像
    # --------------------------------------------------------

    st.image(
        image,
        caption=current["filename"],
        width=500
    )

    st.divider()

    # --------------------------------------------------------
    # 4種類のGrad-CAM
    # --------------------------------------------------------

    st.subheader("4種類のGrad-CAM")

    # 4クラス分のGrad-CAMを必要時に生成
    gradcam_images = get_gradcam_images(current)

    # 2列で表示
    cols = st.columns(2)

    for i, key in enumerate(CLASS_KEYS):

        probability = current["probabilities"][key]

        with cols[i % 2]:

            st.subheader(CLASSES[i])

            st.image(
                gradcam_images[i],
                use_container_width=True
            )

            st.metric(
                "存在確率",
                f"{probability:.1%}"
            )

    st.divider()

    # --------------------------------------------------------
    # 前へ / 次へ
    # --------------------------------------------------------

    col1, col2, col3 = st.columns([1, 2, 1])

    with col1:

        if st.button(
            "← 前へ",
            disabled=(index == 0),
            use_container_width=True
        ):

            st.session_state.current_index -= 1
            st.rerun()

    with col2:

        selected = st.selectbox(
            "画像を選択",
            range(len(results)),
            index=index,
            format_func=lambda i: results[i]["filename"]
        )

        if selected != index:

            st.session_state.current_index = selected
            st.rerun()

    with col3:

        if st.button(
            "次へ →",
            disabled=(index == len(results) - 1),
            use_container_width=True
        ):

            st.session_state.current_index += 1
            st.rerun()

    st.divider()

    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------

    st.subheader("判定結果CSV")

    csv_data = []

    for result in results:

        p = result["probabilities"]

        csv_data.append(
            {
                "画像ファイル名": result["filename"],
                "割れ存在確率": p["crack"],
                "剥離存在確率": p["delamination"],
                "ピンホール存在確率": p["pinhole"],
                "その他の欠陥存在確率": p["other"],
            }
        )

    df = pd.DataFrame(csv_data)

    csv_data = df.to_csv(
        index=False
    ).encode("utf-8-sig")

    st.download_button(
        "CSVをダウンロード",
        data=csv_data,
        file_name="inspection_results.csv",
        mime="text/csv"
    )

    st.divider()

    st.subheader("ヒートマップZIP出力")
    st.write("ダウンロードするヒートマップの条件を選択してください。")

    target_mode_label = st.radio(
        "出力対象クラス",
        ["閾値以上の欠陥クラスのみ", "すべてのクラス"],
        key="gradcam_target_mode"
    )
    target_mode = "threshold" if target_mode_label == "閾値以上の欠陥クラスのみ" else "all"

    folder_mode_label = st.radio(
        "フォルダ構成",
        ["画像ごとにフォルダ分け", "クラスごとにフォルダ分け"],
        key="gradcam_folder_mode"
    )
    folder_mode = "image" if folder_mode_label == "画像ごとにフォルダ分け" else "class"

    if st.button("ヒートマップZIPを生成", key="generate_gradcam_zip", type="primary"):
        zip_results = st.session_state.filtered_results
        if not zip_results:
            st.warning("ZIPに含める画像がありません。")
        else:
            with st.spinner("Grad-CAM画像とZIPを生成しています..."):
                st.session_state.gradcam_zip_data = create_heatmap_zip(
                    results=zip_results, target_mode=target_mode, folder_mode=folder_mode
                )
            st.success("Grad-CAM ZIPの生成が完了しました。")

    if st.session_state.get("gradcam_zip_data") is not None:
        st.download_button(
            label="ヒートマップZIPをダウンロード",
            data=st.session_state.gradcam_zip_data,
            file_name="heatmaps.zip", mime="application/zip", key="download_gradcam_zip"
        )

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("判定後ページへ戻る", use_container_width=True, key="back_to_settings_from_display"):
            st.session_state.page = "settings"
            st.rerun()
    with col2:
        if st.button("トップページへ戻る", use_container_width=True, key="back_to_home_from_display"):
            st.session_state.show_home_confirm = True

    if st.session_state.show_home_confirm:
        st.warning("トップページに戻ると判定結果は破棄されます。本当に戻りますか？")
        confirm_col1, confirm_col2 = st.columns(2)
        with confirm_col1:
            if st.button("はい", key="confirm_home_from_display"):
                st.session_state.results = []
                st.session_state.images = []
                st.session_state.filtered_results = []
                st.session_state.selected_patterns = []
                st.session_state.prediction_done = False
                st.session_state.current_index = 0
                st.session_state.gradcam_zip_data = None
                st.session_state.show_home_confirm = False
                st.session_state.page = "home"
                st.rerun()
        with confirm_col2:
            if st.button("いいえ", key="cancel_home_from_display"):
                st.session_state.show_home_confirm = False
                st.rerun()

def pil_image_to_bytes(image):
    """
    PIL画像をPNG形式のバイトデータに変換する
    """
    image_bytes = io.BytesIO()
    image.save(image_bytes, format="PNG")
    return image_bytes.getvalue()


def create_heatmap_zip(
    results,
    target_mode,
    folder_mode
):
    """
    Grad-CAM画像を生成し、ZIPファイルを作成する

    target_mode:
        "threshold" : 閾値以上のクラスのみ
        "all"       : すべてのクラス

    folder_mode:
        "image" : 画像ごとにフォルダ分け
        "class" : クラスごとにフォルダ分け
    """

    class_names = [
        "割れ",
        "剥離",
        "ピンホール",
        "その他の欠陥"
    ]

    class_keys = [
        "crack",
        "delamination",
        "pinhole",
        "other"
    ]

    # ZIPファイルを一時ファイル上に作成
    with tempfile.TemporaryFile(mode="w+b") as zip_buffer:

        with zipfile.ZipFile(
            zip_buffer,
            mode="w",
            compression=zipfile.ZIP_DEFLATED
        ) as zip_file:

            for result in results:

                filename = result["filename"]
                image_stem = os.path.splitext(filename)[0]
                pattern = result["pattern"]

                # 対象クラスを決定
                if target_mode == "threshold":

                    target_classes = [
                        i
                        for i, key in enumerate(class_keys)
                        if pattern[key]
                    ]

                else:
                    target_classes = list(range(4))

                # 対象クラスがなければ次の画像へ
                if not target_classes:
                    continue

                # この画像のGrad-CAMを1回だけ生成
                gradcam_images = get_gradcam_images(result)

                for target_class in target_classes:

                    class_name = class_names[target_class]
                    heatmap_image = gradcam_images[target_class]

                    # PNG形式のバイトデータに変換
                    image_bytes = pil_image_to_bytes(
                        heatmap_image
                    )

                    # ZIP内の保存先を決定
                    if folder_mode == "image":
                        zip_path = (
                            f"{image_stem}/"
                            f"{class_name}.png"
                        )
                    else:
                        zip_path = (
                            f"{class_name}/"
                            f"{image_stem}.png"
                        )

                    # ZIPに追加
                    zip_file.writestr(
                        zip_path,
                        image_bytes
                    )

                    del image_bytes

                # この画像のヒートマップへの参照を解放
                del gradcam_images

        # ZIP作成完了後、先頭から読み込む
        zip_buffer.seek(0)
        zip_bytes = zip_buffer.read()

    return zip_bytes

# ============================================================
# ページ切り替え
# ============================================================

if st.session_state.page == "home":

    home_page()

elif st.session_state.page == "settings":

    settings_page()

elif st.session_state.page == "display":

    display_page()
