import os
import io
import math
from typing import Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

import timm
import streamlit as st
import requests


# ---------------------
# Config / Constants
# ---------------------
CLASS_NAMES = [
    'dyed-lifted-polyps', 'dyed-resection-margins', 'esophagitis',
    'normal-cecum', 'normal-pylorus', 'normal-z-line', 'polyps', 'ulcerative-colitis'
]
NUM_CLASSES = len(CLASS_NAMES)
IMAGE_SIZE = 224
DEFAULT_WEIGHTS = 'clinFusionNet_best.pth'  # or set a direct URL to auto-download


# ---------------------
# Lightweight medical prior used by the model
# ---------------------
class MedicalKnowledgeBase:
    def __init__(self):
        self.clinical_attention_weights = {
            'polyps': 1.0,
            'ulcerative-colitis': 0.95,
            'esophagitis': 0.9,
            'dyed-lifted-polyps': 0.85,
            'dyed-resection-margins': 0.8,
            'normal-cecum': 0.7,
            'normal-pylorus': 0.65,
            'normal-z-line': 0.6
        }

    def get_medical_attention_weight(self, class_name: str) -> float:
        return self.clinical_attention_weights.get(class_name, 0.5)


medical_kb = MedicalKnowledgeBase()


# ---------------------
# Model definition (inference only)
# ---------------------
class MedicalAttention(nn.Module):
    def __init__(self, feature_dim: int = 768, num_classes: int = 8, dropout: float = 0.25):
        super().__init__()
        self.medical_embedding = nn.Parameter(torch.randn(num_classes, 64))
        self.query = nn.Linear(feature_dim, 64)
        self.key = nn.Linear(64, 64)
        self.value = nn.Linear(feature_dim, feature_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if features.dim() == 2:
            features = features.unsqueeze(1)
        B, N, D = features.shape
        q = self.query(features)
        k = self.key(self.medical_embedding).unsqueeze(0).expand(B, -1, -1)
        attn = self.softmax(torch.matmul(q, k.transpose(1, 2)) / math.sqrt(64))
        attn = self.dropout(attn)
        v = self.value(features)
        attended = torch.matmul(attn.transpose(1, 2), v)
        return attended.mean(1), attn.mean(1)


class ClinFusionNet(nn.Module):
    def __init__(
        self,
        num_classes: int = 8,
        backbone_pretrained: bool = False,
        cnn_backbone_name: str = 'convnext_xlarge_384_in22ft1k',
        vit_backbone_name: str = 'swin_large_patch4_window7_224',
    ):
        super().__init__()
        # Use pretrained=False to avoid large downloads; weights come from the checkpoint
        self.cnn_backbone = timm.create_model(
            cnn_backbone_name, pretrained=backbone_pretrained, features_only=True
        )
        self.cnn_feature_dim = self.cnn_backbone.feature_info[-1]['num_chs']

        self.vit_backbone = timm.create_model(
            vit_backbone_name, pretrained=backbone_pretrained, num_classes=0
        )
        self.vit_feature_dim = self.vit_backbone.num_features

        self.medical_attention = MedicalAttention(self.vit_feature_dim, num_classes, dropout=0.25)

        self.cnn_adapter = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.BatchNorm1d(self.cnn_feature_dim),
            nn.Linear(self.cnn_feature_dim, 768), nn.LayerNorm(768), nn.GELU(), nn.Dropout(0.3)
        )
        self.vit_adapter = nn.Sequential(
            nn.BatchNorm1d(self.vit_feature_dim),
            nn.Linear(self.vit_feature_dim, 768), nn.LayerNorm(768), nn.GELU(), nn.Dropout(0.3)
        )

        self.fusion_layer = nn.Sequential(
            nn.Linear(1536, 768), nn.LayerNorm(768), nn.GELU(), nn.Dropout(0.4),
            nn.Linear(768, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.3)
        )
        self.classifier = nn.Linear(512, num_classes)

        self._init_medical_weights()

    def _init_medical_weights(self):
        with torch.no_grad():
            for i, cname in enumerate(CLASS_NAMES):
                w = medical_kb.get_medical_attention_weight(cname)
                self.medical_attention.medical_embedding.data[i] *= w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cnn_feat = self.cnn_backbone(x)[-1]
        cnn_vec = self.cnn_adapter(cnn_feat)
        vit_tok = self.vit_backbone(x)
        vit_vec, _ = self.medical_attention(vit_tok)
        vit_vec = self.vit_adapter(vit_vec)
        fused = self.fusion_layer(torch.cat([cnn_vec, vit_vec], dim=1))
        logits = self.classifier(fused)
        return logits


# ---------------------
# Utilities
# ---------------------
def _clean_state_dict_keys(state_dict: dict) -> dict:
    # Remove 'module.' prefix if present (DDP/DataParallel)
    if any(k.startswith('module.') for k in state_dict.keys()):
        return {k[len('module.'):]: v for k, v in state_dict.items()}
    return state_dict


def _download_if_url(path_or_url: str) -> str:
    if path_or_url.lower().startswith('http://') or path_or_url.lower().startswith('https://'):
        os.makedirs('weights', exist_ok=True)
        local_name = os.path.basename(path_or_url.split('?')[0]) or 'model.pth'
        local_path = os.path.join('weights', local_name)
        if os.path.isfile(local_path) and os.path.getsize(local_path) > 0:
            return local_path
        with requests.get(path_or_url, stream=True, timeout=120) as r:
            r.raise_for_status()
            total = int(r.headers.get('content-length', 0))
            chunk = 1024 * 1024
            downloaded = 0
            with open(local_path, 'wb') as f:
                for data in r.iter_content(chunk_size=chunk):
                    if data:
                        f.write(data)
                        downloaded += len(data)
                        if total:
                            st.progress(min(1.0, downloaded / total))
        return local_path
    return path_or_url


@st.cache_resource(show_spinner=False)
def load_model(weights_path: str, device: torch.device, model_size: str) -> nn.Module:
    if model_size == 'Large (ConvNeXt-XL + Swin-L)':
        cnn_name = 'convnext_xlarge_384_in22ft1k'
        vit_name = 'swin_large_patch4_window7_224'
    else:
        # Lite demo (faster on CPU). Note: your large checkpoint will not match.
        cnn_name = 'convnext_tiny'
        vit_name = 'swin_tiny_patch4_window7_224'

    model = ClinFusionNet(
        num_classes=NUM_CLASSES,
        backbone_pretrained=False,
        cnn_backbone_name=cnn_name,
        vit_backbone_name=vit_name,
    )
    model.to(device)

    # Only try loading weights when using Large model (matching your training)
    if model_size == 'Large (ConvNeXt-XL + Swin-L)':
        path = _download_if_url(weights_path)
        if os.path.isfile(path):
            state = torch.load(path, map_location=device)
            if isinstance(state, dict) and 'state_dict' in state and isinstance(state['state_dict'], dict):
                state = state['state_dict']
            state = _clean_state_dict_keys(state)
            try:
                model.load_state_dict(state, strict=True)
            except Exception:
                model.load_state_dict(state, strict=False)
    model.eval()
    return model


def preprocess_image(image: Image.Image) -> torch.Tensor:
    tfm = T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return tfm(image.convert('RGB'))


def predict(model: nn.Module, img_tensor: torch.Tensor, device: torch.device):
    with torch.no_grad():
        img = img_tensor.unsqueeze(0).to(device)
        logits = model(img)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    return probs


# ---------------------
# Streamlit UI
# ---------------------
st.set_page_config(page_title='ClinFusion-Net Demo', layout='centered')
st.title('ClinFusion-Net — Endoscopic Image Classification')
st.caption('Upload an image to classify. Model: ConvNeXtXL + SwinL fusion.')

col1, col2 = st.columns([2, 1])
with col1:
    weights_file = st.text_input('Weights file path or URL', value=DEFAULT_WEIGHTS, help='Provide a local filename (e.g., clinFusionNet_best.pth) or a direct URL to the .pth file')
with col2:
    model_size = st.selectbox('Model size', ['Large (ConvNeXt-XL + Swin-L)', 'Lite (ConvNeXt-T + Swin-T)'], index=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
st.write(f'Running on: `{device.type}`')

@st.cache_data(show_spinner=False)
def load_example_image_bytes() -> bytes:
    return b''

uploaded = st.file_uploader('Upload an image (jpg, jpeg, png)', type=['jpg', 'jpeg', 'png'])

if uploaded is not None:
    try:
        pil = Image.open(uploaded)
    except Exception:
        st.error('Could not read the uploaded file as an image.')
        st.stop()

    st.image(pil, caption='Input image', use_container_width=True)

    with st.spinner('Loading model… this may take a minute the first time.'):
        model = load_model(weights_file, device, model_size)

    img_tensor = preprocess_image(pil)
    with st.spinner('Running inference…'):
        probs = predict(model, img_tensor, device)

    pred_idx = int(np.argmax(probs))
    pred_label = CLASS_NAMES[pred_idx]
    st.success(f'Prediction: {pred_label} (confidence {probs[pred_idx]:.2%})')

    st.subheader('Class probabilities')
    st.bar_chart({
        'probability': {cls: float(p) for cls, p in zip(CLASS_NAMES, probs)}
    })

else:
    st.info('Upload an image to get started.')
