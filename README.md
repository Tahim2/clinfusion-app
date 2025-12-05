# ClinFusion-Net Streamlit App

A minimal Streamlit app to run inference with your ClinFusion-Net model (ConvNeXt-XL + Swin-L fusion with a small medical attention block).

## Files
- `app.py` — Streamlit UI and model definition for inference
- `requirements.txt` — Python dependencies
- `clinFusionNet_best.pth` — Your trained weights (place alongside `app.py`), or provide a URL in the app

## Local Run (Windows PowerShell)
```powershell
cd c:\Users\User\Downloads\ClinFusion
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```
Then open the browser link shown in the terminal. Keep `Weights file path or URL` as `clinFusionNet_best.pth` if the file is in the same folder, or paste a direct download URL.

Notes:
- The large backbones are heavy. On CPU, first-time load/inference can be slow. If you only have CPU, try the `Lite (ConvNeXt-T + Swin-T)` option (it won’t use your large checkpoint).
- If you have a GPU, PyTorch will use it automatically.

## Streamlit Community Cloud Deployment
1. Push these files to a GitHub repo at the repo root:
   - `app.py`
   - `requirements.txt`
   - (Optional) `clinFusionNet_best.pth` — only if < ~500MB and allowed by your repo policy
2. Alternatively, host the weights externally (GitHub Releases, Hugging Face, S3) and paste the URL into the app’s input field.
3. In Streamlit Cloud:
   - "New app" → Select your repo/branch → set `file` to `app.py`.
   - Click Deploy.

### Tips for Cloud
- If deployment times out or memory is limited, use the `Lite` model option to verify the UI works.
- Prefer hosting weights via a direct URL; the app will download and cache them on first run.
- If `timm` model downloads are attempted: we set `pretrained=False` and rely on your checkpoint, so there should be no large external downloads.

## Usage
- Upload an endoscopic image (JPG/PNG).
- The app shows the predicted class and a probability bar chart.

## Troubleshooting
- "Could not read uploaded file": ensure the file is an image (jpg/jpeg/png).
- "Weight mismatch" or poor predictions: confirm you’re using `Large (ConvNeXt-XL + Swin-L)` with your trained `clinFusionNet_best.pth`.
- Install issues on Cloud: pin versions in `requirements.txt` as needed; consider smaller backbones in the interim.
