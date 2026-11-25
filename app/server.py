"""ImplicitMABSA — local web app: serves the HTML/CSS/JS UI in app/web/ and answers POST /api/predict
with the project's own trained models (the same exported checkpoints the Gradio tester uses).

    pip install -r app/requirements.txt
    python app/server.py --models-dir models
Open http://127.0.0.1:8000
"""
import argparse
import io
import re
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from transformers import AutoTokenizer, ViTImageProcessor, ViTModel

from test_model import SENTIMENT_NAMES, load_models

WEB_DIR = Path(__file__).resolve().parent / "web"
# plain names promised to the UI; 2015 uses OTHER, 2017 uses MISC for the same "other named entity" class
TYPE_KEYS = {"PER": "person", "LOC": "location", "ORG": "organisation", "OTHER": "other", "MISC": "other"}
MAX_IMAGE_BYTES = 15 * 1024 * 1024


def create_app(models_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = load_models(models_dir, device)
    infos = [i for _, i in models.values()]
    toks = {e: AutoTokenizer.from_pretrained(e) for e in {i["encoder"] for i in infos if i["encoder"]}}
    vit_name = infos[0]["vit_model"]
    vit_proc = ViTImageProcessor.from_pretrained(vit_name)
    vit = ViTModel.from_pretrained(vit_name, add_pooling_layer=False).to(device).eval()
    datasets = sorted({ds for _, ds in models})

    app = FastAPI(title="ImplicitMABSA")

    def error(msg, status=400):
        return JSONResponse({"error": msg}, status_code=status)

    @app.get("/api/info")
    def info():
        return {"datasets": datasets, "device": str(device),
                "models": {f"{t}/{ds}": {"model": i["model"], "setting": i["setting"], "seed": i["seed"],
                                         "test_acc": i["test_acc"], "macro_f1": i["test_macro_f1"]}
                           for (t, ds), (_, i) in models.items()}}

    @app.post("/api/predict")
    async def predict(dataset: str = Form(""), tweet: str = Form(""), aspect: str = Form(""),
                      image: UploadFile | None = File(None)):
        tweet, aspect = tweet.strip(), aspect.strip()
        if dataset not in datasets:
            return error(f"Unknown dataset '{dataset}'. Choose one of: {', '.join(datasets)}.")
        if not tweet:
            return error("Type a tweet first.")
        if image is None:
            return error("Upload an image: these are text + image models.")
        raw = await image.read()
        if not raw:
            return error("The uploaded image is empty.")
        if len(raw) > MAX_IMAGE_BYTES:
            return error("The image is larger than 15 MB.", 413)
        try:
            pil = Image.open(io.BytesIO(raw)).convert("RGB")
        except (UnidentifiedImageError, OSError):
            return error("That file couldn't be read as an image. Use a JPG or PNG.")

        hidden = bool(aspect) and re.search(re.escape(aspect), tweet, flags=re.IGNORECASE) is not None
        try:
            with torch.no_grad():
                pix = vit_proc(images=pil, return_tensors="pt")["pixel_values"].to(device)
                tokens = vit(pixel_values=pix).last_hidden_state      # [1, 197, 768], same features as training
                out, used, text_seen = {}, {}, tweet
                for task in ("sentiment", "aspect_type"):
                    if (task, dataset) not in models:
                        return error(f"No {task} model is loaded for {dataset}.", 500)
                    m, inf = models[(task, dataset)]
                    tok = toks[inf["encoder"]]
                    text = re.sub(re.escape(aspect), tok.mask_token, tweet, flags=re.IGNORECASE) if hidden else tweet
                    text_seen = text
                    enc = tok([text], truncation=True, max_length=inf["max_len"], return_tensors="pt").to(device)
                    probs = torch.softmax(m(enc["input_ids"], enc["attention_mask"], tokens[:, 0], tokens), -1)[0].tolist()
                    names = SENTIMENT_NAMES if task == "sentiment" else TYPE_KEYS
                    out[task] = {names.get(c, c): round(float(p), 4) for c, p in zip(inf["classes"], probs)}
                    used[task] = {"model": inf["model"], "test_acc": round(inf["test_acc"], 4),
                                  "macro_f1": round(inf["test_macro_f1"], 4)}
        except Exception as e:  # report, don't crash the server
            return error(f"The model failed on this input: {e}", 500)

        return {"sentiment": out["sentiment"], "aspect_type": out["aspect_type"],
                "text_seen": text_seen, "aspect_hidden": hidden, "models": used}

    @app.get("/")
    def home():
        return RedirectResponse("/index.html")

    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    app = create_app(args.models_dir)
    print(f"\nImplicitMABSA is running: open http://{args.host}:{args.port}\n")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
