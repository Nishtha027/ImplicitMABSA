# ImplicitMABSA: Multimodal Implicit Aspect-Based Sentiment Analysis Using Texts and Images

Code for a study of whether combining tweet text with its image helps (1) predict the sentiment toward an aspect
that is **not stated** in the text, and (2) identify **what kind of aspect** it is, on the Twitter-2015 and
Twitter-2017 multimodal ABSA benchmarks.

## Pipelinessssss

| Step | File | What it does |
|---|---|---|
| 1 | `src/01_vit_image_branch.py` (`notebooks/vit_image_branch.ipynb`) | Validates every image, builds a `sample_id` → image mapping, and extracts frozen `google/vit-base-patch16-224` features per split: CLS (768-d), mean-pooled patches (768-d) and all 197 tokens (float16). |
| 2 | `src/02_train_fusion.py` (`notebooks/train_fusion.ipynb`) | Trains and evaluates image-only, text-only, text+image concatenation and text+image cross-attention models on two tasks, 3 seeds each. |

Both scripts were run as Kaggle notebooks (GPU T4). They locate their inputs under `/kaggle/input` automatically;
paths can be overridden with the `IMABSA_*` environment variables at the top of each script. In the Kaggle editor
the training notebook runs a short smoke test; a committed run ("Save & Run All") runs the full experiment.

### Data (not included in this repository)
* Twitter-2015 / Twitter-2017 aspect-sentiment TSVs + images: Kaggle datasets `mdfaishalahmedrudroo/twitter2015`,
  `wentengh/twitter2017` (TomBERT release format). Original train / dev / test splits are kept.
* Entity tags for aspect types: Kaggle dataset `mocmeo/data-twitter-2015-2017` (multimodal NER release of the same
  tweets), aligned to each ABSA row by image id + aspect tokens.

## Testing the trained models on your own input

The research runs store metrics and predictions, not weights. To obtain the trained models themselves,
`02_train_fusion.py` has an export mode (`IMABSA_FIXED_ENCODER`, `IMABSA_EXPORT_RUNS`): it re-trains the listed
configurations with the same code, data, encoder, hyperparameters and seed, and saves each model to
`models/<task>_<dataset>_<setting>_<model>_seed<k>/` (`model_fp16.pt` + `model_info.json` with that model's own test
metrics). `notebooks/export_models.ipynb` exports the best fusion configuration per dataset (seed 42) for masked
aspect sentiment and for aspect-type identification (Kaggle notebook "ImplicitMABSA Model Export").

With the exported `models/` folder next to this README:

```
pip install -r app/requirements.txt
python app/test_model.py --models-dir models
```

Open http://127.0.0.1:7860, choose the dataset the models were trained on, type a tweet, optionally the aspect, and
upload an image. If the aspect appears in the tweet it is masked before prediction (the masked setting the models were
trained on); otherwise the tweet is used as is. Runs on CPU. Re-training a seed on GPU reproduces the reported run
closely but not bit-for-bit; each exported model reports its own test accuracy.

### Web app (local)

`app/web/` is an HTML/CSS/JS front end (Home + Lab pages); `app/server.py` serves it and answers
`POST /api/predict` (form fields `dataset`, `tweet`, `aspect`, `image`) with the same exported models and the same
masking as the tester above. `GET /api/info` lists the loaded models and their test metrics.

```
pip install -r app/requirements.txt
python app/server.py --models-dir models
```

Open http://127.0.0.1:8000 and go to **Lab**. On Windows, `app/START_WEBSITE.bat` does the same (it looks for
`models/` next to it, then one folder up). `lab.html?mock` shows the UI with a fixed sample answer, without models.

Sentiment labels in the source files: 0 = negative, 1 = neutral, 2 = positive (verified against the 4-line copies of
the same data, which use −1/0/1, on all 11,314 matched rows).

## Tasks and settings
* **Aspect sentiment** (3 classes, raw labels 0/1/2 as in the source files)
  * `masked`: the aspect term is replaced by the mask token and not given to the model (synthetic implicitness).
  * `explicit`: the aspect is in the sentence and also given as a second segment (standard benchmark setting).
* **Implicit aspect identification** (`aspect_type`): with the aspect masked, predict its entity type
  (2015: LOC / ORG / OTHER / PER; 2017: LOC / MISC / ORG / PER). Rows whose aspect could not be aligned to a tagged
  entity are excluded from this task only (2015: ~8–10% of rows; 2017: 10 rows).

## Models
* `image_only`: MLP on frozen ViT CLS + mean-pooled patch tokens.
* `text_only`: fine-tuned `cardiffnlp/twitter-roberta-base-sentiment-latest`, CLS token.
* `concat`: text CLS ⊕ projected ViT CLS.
* `cross_attn`: every text token attends over the 197 ViT tokens; gated residual fusion; text CLS ⊕ pooled fused tokens.

The text encoder and learning rate (3e-5) were chosen on **validation** macro-F1 (text-only, masked, one seed) among
`bert-base-uncased`, `roberta-base`, `vinai/bertweet-base` and `cardiffnlp/twitter-roberta-base-sentiment-latest`
× {2e-5, 3e-5}. Note: the chosen encoder was already fine-tuned on tweet sentiment (TweetEval) before our training.

Protocol: AdamW, 8 epochs, batch 32, best epoch by validation macro-F1, test evaluated once at that epoch,
seeds 42/43/44, mean ± std reported.

## Results (test set, %, mean ± std over 3 seeds)

### Aspect sentiment
| Dataset | Setting | Model | Accuracy | Macro-P | Macro-R | Macro-F1 |
|---|---|---|---|---|---|---|
| Twitter-2015 | masked | image-only | 54.39 ± 0.68 | 43.35 ± 1.37 | 41.53 ± 1.26 | 42.01 ± 1.38 |
| | | text-only | 77.60 ± 0.59 | 74.27 ± 1.04 | 71.63 ± 1.33 | 72.76 ± 0.60 |
| | | concat | 77.66 ± 0.82 | 74.44 ± 1.84 | 71.80 ± 0.82 | 72.88 ± 0.17 |
| | | cross-attn | 77.56 ± 0.37 | 74.16 ± 0.40 | 71.30 ± 0.67 | 72.50 ± 0.45 |
| | explicit | text-only | 77.72 ± 0.77 | 74.39 ± 0.45 | 72.42 ± 3.11 | 73.20 ± 1.88 |
| | | concat | 76.44 ± 0.72 | 71.76 ± 0.92 | 71.25 ± 1.56 | 71.49 ± 1.26 |
| | | cross-attn | 77.08 ± 0.24 | 72.18 ± 0.31 | 72.63 ± 1.15 | 72.39 ± 0.68 |
| Twitter-2017 | masked | image-only | 46.89 ± 1.00 | 44.99 ± 1.00 | 41.79 ± 0.74 | 42.43 ± 0.22 |
| | | text-only | 71.47 ± 1.20 | 70.20 ± 1.33 | 68.75 ± 2.56 | 69.37 ± 2.03 |
| | | concat | 71.96 ± 0.89 | 70.94 ± 1.01 | 70.45 ± 0.49 | 70.64 ± 0.69 |
| | | cross-attn | 72.10 ± 0.45 | 71.07 ± 0.46 | 70.57 ± 0.81 | 70.77 ± 0.27 |
| | explicit | text-only | 71.88 ± 0.74 | 71.40 ± 1.16 | 71.29 ± 1.93 | 71.19 ± 1.14 |
| | | concat | 71.99 ± 1.76 | 71.62 ± 1.32 | 70.58 ± 2.25 | 71.01 ± 1.67 |
| | | cross-attn | 71.02 ± 0.89 | 69.57 ± 0.97 | 71.53 ± 1.30 | 70.33 ± 1.04 |

Majority-class test accuracy: 58.53 (2015), 46.43 (2017).

### Implicit aspect identification (aspect type, aspect masked)
| Dataset | Model | Accuracy | Macro-P | Macro-R | Macro-F1 |
|---|---|---|---|---|---|
| Twitter-2015 | image-only | 49.05 ± 1.69 | 39.90 ± 0.98 | 37.95 ± 0.39 | 37.11 ± 0.61 |
| | text-only | 72.77 ± 0.54 | 66.15 ± 0.21 | 65.76 ± 0.44 | 65.79 ± 0.19 |
| | concat | 73.61 ± 0.84 | 66.91 ± 1.33 | 66.76 ± 0.97 | 66.74 ± 1.09 |
| | cross-attn | 72.56 ± 0.47 | 65.57 ± 1.11 | 65.35 ± 1.04 | 65.29 ± 0.93 |
| Twitter-2017 | image-only | 53.53 ± 0.49 | 40.29 ± 0.24 | 36.95 ± 0.68 | 37.03 ± 0.71 |
| | text-only | 81.21 ± 0.84 | 78.36 ± 2.04 | 75.16 ± 1.16 | 76.48 ± 1.17 |
| | concat | 81.21 ± 0.46 | 76.99 ± 0.63 | 75.56 ± 0.20 | 76.17 ± 0.37 |
| | cross-attn | 80.59 ± 0.20 | 76.44 ± 0.17 | 75.16 ± 0.84 | 75.70 ± 0.50 |

Majority-class test accuracy: 39.47 (2015), 47.12 (2017).

### Summary of findings
* Text carries most of the signal for both tasks; image-only models are well above chance for aspect type but far
  below text-based models.
* Adding the image gives at most ~1 point (e.g. 2017 masked sentiment macro-F1 69.37 → 70.77; 2015 aspect-type
  macro-F1 65.79 → 66.74), which is within about one standard deviation across seeds — not a significant gain.
* Cross-attention does not outperform simple concatenation.

### Limitations
* Masking the aspect term is a controlled proxy; it is not the same as naturally implicit aspects.
* Aspect identification predicts the entity **type**, not the exact aspect span.
* Three seeds only; no significance testing beyond mean ± std.

Per-run results, per-class scores, training curves and test predictions (keyed by `sample_id`) are written by the
training script to `results_runs.csv`, `results_per_class.csv`, `history.csv` and `predictions/`.
