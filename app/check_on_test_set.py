"""Run the saved project models on real Twitter-2015/2017 test samples through the same code path as the test page
(test_model.py) and compare with the gold labels.

    python check_on_test_set.py --data "<folder containing twitter2015/ twitter2015_images/ ...>" --n 200
"""
import argparse
import random
import sys
import time
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import test_model  # noqa: E402

SENT = {"0": "negative", "1": "neutral", "2": "positive"}


def load_test(data, ds):
    rows = []
    lines = (Path(data) / ds / "test.tsv").read_text(encoding="utf-8-sig").split("\n")[1:]
    for line in lines:
        line = line.rstrip("\r")
        if line.strip():
            idx, lab, img, text, asp = line.split("\t")
            rows.append({"id": idx, "label": SENT[lab.strip()], "image": img.strip(), "text": text, "aspect": asp.strip()})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--models-dir", default=str(Path(__file__).parent / "models"))
    ap.add_argument("--n", type=int, default=200, help="random test samples per dataset (0 = all)")
    ap.add_argument("--show", type=int, default=5)
    args = ap.parse_args()
    _, predict = test_model.build(args.models_dir)
    for ds in ("twitter2015", "twitter2017"):
        rows = load_test(args.data, ds)
        sample = rows if args.n == 0 else random.Random(0).sample(rows, min(args.n, len(rows)))
        correct, t0 = 0, time.time()
        for k, r in enumerate(sample):
            tweet = r["text"].replace("$T$", r["aspect"])
            img = Image.open(Path(args.data) / f"{ds}_images" / r["image"])
            s, a, _ = predict(ds, tweet, r["aspect"], img)
            pred = max(s, key=s.get)
            correct += pred == r["label"]
            if k < args.show:
                print(f"[{ds} #{r['id']}] {tweet}\n    aspect: {r['aspect']!r} | gold: {r['label']} | predicted: {pred} "
                      f"({s[pred]:.0%}) | aspect type: {max(a, key=a.get)}")
        print(f"==> {ds}: sentiment accuracy on {len(sample)} real test samples = {correct / len(sample):.2%} "
              f"({time.time() - t0:.0f}s)\n")


if __name__ == "__main__":
    main()
