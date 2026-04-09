from flask import Flask, render_template, request
import os
import json
import random

from model.text_model import predict_text, explain_text
from model.image_model import predict_image
from model.fusion import fuse_predictions

app = Flask(__name__)

UPLOAD_FOLDER = "static"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():

    result = None

    if request.method == "POST":
        text = request.form["text"]
        image = request.files.get("image")
        fetched_image = request.form.get("fetched_image")

        if image and image.filename != "":
            image_path = os.path.join(UPLOAD_FOLDER, image.filename)
            image.save(image_path)

        elif fetched_image:
            image_path = fetched_image

        else:
            image_path = None

        text_label, text_score = predict_text(text)
        if image_path:
            image_label, image_score, gradcam_path = predict_image(image_path)
        else:
            image_label, image_score, gradcam_path = "N/A", 0.0, None
        highlighted_text, keywords = explain_text(text)

        final_label, final_score = fuse_predictions(
            text_label, text_score,
            image_label, image_score
        )

        result = {
            "text_label": text_label,
            "image_label": image_label,
            "final_label": final_label,
            "score": round(final_score * 100, 2),
            "keywords": keywords,
            "highlighted_text": highlighted_text,
            "gradcam": gradcam_path,
            "image_path": image_path
        }

    return render_template("index.html", result=result)

@app.route("/fetch")
def fetch_data():

    with open("dataset/news_data.json") as f:
        news = json.load(f)

    sample = random.choice(news)

    return {
        "text": sample["text"],
        "image": sample["image"]
    }

if __name__ == "__main__":
    app.run(debug=True)