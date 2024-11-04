import argparse
from collections import defaultdict

import torch

from benchmark.scoring import overlap_score
from surya.input.processing import convert_if_not_rgb
from surya.model.recognition.model import load_model as load_recognition_model
from surya.model.recognition.processor import load_processor as load_recognition_processor
from surya.ocr import run_recognition
from surya.postprocessing.text import draw_text_on_image
from surya.settings import settings
from surya.languages import CODE_TO_LANGUAGE
from surya.benchmark.tesseract import tesseract_ocr_parallel, surya_lang_to_tesseract, TESS_CODE_TO_LANGUAGE
import os
import datasets
import json
import time
from tabulate import tabulate
from paddleocr.paddleocr import parse_lang
from surya.benchmark.paddle import paddle_ocr_all
from paddleocr import PaddleOCR
from PIL import Image
import io
from tqdm import tqdm

KEY_LANGUAGES = ["Chinese", "Spanish", "English", "Arabic", "Hindi", "Bengali", "Russian", "Japanese"]


def main():
    parser = argparse.ArgumentParser(description="Detect bboxes in a PDF.")
    parser.add_argument("--results_dir", type=str, help="Path to JSON file with OCR results.", default=os.path.join(settings.RESULT_DIR, "benchmark"))
    parser.add_argument("--max", type=int, help="Maximum number of pdf pages to OCR.", default=None)
    parser.add_argument("--batch_size", type=int, help="Batch size to use", default=32)
    parser.add_argument("--debug", type=int, help="Debug level - 1 dumps bad detection info, 2 writes out images.", default=0)
    parser.add_argument("--tesseract", action="store_true", help="Run tesseract instead of surya.", default=False)
    parser.add_argument("--paddle", action="store_true", help="Run paddle.", default=False)
    parser.add_argument("--surya", action="store_true", help="Run paddle.", default=False)
    parser.add_argument("--langs", type=str, help="Specify certain languages to benchmark.", default=None)
    parser.add_argument("--tess_cpus", type=int, help="Number of CPUs to use for tesseract.", default=28)
    parser.add_argument("--compile", action="store_true", help="Compile the model.", default=False)
    parser.add_argument("--specify_language", action="store_true", help="Pass language codes into the model.", default=False)
    args = parser.parse_args()

    if args.compile:
        assert settings.RECOGNITION_STATIC_CACHE, "You must set RECOGNITION_STATIC_CACHE to compile the model."

    rec_model = load_recognition_model()
    rec_processor = load_recognition_processor()

    dataset = datasets.load_dataset(settings.RECOGNITION_BENCH_DATASET_NAME, split="train", keep_in_memory=False)

    if args.langs:
        langs = args.langs.split(",")
        dataset = dataset.filter(lambda x: x["language"] in langs)

    if args.max:
        dataset = dataset.select(range(args.max))

    batched_ds = dataset.batch(args.batch_size)

    surya_scores = defaultdict(list)
    paddle_scores = defaultdict(list)
    tess_scores = defaultdict(list)
    paddle_models: dict[str, PaddleOCR] = {}
    img_len = 0
    paddle_img_len = 0
    tess_img_len = 0
    surya_time = 0
    paddle_time = 0
    tess_time = 0
    for idx, batch in tqdm(enumerate(batched_ds), desc='recog batch', total=len(batched_ds)):
        images = [Image.open(io.BytesIO(img["bytes"])) for img in batch["image"]]
        images = convert_if_not_rgb(images)
        bboxes = [bbox for bbox in batch["bboxes"]]
        line_text = [txt for txt in batch["text"]]
        languages = [ln for ln in batch["language"]]
        st, sl, pt, pl, tt, tl = process_one_batch(images, bboxes, line_text, languages, rec_model, rec_processor, 
                          args, idx == 0,
                          surya_scores, paddle_scores, tess_scores, paddle_models)
        
        surya_time += st
        img_len += sl
        paddle_time += pt
        paddle_img_len += pl
        tess_time += tt
        tess_img_len += tl
        

    result_path = os.path.join(args.results_dir, "rec_bench")
    os.makedirs(result_path, exist_ok=True)

    benchmark_stats = {}
    if args.surya:
        flat_surya_scores = [s for l in surya_scores for s in surya_scores[l]]
        benchmark_stats = {
            "surya": {
                "avg_score": sum(flat_surya_scores) / max(1, len(flat_surya_scores)),
                "lang_scores": {l: sum(scores) / max(1, len(scores)) for l, scores in surya_scores.items()},
                "time_per_img": surya_time / max(1, img_len)
            }
        }

        with open(os.path.join(result_path, "surya_scores.json"), "w+") as f:
            json.dump(surya_scores, f)

    if args.paddle:
        flat_paddle_scores = [s for l in paddle_scores for s in paddle_scores[l]]
        benchmark_stats["paddle"] = {
            "avg_score": sum(flat_paddle_scores) / len(flat_paddle_scores),
            "lang_scores": {l: sum(scores) / len(scores) for l, scores in paddle_scores.items()},
            "time_per_img": paddle_time / paddle_img_len
        }

        with open(os.path.join(result_path, "paddle_scores.json"), "w+") as f:
            json.dump(paddle_scores, f)

    if args.tesseract:
        flat_tess_scores = [s for l in tess_scores for s in tess_scores[l]]
        benchmark_stats["tesseract"] = {
            "avg_score": sum(flat_tess_scores) / len(flat_tess_scores),
            "lang_scores": {l: sum(scores) / len(scores) for l, scores in tess_scores.items()},
            "time_per_img": tess_time / tess_img_len
        }

        with open(os.path.join(result_path, "tesseract_scores.json"), "w+") as f:
            json.dump(tess_scores, f)

    with open(os.path.join(result_path, "results.json"), "w+") as f:
        json.dump(benchmark_stats, f)

    key_languages = [k for k in KEY_LANGUAGES if k in surya_scores]
    table_headers = ["Model", "Time per page (s)", "Avg Score"] + key_languages
    table_data = [
    ]
    if args.surya:
        table_data.append(["surya", benchmark_stats["surya"]["time_per_img"], benchmark_stats["surya"]["avg_score"]] + [benchmark_stats["surya"]["lang_scores"][l] for l in key_languages])
    if args.tesseract:
        table_data.append(
            ["tesseract", benchmark_stats["tesseract"]["time_per_img"], benchmark_stats["tesseract"]["avg_score"]] + [benchmark_stats["tesseract"]["lang_scores"].get(l, 0) for l in key_languages]
        )
    if args.paddle:
        table_data.append(
            ["paddle", benchmark_stats["paddle"]["time_per_img"], benchmark_stats["paddle"]["avg_score"]] + [benchmark_stats["paddle"]["lang_scores"].get(l, 0) for l in key_languages]
        )

    print(tabulate(table_data, headers=table_headers, tablefmt="github"))
    print("Only a few major languages are displayed. See the result path for additional languages.")

    print(f"Wrote results to {result_path}")




def process_one_batch(images, bboxes, line_text, languages, rec_model, rec_processor, args, first, 
                      surya_scores, paddle_scores, tess_scores, paddle_models: dict[str, PaddleOCR]):
    print(f"Loaded {len(images)} images. Running OCR...")

    lang_list = []
    for l in languages:
        if not isinstance(l, list):
            lang_list.append([l])
        else:
            lang_list.append(l)
    n_list = [None] * len(images)

    if args.surya:
        if args.compile and first:
            torch.set_float32_matmul_precision('high')
            torch._dynamo.config.cache_size_limit = 64
            rec_model.decoder.model = torch.compile(rec_model.decoder.model)
            # Run through one batch to compile the model
            run_recognition(images[:1], lang_list[:1], rec_model, rec_processor, bboxes=bboxes[:1])

        start = time.time()
        predictions_by_image = run_recognition(images, lang_list if args.specify_language else n_list, rec_model, rec_processor, bboxes=bboxes)
        surya_time = time.time() - start

        for idx, (pred, ref_text, lang) in tqdm(enumerate(zip(predictions_by_image, line_text, lang_list)), desc='scoring surya', total=len(predictions_by_image)):
            pred_text = [l.text for l in pred.text_lines]
            image_score = overlap_score(pred_text, ref_text)
            for l in lang:
                surya_scores[CODE_TO_LANGUAGE[l]].append(image_score)
        
        surya_img_len = len(images)
    else:
        surya_time = 0
        surya_img_len = 0

    paddle_time = 0
    paddle_img_len = 0
    if args.paddle:
        paddle_valid = []
        paddle_langs = []
        orig_langs = []

        for idx, lang in enumerate(lang_list):
            try:
                l = lang[0]
                if l == 'zh':
                    l = 'ch'
                paddle_lang, _ = parse_lang(l)
                orig_langs.append(lang[0])
                paddle_valid.append(idx)
                paddle_langs.append(paddle_lang)

                if paddle_lang not in paddle_models.keys():
                    paddle_models[paddle_lang] = PaddleOCR(lang = paddle_lang)

            except AssertionError as e:
                print("Lang {}, not supported by paddle".format(lang))

        paddle_imgs = [images[i] for i in paddle_valid]
        paddle_bboxes = [bboxes[i] for i in paddle_valid]
        paddle_reference = [line_text[i] for i in paddle_valid]

        start = time.time()
        paddle_predictions = paddle_ocr_all(paddle_imgs, paddle_bboxes, paddle_langs, paddle_models)
        paddle_time = time.time() - start
        paddle_img_len = len(paddle_imgs)

        for idx, (pred, ref_text, lang) in tqdm(enumerate(zip(paddle_predictions, paddle_reference, orig_langs)), desc='scoring paddle', total=len(paddle_predictions)):
            image_score = overlap_score(pred, ref_text)
            paddle_scores[CODE_TO_LANGUAGE[lang]].append(image_score)

        
    tess_time = 0
    tess_img_len = 0
    if args.tesseract:
        tess_valid = []
        tess_langs = []
        for idx, lang in enumerate(lang_list):
            # Tesseract does not support all languages
            tess_lang = surya_lang_to_tesseract(lang[0])
            if tess_lang is None:
                continue

            tess_valid.append(idx)
            tess_langs.append(tess_lang)

        tess_imgs = [images[i] for i in tess_valid]
        tess_bboxes = [bboxes[i] for i in tess_valid]
        tess_reference = [line_text[i] for i in tess_valid]
        start = time.time()
        tess_predictions = tesseract_ocr_parallel(tess_imgs, tess_bboxes, tess_langs, cpus=args.tess_cpus)
        tess_time = time.time() - start
        tess_img_len = len(tess_imgs)

        for idx, (pred, ref_text, lang) in tqdm(enumerate(zip(tess_predictions, tess_reference, tess_langs)), desc='scoring tesseract', total=len(tess_predictions)):
            image_score = overlap_score(pred, ref_text)
            tess_scores[TESS_CODE_TO_LANGUAGE[lang]].append(image_score)

    return surya_time, surya_img_len, paddle_time, paddle_img_len, tess_time, tess_img_len


    #if args.debug >= 1:
    #    bad_detections = []
    #    for idx, (score, lang) in enumerate(zip(flat_surya_scores, lang_list)):
    #        if score < .8:
    #            bad_detections.append((idx, lang, score))
    #    print(f"Found {len(bad_detections)} bad detections. Writing to file...")
    #    with open(os.path.join(result_path, "bad_detections.json"), "w+") as f:
    #        json.dump(bad_detections, f)

    #if args.debug == 2:
    #    for idx, (image, pred, ref_text, bbox, lang) in enumerate(zip(images, predictions_by_image, line_text, bboxes, lang_list)):
    #        pred_image_name = f"{'_'.join(lang)}_{idx}_pred.png"
    #        ref_image_name = f"{'_'.join(lang)}_{idx}_ref.png"
    #        pred_text = [l.text for l in pred.text_lines]
    #        pred_image = draw_text_on_image(bbox, pred_text, image.size, lang)
    #        pred_image.save(os.path.join(result_path, pred_image_name))
    #        ref_image = draw_text_on_image(bbox, ref_text, image.size, lang)
    #        ref_image.save(os.path.join(result_path, ref_image_name))
    #        image.save(os.path.join(result_path, f"{'_'.join(lang)}_{idx}_image.png"))


if __name__ == "__main__":
    main()
