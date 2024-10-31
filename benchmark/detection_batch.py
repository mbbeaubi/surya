import argparse
import collections
import copy
import json

from surya.benchmark.paddle import paddle_detect
from surya.benchmark.bbox import get_pdf_lines
from surya.benchmark.metrics import precision_recall
from surya.benchmark.tesseract import tesseract_parallel
from surya.model.detection.model import load_model, load_processor
from surya.input.processing import open_pdf, get_page_images, convert_if_not_rgb
from surya.detection import batch_text_detection
from surya.postprocessing.heatmap import draw_polys_on_image
from surya.postprocessing.util import rescale_bbox
from surya.settings import settings
import os
import time
from tabulate import tabulate
import datasets
from tqdm import tqdm
from PIL import Image
import io


def main():
    parser = argparse.ArgumentParser(description="Detect bboxes in a PDF.")
    parser.add_argument("--results_dir", type=str, help="Path to JSON file with OCR results.", default=os.path.join(settings.RESULT_DIR, "benchmark"))
    parser.add_argument("--max", type=int, help="Maximum number of pdf pages to OCR.", default=None)
    parser.add_argument("--tesseract", action="store_true", help="Run tesseract as well.", default=False)
    parser.add_argument("--paddle", action="store_true", help="Run paddle paddle.", default=False)
    parser.add_argument("--surya", action="store_true", help="Run paddle paddle.", default=False)
    parser.add_argument("--batch_size", type=int, help="Batch size to use", default=32)
    args = parser.parse_args()

    pathname = "det_bench"
    # These have already been shuffled randomly, so sampling from the start is fine
    ds = datasets.load_dataset(settings.DETECTOR_BENCH_DATASET_NAME, split="train", keep_in_memory=False)
    if args.max:
        ds = ds.select(range(args.max))

    ds_batch = ds.batch(args.batch_size)

    if args.surya:
        model = load_model()
        processor = load_processor()

    paddle_time = 0
    surya_time = 0
    tess_time = 0
    img_len = 0
    page_metrics = collections.OrderedDict()
    for batch_idx, batch in tqdm(enumerate(ds_batch), desc='detection batch', total=len(ds_batch)):
        images = [Image.open(io.BytesIO(img['bytes'])) for img in batch['image']]
        images = convert_if_not_rgb(images)
        bboxes = [bbox for bbox in batch['bboxes']]
        img_len += len(images)
        
        correct_boxes = []
        for i, boxes in enumerate(bboxes):
            img_size = images[i].size
            # 1000,1000 is bbox size for doclaynet
            correct_boxes.append([rescale_bbox(b, (1000, 1000), img_size) for b in boxes])
        
        if args.paddle:
            print("Running paddle detection")
            start = time.time()
            paddle_predictions = paddle_detect(images)
            paddle_time += (time.time() - start)
            print("Finished paddle detection")
        else:
            paddle_predictions = [None] * len(images)
            paddle_time = None

        if args.surya:
            start = time.time()
            predictions = batch_text_detection(images, model, processor)
            surya_time += (time.time() - start)
        else:
            predictions = [None] * len(images)

        if args.tesseract:
            start = time.time()
            tess_predictions = tesseract_parallel(images)
            tess_time += (time.time() - start)
        else:
            tess_predictions = [None] * len(images)
            tess_time = None


        for inner_idx, (tb, sb, cb, pb) in enumerate(zip(tess_predictions, predictions, correct_boxes, paddle_predictions)):
            idx = args.batch_size * batch_idx + inner_idx
            
            if args.surya:
                surya_boxes = [s.bbox for s in sb.bboxes]
                surya_metrics = precision_recall(surya_boxes, cb)
            else:
                surya_metrics = None

            if tb is not None:
                tess_metrics = precision_recall(tb, cb)
            else:
                tess_metrics = None

            if pb is not None:
                paddle_metrics = precision_recall(pb, cb)
            else:
                paddle_metrics = None

            page_metrics[idx] = {
                "surya": surya_metrics,
                "tesseract": tess_metrics,
                "paddle": paddle_metrics
            }


    folder_name = os.path.basename(pathname).split(".")[0]
    result_path = os.path.join(args.results_dir, folder_name)
    os.makedirs(result_path, exist_ok=True)

    mean_metrics = {}
    models = []
    if args.surya:
        models.append("surya")
    if args.tesseract:
        models.append("tesseract")
    if args.paddle:
        models.append("paddle")

    metric_types = sorted(page_metrics[0][models[0]].keys())

    for k in models:
        for m in metric_types:
            metric = []
            for page in page_metrics:
                metric.append(page_metrics[page][k][m])
            if k not in mean_metrics:
                mean_metrics[k] = {}
            mean_metrics[k][m] = sum(metric) / len(metric)

    out_data = {
        "times": {
            "surya": surya_time,
            "tesseract": tess_time,
            "paddle": paddle_time
        },
        "metrics": mean_metrics,
        "page_metrics": page_metrics
    }

    with open(os.path.join(result_path, "results.json"), "w+") as f:
        json.dump(out_data, f, indent=4)

    table_headers = ["Model", "Time (s)", "Time per page (s)"] + metric_types
    table_data = []
    if args.surya:
        table_data.append(["surya", surya_time, surya_time / img_len] + [mean_metrics["surya"][m] for m in metric_types])
    if args.tesseract:
        table_data.append(
            ["tesseract", tess_time, tess_time / img_len] + [mean_metrics["tesseract"][m] for m in metric_types]
        )

    if args.paddle:
        table_data.append(
            ["paddle", paddle_time, paddle_time / img_len] + [mean_metrics["paddle"][m] for m in metric_types]
        )

    print(tabulate(table_data, headers=table_headers, tablefmt="github"))
    print("Precision and recall are over the mutual coverage of the detected boxes and the ground truth boxes at a .5 threshold.  There is a precision penalty for multiple boxes overlapping reference lines.")
    print(f"Wrote results to {result_path}")


if __name__ == "__main__":
    main()
