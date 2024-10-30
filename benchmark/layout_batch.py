import argparse
import collections
import copy
import json

from surya.benchmark.metrics import precision_recall
from surya.detection import batch_text_detection
from surya.model.detection.model import load_model, load_processor
from surya.input.processing import open_pdf, get_page_images, convert_if_not_rgb
from surya.layout import batch_layout_detection
from surya.postprocessing.heatmap import draw_polys_on_image, draw_bboxes_on_image
from surya.postprocessing.util import rescale_bbox
from surya.settings import settings
import os
import time
from tabulate import tabulate
import datasets
from PIL import Image
import io
import sys
from surya.benchmark.paddle import paddle_layout_all


def main():
    parser = argparse.ArgumentParser(description="Benchmark layout models.")
    parser.add_argument("--results_dir", type=str, help="Path to JSON file with OCR results.", default=os.path.join(settings.RESULT_DIR, "benchmark"))
    parser.add_argument("--max", type=int, help="Maximum number of images to run benchmark on.", default=None)
    parser.add_argument("--batch_size", type=int, help="Number of images to load per batch", default=32)
    parser.add_argument("--paddle", action="store_true", help="Run paddle paddle.", default=False)
    args = parser.parse_args()

    model = load_model(checkpoint=settings.LAYOUT_MODEL_CHECKPOINT)
    processor = load_processor(checkpoint=settings.LAYOUT_MODEL_CHECKPOINT)
    det_model = load_model()
    det_processor = load_processor()

    pathname = "layout_bench"
    # These have already been shuffled randomly, so sampling from the start is fine
    ds = datasets.load_dataset(settings.LAYOUT_BENCH_DATASET_NAME, split="train")
    if args.max:
        ds = ds.select(range(args.max))

    ds_batch = ds.batch(args.batch_size)

    label_alignment = { # First is publaynet, second is surya
        "Image": [["Figure"], ["Picture", "Figure"], ['figure']],
        "Table": [["Table"], ["Table"], ['table']],
        "Text": [["Text", "List"], ["Text", "Formula", "Footnote", "Caption", "List-item", "Page-header", "Page-footer"], 
                 ['text', 'figure_caption', 'table_caption', 'header', 'footer', 'reference', 'equation']],
        "Title": [["Title"], ["Section-header", "Title"], ['title']]
    }

    surya_time = 0
    paddle_time = 0
    page_metrics = collections.OrderedDict()
    img_len = 0
    for batch_idx, batch in enumerate(ds_batch):
        images = [Image.open(io.BytesIO(img['bytes'])) for img in batch['image']]
        images = convert_if_not_rgb(images)
        img_len += len(images)
        bboxes = [bbox for bbox in batch['bboxes']]
        labels = [label for label in batch['labels']]
        labeled_boxes_batched = [list(zip(zipped[0], zipped[1])) for zipped in zip(labels, bboxes)]

        start = time.time()
        line_predictions = batch_text_detection(images, det_model, det_processor)
        layout_predictions = batch_layout_detection(images, model, processor, line_predictions)
        surya_time += (time.time() - start)

        paddle_layout_predictions = [None] * len(layout_predictions)
        if args.paddle:
            start = time.time()
            paddle_layout_predictions = paddle_layout_all(images)
            paddle_time += (time.time() - start)

        for sub_idx, (labeled_boxes, pred, paddle_pred) in enumerate(zip(labeled_boxes_batched, layout_predictions, paddle_layout_predictions)):
            idx = batch_idx * args.batch_size + sub_idx
            page_results = {'surya': {}}
            if args.paddle:
                page_results['paddle'] = {}
            
            for label_name in label_alignment:
                correct_cats, surya_cats, paddle_cats = label_alignment[label_name]
                correct_bboxes = [b for (l, b) in labeled_boxes if l in correct_cats]

                pred_bboxes = [b.bbox for b in pred.bboxes if b.label in surya_cats]
                
                metrics = precision_recall(pred_bboxes, correct_bboxes, penalize_double=False)
                weight = len(correct_bboxes)
                metrics["weight"] = weight
                page_results['surya'][label_name] = metrics

                if paddle_pred:
                    pred_bboxes = [b['bbox'] for b in paddle_pred if b['label'] in paddle_cats]

                    metrics = precision_recall(pred_bboxes, correct_bboxes, penalize_double=False)
                    weight = len(correct_bboxes)
                    metrics["weight"] = weight
                    page_results['paddle'][label_name] = metrics
            
            page_metrics[idx] = page_results
    
    
    folder_name = os.path.basename(pathname).split(".")[0]
    result_path = os.path.join(args.results_dir, folder_name)
    os.makedirs(result_path, exist_ok=True)


    providers = ['surya']
    provider_mean_metrics = {'surya': collections.defaultdict(dict)}
    if args.paddle:
        providers.append('paddle')
        provider_mean_metrics['paddle'] = collections.defaultdict(dict)

    layout_types = sorted(page_metrics[0]['surya'].keys())
    metric_types = sorted(page_metrics[0]['surya'][layout_types[0]].keys())
    metric_types.remove("weight")

    for l in layout_types:
        for m in metric_types:
            for p in providers:
                metric = []
                total = 0
                for outer_page in page_metrics.values():
                    page = outer_page[p]
                    metric.append(page[l][m] * page[l]["weight"])
                    total += page[l]["weight"]

                value = sum(metric)
                if value > 0:
                    value /= total
                provider_mean_metrics[p][l][m] = value

    out_data = {
        "surya_time": surya_time,
        "metrics": provider_mean_metrics,
        "page_metrics": page_metrics
    }

    with open(os.path.join(result_path, "results.json"), "w+") as f:
        json.dump(out_data, f, indent=4)

    table_headers = ["Provider", "Layout Type", ] + metric_types
    table_data = []
    for provider in providers:
        for layout_type in layout_types:
            table_data.append([provider, layout_type, ] + [f"{provider_mean_metrics[provider][layout_type][m]:.2f}" for m in metric_types])

    print(tabulate(table_data, headers=table_headers, tablefmt="github"))
    print(f"Surya took {surya_time / img_len:.2f} seconds per image, and {surya_time:.1f} seconds total.")
    if args.paddle:
        print(f"Paddle took {paddle_time / img_len:.2f} seconds per image, and {paddle_time:.1f} seconds total.")
    print("Precision and recall are over the mutual coverage of the detected boxes and the ground truth boxes at a .5 threshold.")
    print(f"Wrote results to {result_path}")


if __name__ == "__main__":
    main()
