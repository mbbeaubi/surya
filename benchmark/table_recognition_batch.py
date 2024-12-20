import argparse
import collections
import copy
import json

from tabulate import tabulate

from surya.input.processing import convert_if_not_rgb
from surya.model.table_rec.model import load_model
from surya.model.table_rec.processor import load_processor
from surya.tables import batch_table_recognition, get_batch_size
from surya.settings import settings
from surya.benchmark.metrics import rank_accuracy, penalized_iou_score
from surya.benchmark.tatr import load_tatr, batch_inference_tatr
import os
import time
import datasets
from tqdm import tqdm
from PIL import Image
import io


def main():
    parser = argparse.ArgumentParser(description="Benchmark surya table recognition model.")
    parser.add_argument("--results_dir", type=str, help="Path to JSON file with benchmark results.", default=os.path.join(settings.RESULT_DIR, "benchmark"))
    parser.add_argument("--max", type=int, help="Maximum number of images to run benchmark on.", default=None)
    parser.add_argument("--tatr", action="store_true", help="Run table transformer.", default=False)
    parser.add_argument("--paddle", action="store_true", help="Run paddle paddle.", default=False)
    parser.add_argument("--batch_size", type=int, help="Batch size to use", default=32)

    args = parser.parse_args()

    model = load_model()
    processor = load_processor()

    pathname = "table_rec_bench"
    # These have already been shuffled randomly, so sampling from the start is fine
    split = "train"
    if args.max is not None:
        split = f"train[:{args.max}]"
    ds = datasets.load_dataset(settings.TABLE_REC_BENCH_DATASET_NAME, split=split)

    batch_ds = ds.batch(args.batch_size)
    surya_time = 0
    paddle_time = 0
    page_metrics = collections.OrderedDict()
    for batch_idx, batch in tqdm(enumerate(batch_ds), desc='table recognition'):
        images = list(batch["image"])
        images = convert_if_not_rgb(images)
        bboxes = list(batch["bboxes"])


        start = time.time()
        bboxes = [[{"bbox": b, "text": None} for b in bb] for bb in bboxes]
        table_rec_predictions = batch_table_recognition(images, bboxes, model, processor)
        surya_time = time.time() - start

        mean_col_iou = 0
        mean_row_iou = 0
        for i, pred in enumerate(table_rec_predictions):
            idx = batch_idx * args.batch_size + i
            
            row = dataset[idx]
            
            pred_row_boxes = [p.bbox for p in pred.rows]
            pred_col_bboxes = [p.bbox for p in pred.cols]
            actual_row_bboxes = row["rows"]
            actual_col_bboxes = row["cols"]
            row_score = penalized_iou_score(pred_row_boxes, actual_row_bboxes)
            col_score = penalized_iou_score(pred_col_bboxes, actual_col_bboxes)
            page_results = {
                "row_score": row_score,
                "col_score": col_score,
                "row_count": len(actual_row_bboxes),
                "col_count": len(actual_col_bboxes)
            }

            mean_col_iou += col_score
            mean_row_iou += row_score

            page_metrics[idx] = page_results

        mean_col_iou /= len(table_rec_predictions)
        mean_row_iou /= len(table_rec_predictions)



    folder_name = os.path.basename(pathname).split(".")[0]
    result_path = os.path.join(args.results_dir, folder_name)
    os.makedirs(result_path, exist_ok=True)

    
    

    out_data = {"surya": {
        "time": surya_time,
        "mean_row_iou": mean_row_iou,
        "mean_col_iou": mean_col_iou,
        "page_metrics": page_metrics
    }}

    if args.tatr:
        tatr_model = load_tatr()
        start = time.time()
        tatr_predictions = batch_inference_tatr(tatr_model, images, 1)
        tatr_time = time.time() - start

        page_metrics = collections.OrderedDict()
        mean_col_iou = 0
        mean_row_iou = 0
        for idx, pred in enumerate(tatr_predictions):
            row = dataset[idx]
            pred_row_boxes = [p["bbox"] for p in pred["rows"]]
            pred_col_bboxes = [p["bbox"] for p in pred["cols"]]
            actual_row_bboxes = row["rows"]
            actual_col_bboxes = row["cols"]
            row_score = penalized_iou_score(pred_row_boxes, actual_row_bboxes)
            col_score = penalized_iou_score(pred_col_bboxes, actual_col_bboxes)
            page_results = {
                "row_score": row_score,
                "col_score": col_score,
                "row_count": len(actual_row_bboxes),
                "col_count": len(actual_col_bboxes)
            }

            mean_col_iou += col_score
            mean_row_iou += row_score

            page_metrics[idx] = page_results

        mean_col_iou /= len(tatr_predictions)
        mean_row_iou /= len(tatr_predictions)

        out_data["tatr"] = {
            "time": tatr_time,
            "mean_row_iou": mean_row_iou,
            "mean_col_iou": mean_col_iou,
            "page_metrics": page_metrics
        }


    with open(os.path.join(result_path, "results.json"), "w+") as f:
        json.dump(out_data, f, indent=4)

    table = [
        ["Model", "Row Intersection", "Col Intersection", "Time Per Image"],
        ["Surya", f"{out_data['surya']['mean_row_iou']:.2f}", f"{out_data['surya']['mean_col_iou']:.2f}",
         f"{surya_time / len(images):.2f}"],
    ]

    if args.tatr:
        table.append(["Table transformer", f"{out_data['tatr']['mean_row_iou']:.2f}", f"{out_data['tatr']['mean_col_iou']:.2f}",
         f"{tatr_time / len(images):.2f}"])

    print(tabulate(table, headers="firstrow", tablefmt="github"))

    print("Intersection is the average of the intersection % between each actual row/column, and the predictions.  With penalties for too many/few predictions.")
    print("Note that table transformers is unbatched, since the example code in the repo is unbatched.")
    print(f"Wrote results to {result_path}")


if __name__ == "__main__":
    main()