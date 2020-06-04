"""
Generate mAP score for given model, use the given test dataset with labels.

We are using the mAP tool developed by
https://github.com/Cartucho/mAP, This command generate the ground
truth and predicted values in a files that are compatible to the
mAP command.

NOTE: This project isn't including that command, use it externally.
"""
import os
import click
import pathlib
import numpy as np
import pandas as pd
from PIL import Image
from mystique.initial_setups import set_graph_and_tensors
from mystique.detect_objects import ObjectDetection
from mystique.utils import xml_to_csv, id_to_label


@click.command()
@click.option(
    "--test-dir",
    help="Test image directory, it should be labelmg generated directory",
    required=True)
@click.option(
    "--ground-truth-dir",
    help="Export the ground trught labels to this dir, use the same img name",
    required=True)
@click.option(
    "--pred-truth-dir",
    help="Export the ground trught labels to this dir, use the same img name",
    required=True)
@click.option(
    "--bbox-min-score",
    help="Minimum bbox score from the model to be considered.",
    default=0.9,
    required=False)
def generate_map(test_dir, ground_truth_dir, pred_truth_dir, bbox_min_score):
    # columns used: filename, xmin, ymin, xmax, ymax
    gt_dir = pathlib.Path(ground_truth_dir)
    pd_dir = pathlib.Path(pred_truth_dir)
    gt_dir.mkdir(parents=True, exist_ok=True)
    pd_dir.mkdir(parents=True, exist_ok=True)

    not os.path.exists(ground_truth_dir) and os.mkdir(ground_truth_dir)
    not os.path.exists(pred_truth_dir) and os.mkdir(pred_truth_dir)

    data_df = xml_to_csv(test_dir)
    object_detection = ObjectDetection(*set_graph_and_tensors())
    images = np.unique(data_df['filename'].tolist())

    for img_name in images:
        image = Image.open(f"{test_dir}/{img_name}")
        image = image.convert("RGB")
        width, height = image.size
        image_np = np.asarray(image)
        result, _index = object_detection.get_objects(image_np)

        classes = result['detection_classes'].tolist()
        scores = result['detection_scores'].tolist()
        boxes = result['detection_boxes'].tolist()

        # import pdb; pdb.set_trace()
        preds = []
        pred_iter = zip(classes, scores, boxes)
        for pred in pred_iter:
            label_id, score, bbox = pred
            if score > bbox_min_score:
                ymin = bbox[0] * height
                xmin = bbox[1] * width
                ymax = bbox[2] * height
                xmax = bbox[3] * width
                preds.append(
                    (id_to_label(label_id), score, xmin, ymin, xmax, ymax)
                )

        columns = ['class', 'score', 'xmin', 'ymin', 'xmax', 'ymax']
        fname = img_name.split('.')[0]
        pd.DataFrame.from_records(preds, columns=columns).to_csv(
            f"{pd_dir}/{fname}.txt", header=False, sep=" ", index=False
        )
        # Save the ground truth labels.
        columns = ['class', 'xmin', 'ymin', 'xmax', 'ymax']
        data_df[data_df.filename == img_name][columns].to_csv(
            f"{gt_dir}/{fname}.txt",
            header=False, sep=" ", index=False
        )


if __name__ == "__main__":
    generate_map()
