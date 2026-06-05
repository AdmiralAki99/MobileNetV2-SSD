import argparse
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Analyse anchor boxes for a dataset")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset to analyse (e.g., 'voc' or 'vis_drone')")
    parser.add_argument("--k", type=int, default=9, help="Number of anchor boxes to generate")
    parser.add_argument("--input_size", type=int, default=300, help="Input size for the model")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--classes_file", type=str, default=None, help="Path to the classes file.")

    args = parser.parse_args()

    return {
        "dataset": args.dataset,
        "k": args.k,
        "input_size": args.input_size,
        "seed": args.seed,
        "classes_file": args.classes_file,
    }


def collect_boxes(dataset_name: str, input_size: int, classes_file: str | None = None):
    match dataset_name:
        case "voc":
            from datasets.voc import VOCDataset

            dataset = VOCDataset(root="datasets/VOCdevkit", split="train", classes_file=classes_file)
        case "vis_drone":
            from datasets.vis_drone import VisDroneDataset

            dataset = VisDroneDataset(root="datasets/VisDrone", split="train", classes_file=classes_file)
        case _:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

    # Getting the boxes and the image sizes
    norm_sizes = []

    for boxes, size in dataset.iter_annotations():
        for box in boxes:
            x1, y1, x2, y2 = box
            w = x2 - x1
            h = y2 - y1

            w_norm = w / size[1]
            h_norm = h / size[0]

            # Normalizing the box dimensions by the input size
            w = w_norm * input_size
            h = h_norm * input_size

            norm_sizes.append([h, w])

    return np.array(norm_sizes, dtype=np.float32)


def iou_distance(box, anchors):
    intersection = np.minimum(box[0], anchors[:, 0]) * np.minimum(box[1], anchors[:, 1])
    union = (box[0] * box[1]) + (anchors[:, 0] * anchors[:, 1]) - intersection
    iou = intersection / (union + 1e-10)
    return 1 - iou


def kmeans_iou(boxes, k, seed=42, max_iters=300):
    np.random.seed(seed=seed)
    indices = np.random.choice(len(boxes), size=k, replace=False)
    centroids = boxes[indices].copy()

    prev_assignments = np.full(len(boxes), -1)

    # Randomly initializing the anchors from the boxes
    for _ in range(max_iters):
        distances = np.array([iou_distance(box, centroids) for box in boxes])
        assignments = np.argmin(distances, axis=1)

        # Checking if convergence is reached
        if np.array_equal(centroids, prev_assignments):
            break

        prev_assignments = assignments.copy()

        for i in range(k):
            cluster_boxes = boxes[assignments == i]
            if len(cluster_boxes) > 0:
                centroids[i] = np.median(cluster_boxes, axis=0)

    area = centroids[:, 0] * centroids[:, 1]
    centroids = centroids[np.argsort(area)]

    return centroids


def execute_analysis():

    args = parse_args()

    print(f"Collecting boxes for dataset: {args['dataset']}")
    boxes = collect_boxes(args["dataset"], args["input_size"], args["classes_file"])

    print(f"Running K-Means with k={args['k']} and seed={args['seed']}")
    anchors = kmeans_iou(boxes, args["k"], seed=args["seed"])

    print("Anchor boxes (h, w):")
    for anchor in anchors:
        print(f"{anchor[0]:.2f}, {anchor[1]:.2f}")


if __name__ == "__main__":
    execute_analysis()
