# evaluate_retriever.py

import os
import sys
import pandas as pd

print("🚀 Starting evaluation script...")

# 1) Ensure Python can import retrieve.py from this directory
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

# 1a) Inject Doc into __main__ so that retrieve.py's pickle.load finds it
import ingest
import __main__
__main__.Doc = ingest.Doc

try:
    from retrieve import retrieve
    print("✅ Successfully imported retrieve()")
except ImportError as e:
    print("❌ Could not import retrieve() -", e)
    sys.exit(1)

# 2) Define test cases (100 random (query, expected_source) pairs)
test_cases = pd.DataFrame([
    {"query": "compute intersection over union for two bounding boxes", 
     "expected_source": "utils/metrics.py"},
    {"query": "crop and resize image to square", 
     "expected_source": "utils/augmentations.py"},
    {"query": "load YOLOv5 configuration from YAML", 
     "expected_source": "models/yolo.py"},
    {"query": "apply mosaic augmentation during training", 
     "expected_source": "utils/datasets.py"},
    {"query": "save model checkpoint every epoch", 
     "expected_source": "utils/general.py"},
    {"query": "filter overlapping detections using NMS", 
     "expected_source": "utils/metrics.py"},
    {"query": "export model to CoreML format", 
     "expected_source": "export.py"},
    {"query": "run YOLOv5 inference on an image", 
     "expected_source": "detect.py"},
    {"query": "compute average precision per class", 
     "expected_source": "utils/metrics.py"},
    {"query": "read image from disk with OpenCV", 
     "expected_source": "utils/general.py"},
    {"query": "compute intersection area between two boxes",
     "expected_source": "utils/metrics.py"},
    {"query": "apply letterbox padding to input image",
     "expected_source": "utils/augmentations.py"},
    {"query": "save one image patch to disk",
     "expected_source": "utils/plots.py"},
    {"query": "scale image tensor by ratio",
     "expected_source": "utils/torch_utils.py"},
    {"query": "apply random perspective transform",
     "expected_source": "utils/augmentations.py"},
    {"query": "clip bounding boxes to image boundaries",
     "expected_source": "utils/general.py"},
    {"query": "merge detection outputs from multiple scales",
     "expected_source": "models/common.py"},
    {"query": "build YOLOv5 backbone network",
     "expected_source": "models/yolo.py"},
    {"query": "create detection head for YOLOv5",
     "expected_source": "models/yolo.py"},
    {"query": "load pretrained YOLOv5 model",
     "expected_source": "hubconf.py"},
    {"query": "parse command-line arguments for training",
     "expected_source": "classify/train.py"},
    {"query": "initialize optimizer with weight decay",
     "expected_source": "train.py"},
    {"query": "log training metrics to TensorBoard",
     "expected_source": "classify/train.py"},
    {"query": "compute average recall for detections",
     "expected_source": "utils/metrics.py"},
    {"query": "read annotations from CSV file",
     "expected_source": "utils/general.py"},
    {"query": "apply softmax to classification outputs",
     "expected_source": "models/common.py"},
    {"query": "perform NMS in cross-entropy loss",
     "expected_source": "utils/metrics.py"},
    {"query": "unfreeze all model layers",
     "expected_source": "utils/general.py"},
    {"query": "freeze backbone layers for fine-tuning",
     "expected_source": "utils/general.py"},
    {"query": "run YOLOv5 train script",
     "expected_source": "train.py"},
    {"query": "evaluate model on validation set",
     "expected_source": "val.py"},
    {"query": "compute Jaccard index for bounding boxes",
     "expected_source": "utils/metrics.py"},
    {"query": "generate anchors for YOLOv5",
     "expected_source": "utils/autoanchor.py"},
    {"query": "context manager for distributed training",
     "expected_source": "utils/torch_utils.py"},
    {"query": "apply random erasing to image",
     "expected_source": "utils/augmentations.py"},
    {"query": "convert segmentation mask to polygon",
     "expected_source": "utils/segment/general.py"},
    {"query": "apply random horizontal flip",
     "expected_source": "utils/augmentations.py"},
    {"query": "update learning rate after each epoch",
     "expected_source": "utils/torch_utils.py"},
    {"query": "apply random vertical flip",
     "expected_source": "utils/augmentations.py"},
    {"query": "compute distance between box centers",
     "expected_source": "utils/metrics.py"},
    {"query": "load class weights from JSON file",
     "expected_source": "utils/general.py"},
    {"query": "apply random brightness adjustment",
     "expected_source": "utils/augmentations.py"},
    {"query": "apply random contrast adjustment",
     "expected_source": "utils/augmentations.py"},
    {"query": "split dataset into train and val sets",
     "expected_source": "utils/datasets.py"},
    {"query": "compute GenAOAnchor adjustment",
     "expected_source": "utils/autoanchor.py"},
    {"query": "apply Sobel filter for edge detection",
     "expected_source": "utils/augmentations.py"},
    {"query": "apply median blur to reduce noise",
     "expected_source": "utils/augmentations.py"},
    {"query": "normalize bounding box coordinates to [0,1]",
     "expected_source": "utils/general.py"},
    {"query": "draw labeled bounding boxes on image",
     "expected_source": "utils/plots.py"},
    {"query": "count total parameters in model",
     "expected_source": "utils/torch_utils.py"},
    {"query": "merge batch of images into grid",
     "expected_source": "utils/plots.py"},
    {"query": "plot bounding boxes on image",
     "expected_source": "utils/plots.py"},
    {"query": "freeze batch normalization layers",
     "expected_source": "models/common.py"},
    {"query": "apply random grayscale augmentation",
     "expected_source": "utils/augmentations.py"},
    {"query": "compute box area and aspect ratio",
     "expected_source": "utils/metrics.py"},
    {"query": "create DataLoader for training dataset",
     "expected_source": "utils/dataloaders.py"},
    {"query": "convert labels to one-hot encoding",
     "expected_source": "utils/general.py"},
    {"query": "load COCO128 dataset for testing",
     "expected_source": "utils/datasets.py"},
    {"query": "apply color jitter augmentation",
     "expected_source": "utils/augmentations.py"},
    {"query": "compute bounding box width and height",
     "expected_source": "utils/metrics.py"},
    {"query": "place model on GPU or CPU automatically",
     "expected_source": "utils/torch_utils.py"},
    {"query": "save detection results to text file",
     "expected_source": "utils/general.py"},
    {"query": "parse JSON for dataset annotations",
     "expected_source": "utils/general.py"},
    {"query": "compute mean pixel value of image",
     "expected_source": "utils/augmentations.py"},
    {"query": "flip image horizontally for augmentation",
     "expected_source": "utils/augmentations.py"},
    {"query": "rotate image by random degrees",
     "expected_source": "utils/augmentations.py"},
    {"query": "apply bilateral filter to image",
     "expected_source": "utils/augmentations.py"},
    {"query": "apply CLAHE to enhance image contrast",
     "expected_source": "utils/augmentations.py"},
    {"query": "compute detection metrics for dataset",
     "expected_source": "utils/metrics.py"},
    {"query": "concatenate image tensors along batch dimension",
     "expected_source": "utils/torch_utils.py"},
    {"query": "initialize batch normalization layers",
     "expected_source": "models/common.py"},
    {"query": "run YOLOv5 on webcam stream",
     "expected_source": "detect.py"},
    {"query": "compute aspect ratio of bounding box",
     "expected_source": "utils/metrics.py"},
    {"query": "compute F1 score based on precision and recall",
     "expected_source": "utils/metrics.py"},
    {"query": "visualize detection results on validation set",
     "expected_source": "utils/plots.py"},
    {"query": "save YOLOv5 training weights periodically",
     "expected_source": "utils/general.py"},
    {"query": "run segmentation inference on tiled images",
     "expected_source": "utils/segment/segment.py"},
    {"query": "adjust pixel padding for letterbox",
     "expected_source": "utils/augmentations.py"},
    {"query": "compute IoU matrix for two sets of boxes",
     "expected_source": "utils/metrics.py"},
    {"query": "parse model depth and width from YAML",
     "expected_source": "models/yolo.py"},
    {"query": "clip segmentation masks to image size",
     "expected_source": "utils/segment/general.py"},
    {"query": "apply grid dropout augmentation",
     "expected_source": "utils/augmentations.py"},
    {"query": "read class names from text file",
     "expected_source": "utils/general.py"},
    {"query": "compute scale and padding for image resize",
     "expected_source": "utils/augmentations.py"},
    {"query": "concatenate anchors for YOLO",
     "expected_source": "utils/autoanchor.py"},
    {"query": "apply random affine transform to image",
     "expected_source": "utils/augmentations.py"},
    {"query": "compute NMS threshold overlap",
     "expected_source": "utils/metrics.py"},
    {"query": "create YOLOv5 training schedule",
     "expected_source": "utils/torch_utils.py"},
    {"query": "merge segment masks from sub-models",
     "expected_source": "utils/segment/general.py"},
    {"query": "compute sigmoid activation for logits",
     "expected_source": "models/common.py"},
    {"query": "apply softmax to segmentation predictions",
     "expected_source": "utils/segment/general.py"},
    {"query": "scale segments from model to original image",
     "expected_source": "utils/segment/general.py"},
    {"query": "compute recall and precision per class",
     "expected_source": "utils/metrics.py"},
    {"query": "apply random hue shift to image",
     "expected_source": "utils/augmentations.py"},
    {"query": "save best model based on validation mAP",
     "expected_source": "train.py"},
    {"query": "load custom YOLOv5 pytorch model",
     "expected_source": "hubconf.py"},
    {"query": "apply random noise to images",
     "expected_source": "utils/augmentations.py"},
    {"query": "create validation split loader",
     "expected_source": "utils/datasets.py"},
    {"query": "apply mosaic collage to four images",
     "expected_source": "utils/datasets.py"},
    {"query": "compute Euclidean distance between box centers",
     "expected_source": "utils/metrics.py"},
    {"query": "apply random cropping to images", 
     "expected_source": "utils/augmentations.py"},
    {"query": "save confusion matrix plot",
     "expected_source": "utils/plots.py"},
    {"query": "load dataset annotations into DataFrame",
     "expected_source": "utils/general.py"},
    {"query": "apply random cutout to image",
     "expected_source": "utils/augmentations.py"},
    {"query": "compute bounding box area and center",
     "expected_source": "utils/metrics.py"},
    {"query": "context manager for multiprocessing locks",
     "expected_source": "utils/torch_utils.py"},
    {"query": "apply random erasing augmentation",
     "expected_source": "utils/augmentations.py"},
    {"query": "convert model to TorchScript",
     "expected_source": "export.py"},
    {"query": "compute class-wise AP for detections",
     "expected_source": "utils/metrics.py"},
    {"query": "apply random blur to image",
     "expected_source": "utils/augmentations.py"},
    {"query": "update optimizer learning rate scheduler",
     "expected_source": "utils/torch_utils.py"},
    {"query": "read video frames and preprocess",
     "expected_source": "utils/general.py"},
    {"query": "save evaluated detection results to CSV",
     "expected_source": "utils/general.py"}
])

if test_cases.empty:
    print("❌ No test cases defined! Add at least one (query, expected_source) pair.")
    sys.exit(1)

# 3) Run each query and collect results
results = []
for _, row in test_cases.iterrows():
    query = row["query"]
    expected = row["expected_source"]
    print(f"\n▶ Running query: \"{query}\"  (expecting → {expected})")

    try:
        hits = retrieve(query)
    except Exception as e:
        print("   ⚠ retrieve() raised an exception:", e)
        results.append({
            "query":           query,
            "expected_source": expected,
            "top_hit_source":  None,
            "correct":         False,
            "error":           str(e)
        })
        continue

    if hits:
        top_hit = hits[0]["source"]
        print("   Top hit:", top_hit)
        matched = (expected in top_hit)
    else:
        print("   No hits returned.")
        top_hit = None
        matched = False

    results.append({
        "query":           query,
        "expected_source": expected,
        "top_hit_source":  top_hit,
        "correct":         matched,
        "error":           None
    })

# 4) Build DataFrame and compute accuracy
results_df = pd.DataFrame(results)
accuracy = results_df["correct"].mean() * 100
num_correct = int(results_df["correct"].sum())
total = len(results_df)

# 5) Print summary and detailed results
print(f"\n📊 Retrieval Accuracy: {accuracy:.1f}% ({num_correct} / {total})\n")
print("Detailed Results:")
print("-" * 60)
for _, row in results_df.iterrows():
    print(f"Query    : {row['query']}")
    print(f"Expected : {row['expected_source']}")
    print(f"Top Hit  : {row['top_hit_source']}")
    print(f"Correct  : {row['correct']}")
    if row["error"]:
        print(f"Error    : {row['error']}")
    print("-" * 60)