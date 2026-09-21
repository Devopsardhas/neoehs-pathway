# Detectable Objects

## Why box & bag were missed

The default **YOLO COCO model** knows 80 everyday classes. It does **not** include:

| Object on pathway | COCO class? | Result |
|-------------------|-------------|--------|
| Cardboard box / cotton box | **No** | Never detected |
| Generic bag on floor | Partial — only `handbag`, `backpack`, `suitcase` | Often missed (small, low confidence) |
| Generic machine / equipment | **No** | Never detected |

That is why chairs and laptops show bounding boxes, but the box and bag on the floor do not.

## Solution: YOLO-World mode (now default)

`config.yaml` is set to `mode: world` with `yolov8s-worldv2.pt`. You define exactly what to look for in `custom_classes`:

```yaml
detection:
  mode: "world"
  model: "yolov8s-worldv2.pt"
  confidence: 0.25
  imgsz: 1280
  custom_classes:
    - cardboard box
    - bag
    - machine
    - ...
```

YOLO-World uses text prompts, so **cardboard box**, **bag**, **carton**, **machine**, **equipment** etc. can all be detected without retraining.

## What gets detected now

### Pathway-blocking objects (configured in custom_classes)

| Category | Examples |
|----------|----------|
| **Packages** | cardboard box, carton, package, crate |
| **Bags** | bag, backpack, handbag, suitcase |
| **Furniture** | chair, stool |
| **Equipment** | machine, equipment, printer, fan, ladder, toolbox |
| **Containers** | bucket, bin, trash bin |
| **Electronics** | laptop, monitor |
| **Other** | cart, trolley, helmet, bottle, fire extinguisher |

Add or remove items in `config.yaml` → `detection.custom_classes` anytime.

### Machines on the pathway

| Machine type | Detected? |
|--------------|-----------|
| Printer, fan, laptop, monitor | Yes (in custom_classes) |
| Generic "machine" / "equipment" | Yes (open-vocabulary prompt) |
| Large industrial machine (no clear visual match) | May need a specific label, e.g. `"industrial machine"` or `"generator"` added to custom_classes |

## COCO mode (alternative)

Set `mode: "coco"` and `model: "yolov8n.pt"` to use the original 80 classes:

**Detected:** chair, couch, bed, dining table, tv, laptop, mouse, keyboard, cell phone, microwave, oven, refrigerator, backpack, handbag, suitcase, book, clock, vase, bottle, etc.

**NOT detected:** cardboard box, generic bag on floor, generic machine, cartons, crates, ladders, buckets, carts.

## Tuning tips for floor objects

If box/bag still missed after switching to world mode:

```yaml
detection:
  confidence: 0.20    # lower = more sensitive
  imgsz: 1280         # or 1920 for very small objects
  model: "yolov8m-worldv2.pt"  # larger = more accurate, slower
```

First run downloads the world model (~50 MB) automatically.
