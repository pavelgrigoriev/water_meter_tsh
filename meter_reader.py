import argparse
import os
import json

import cv2
import numpy as np
from ultralytics import YOLO

def load_models(seg_model_path: str, det_model_path: str):
    seg_model = YOLO(seg_model_path)
    det_model = YOLO(det_model_path)
    return seg_model, det_model


def process_image(img_path: str,
                  seg_model, det_model,
                  save_masks: bool,
                  out_dir: str,
                  seg_conf: float,
                  det_conf: float) -> dict:
    img = cv2.imread(img_path)
    if img is None:
        print("Error: unable to load image from path {img}")
        return {}

    seg_res = seg_model(img_path, agnostic_nms=True, device="cpu",
                         task="segment", conf=seg_conf)[0]
    masks = seg_res.masks.data.cpu().numpy()
    classes = seg_res.boxes.cls.cpu().numpy().astype(int)
    boxes = seg_res.boxes.xyxy.cpu().numpy().astype(int)
    orig_h, orig_w = seg_res.masks.orig_shape

    results = {"cold": None, "hot": None}

    for i, (mask_flat, cls_id, (x1, y1, x2, y2)) in enumerate(
            zip(masks, classes, boxes)):
        mask_full = cv2.resize(mask_flat, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        bin_mask = (mask_full > 0.1).astype(np.uint8) * 255
        crop_img = img[y1:y2, x1:x2]
        crop_mask = bin_mask[y1:y2, x1:x2]
        masked_crop = cv2.bitwise_and(crop_img, crop_img, mask=crop_mask)

        label = 'cold' if cls_id == 0 else 'hot'
        if save_masks:
            os.makedirs(out_dir, exist_ok=True)
            mask_path = os.path.join(out_dir, f"mask_{i:02d}_{label}.png")
            cv2.imwrite(mask_path, masked_crop)

        # Detect digits on CPU
        det_res = det_model(masked_crop, agnostic_nms=True, device="cpu", conf=det_conf)[0]
        det_boxes = det_res.boxes.xyxy.cpu().numpy()
        det_classes = det_res.boxes.cls.cpu().numpy().astype(int)

        if results[label] is not None or len(det_classes) == 0:
            continue

        x_centers = (det_boxes[:, 0] + det_boxes[:, 2]) / 2
        order = np.argsort(x_centers)
        digits_str = ''.join(str(d) for d in det_classes[order])
        results[label] = digits_str

        if len(digits_str) >= 4:
            int_part = digits_str[:-3]
            frac_part = digits_str[-3:]
            formatted_number = f"{int_part}.{frac_part}"
        else:
            formatted_number = f"0.{digits_str.zfill(3)}"

        results[label] = formatted_number

    # Если ничего не обнаружили
    if all(v is None for v in results.values()):
        return {}

    return {
        "cold_water_meter": results["cold"] or "",
        "hot_water_meter": results["hot"] or ""
    }


def main():
    parser = argparse.ArgumentParser(
        description="Read water meter values from images using YOLO segmentation and detection on CPU only."
    )
    parser.add_argument("--img-path", required=True,
                        help="Path to input image")
    parser.add_argument("--seg-model", default="roi_model.pt",
                        help="Path to ROI segmentation model")
    parser.add_argument("--det-model", default="detect.pt",
                        help="Path to digit detection model")
    parser.add_argument("--out-dir", default="masked_preprocessed",
                        help="Directory to save masks and output JSON")
    parser.add_argument("--seg-conf", type=float, default=0.1,
                        help="Confidence threshold for segmentation model")
    parser.add_argument("--det-conf", type=float, default=0.1,
                        help="Confidence threshold for detection model")
    parser.add_argument("--save-masks", action="store_true",
                        help="Save masked ROI crops for debugging")
    args = parser.parse_args()

    seg_model, det_model = load_models(args.seg_model, args.det_model)


    res = process_image(args.img_path, seg_model, det_model,
                        save_masks=args.save_masks,
                        out_dir=args.out_dir,
                        seg_conf=args.seg_conf,
                        det_conf=args.det_conf)


    # Если вообще ничего не обработано, вернём пустой JSON
    output_path = os.path.join(args.out_dir, "results.json")
    os.makedirs(args.out_dir, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=2)

    if res:
        print(f"Results written to {output_path}")
    else:
        print("No results found; empty JSON generated.")

if __name__ == "__main__":
    main()
