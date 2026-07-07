import os
import string
from collections import defaultdict

import numpy as np
import torch


_BOOSTER_CACHE = {}


_ALIASES = {
    "tv_monitor": ["tv monitor", "television", "tv", "monitor"],
    "chest_of_drawers": ["chest of drawers", "dresser", "drawer chest"],
    "picture": ["picture", "painting", "framed picture", "wall art"],
    "sofa": ["sofa", "couch"],
    "plant": ["plant", "potted plant"],
    "gym_equipment": ["gym equipment", "exercise equipment", "fitness equipment"],
    "seating": ["seating", "seat", "chair"],
    "clothes": ["clothes", "clothing", "garment"],
    "counter": ["counter", "countertop"],
    "appliances": ["appliances", "appliance", "oven", "refrigerator", "microwave"],
    "cabinet": ["cabinet", "cupboard"],
    "towel": ["towel", "bath towel"],
    "table": ["table", "dining table", "desk"],
    "sink": ["sink", "basin"],
    "bathtub": ["bathtub", "bath tub", "tub"],
    "fireplace": ["fireplace", "hearth"],
}


def normalize_label(label):
    label = str(label).lower()
    label = label.replace("_", " ")
    label = label.split(".")[0]
    label = label.translate(str.maketrans("", "", string.punctuation))
    return " ".join(label.split())


def _category_aliases(category):
    base = normalize_label(category)
    aliases = [base]
    aliases.extend(_ALIASES.get(category, []))
    aliases.extend(_ALIASES.get(base.replace(" ", "_"), []))
    seen = set()
    out = []
    for alias in aliases:
        alias = normalize_label(alias)
        if alias and alias not in seen:
            out.append(alias)
            seen.add(alias)
    return out


def _canonicalize_label(label, alias_to_category):
    label = normalize_label(label)
    if label in alias_to_category:
        return alias_to_category[label]

    for alias, category in alias_to_category.items():
        if alias in label or label in alias:
            return category
    return None


class NoopSemanticBooster:
    enabled = False

    def predict(self, rgb, categories):
        return None


class GroundedSamSemanticBooster:
    enabled = True

    def __init__(self, args, device):
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        from segment_anything import SamPredictor, sam_model_registry

        self.device = torch.device(device)
        self.model_id = args.semantic_boost_model_id
        self.box_threshold = args.semantic_boost_box_threshold
        self.text_threshold = args.semantic_boost_text_threshold
        self.sam_iou_threshold = args.semantic_boost_sam_iou_threshold
        self.min_mask_area = args.semantic_boost_min_mask_area
        self.max_mask_frac = args.semantic_boost_max_mask_frac
        self.max_detections_per_category = args.semantic_boost_max_detections_per_category

        if not os.path.exists(args.semantic_boost_sam_checkpoint):
            raise FileNotFoundError(
                "SAM checkpoint not found: {}".format(args.semantic_boost_sam_checkpoint)
            )

        self.processor = AutoProcessor.from_pretrained(self.model_id)
        if not hasattr(self.processor, "post_process_grounded_object_detection"):
            raise RuntimeError(
                "Transformers processor for {} does not support grounded object "
                "detection post-processing".format(self.model_id)
            )
        self.detector = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_id)
        self.detector.to(self.device)
        self.detector.eval()

        if args.semantic_boost_sam_type not in sam_model_registry:
            raise ValueError("Unknown SAM model type: {}".format(args.semantic_boost_sam_type))
        sam = sam_model_registry[args.semantic_boost_sam_type](
            checkpoint=args.semantic_boost_sam_checkpoint
        )
        sam.to(device=self.device)
        sam.eval()
        self.sam_predictor = SamPredictor(sam)
        print(
            "[semantic_boost] enabled backend=grounded_sam detector={} sam={}".format(
                self.model_id, args.semantic_boost_sam_checkpoint
            )
        )

    def predict(self, rgb, categories):
        if rgb is None or len(categories) == 0:
            return None
        rgb = np.asarray(rgb, dtype=np.uint8)
        h, w = rgb.shape[:2]
        boost = np.zeros((h, w, len(categories)), dtype=np.float32)

        aliases = []
        alias_to_category = {}
        for idx, category in enumerate(categories):
            for alias in _category_aliases(category):
                aliases.append(alias)
                alias_to_category.setdefault(alias, idx)
        if not aliases:
            return boost

        detections = self._detect(rgb, aliases, alias_to_category)
        if not detections:
            return boost

        self.sam_predictor.set_image(rgb)
        image_area = float(h * w)
        for category_idx, boxes in detections.items():
            for box, det_score in boxes:
                masks, scores, _ = self.sam_predictor.predict(
                    box=box.astype(np.float32),
                    multimask_output=True,
                )
                if masks is None or len(masks) == 0:
                    continue
                best_idx = int(np.argmax(scores))
                sam_score = float(scores[best_idx])
                if sam_score < self.sam_iou_threshold:
                    continue
                mask = masks[best_idx].astype(bool)
                area = int(mask.sum())
                if area < self.min_mask_area or area > self.max_mask_frac * image_area:
                    continue
                boost[:, :, category_idx] = np.maximum(
                    boost[:, :, category_idx],
                    mask.astype(np.float32) * float(det_score * sam_score),
                )
        return boost

    def _detect(self, rgb, aliases, alias_to_category):
        from PIL import Image

        image = Image.fromarray(rgb)
        text = ". ".join(aliases) + "."
        inputs = self.processor(images=image, text=text, return_tensors="pt")
        inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.detector(**inputs)

        target_sizes = torch.tensor([rgb.shape[:2]], device=self.device)
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.get("input_ids"),
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=target_sizes,
        )
        if not results:
            return {}

        result = results[0]
        boxes = result.get("boxes", [])
        scores = result.get("scores", [])
        labels = result.get("text_labels", result.get("labels", []))
        by_category = defaultdict(list)

        for box, score, label in zip(boxes, scores, labels):
            category_idx = _canonicalize_label(label, alias_to_category)
            if category_idx is None:
                continue
            score = float(score)
            box = box.detach().float().cpu().numpy()
            box[0::2] = np.clip(box[0::2], 0, rgb.shape[1] - 1)
            box[1::2] = np.clip(box[1::2], 0, rgb.shape[0] - 1)
            if box[2] <= box[0] + 1 or box[3] <= box[1] + 1:
                continue
            by_category[category_idx].append((box, score))

        for category_idx in list(by_category.keys()):
            by_category[category_idx] = sorted(
                by_category[category_idx],
                key=lambda item: item[1],
                reverse=True,
            )[: self.max_detections_per_category]
        return by_category


def get_semantic_booster(args, device):
    backend = getattr(args, "semantic_boost_backend", "none")
    if backend in (None, "", "none", "0", "false", "False"):
        return NoopSemanticBooster()
    if backend != "grounded_sam":
        raise ValueError("Unsupported semantic boost backend: {}".format(backend))

    key = (
        backend,
        str(device),
        args.semantic_boost_model_id,
        args.semantic_boost_sam_type,
        args.semantic_boost_sam_checkpoint,
    )
    if key not in _BOOSTER_CACHE:
        _BOOSTER_CACHE[key] = GroundedSamSemanticBooster(args, device)
    return _BOOSTER_CACHE[key]
