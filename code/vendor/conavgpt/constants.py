HM3D_TARGET_CATEGORIES = [
    "chair",
    "bed",
    "plant",
    "toilet",
    "tv_monitor",
    "sofa",
]

HM3D_SEMANTIC_CATEGORIES = [
    "chair",
    "sofa",
    "plant",
    "bed",
    "toilet",
    "tv_monitor",
    "bathtub",
    "shower",
    "fireplace",
    "appliances",
    "towel",
    "sink",
    "chest_of_drawers",
    "table",
    "stairs",
]

MP3D_TARGET_CATEGORIES = [
    "chair",
    "table",
    "picture",
    "cabinet",
    "cushion",
    "sofa",
    "bed",
    "chest_of_drawers",
    "plant",
    "sink",
    "toilet",
    "stool",
    "towel",
    "tv_monitor",
    "shower",
    "bathtub",
    "counter",
    "fireplace",
    "gym_equipment",
    "seating",
    "clothes",
]

MP3D_CONTEXT_CATEGORIES = [
    "refrigerator",
    "appliances",
    "stairs",
    "door",
    "book",
    "clock",
    "vase",
    "bottle",
    "cup",
    "window",
    "curtain",
    "mirror",
    "shelving",
    "blinds",
]


def _unique(seq):
    out = []
    for x in seq:
        if x not in out:
            out.append(x)
    return out


DATASET_TARGET_CATEGORIES = {
    "hm3d": HM3D_TARGET_CATEGORIES,
    "mp3d": MP3D_TARGET_CATEGORIES,
}

DATASET_SEMANTIC_CATEGORIES = {
    "hm3d": HM3D_SEMANTIC_CATEGORIES,
    "mp3d": _unique(MP3D_TARGET_CATEGORIES + MP3D_CONTEXT_CATEGORIES),
}


def infer_dataset(task_config: str = "") -> str:
    s = (task_config or "").lower()
    if "mp3d" in s or "matterport" in s:
        return "mp3d"
    return "hm3d"


def get_target_categories(dataset: str):
    return DATASET_TARGET_CATEGORIES.get(dataset, HM3D_TARGET_CATEGORIES)


def get_semantic_categories(dataset: str):
    return DATASET_SEMANTIC_CATEGORIES.get(dataset, HM3D_SEMANTIC_CATEGORIES)


def attach_category_config(args):
    dataset = getattr(args, "dataset", None) or infer_dataset(
        getattr(args, "task_config", "")
    )
    args.dataset = dataset
    args.target_categories = list(get_target_categories(dataset))
    args.semantic_categories = list(get_semantic_categories(dataset))
    args.category_to_channel = {
        category: idx for idx, category in enumerate(args.semantic_categories)
    }
    args.goal_id_to_name = list(args.target_categories)
    args.num_sem_categories = len(args.semantic_categories)
    return args


# RedNetResizeWrapper returns MP3D40 predictions as mpcat40index + 1.
REDNET_MP3D40_TO_CATEGORY = {
    # MP3D ObjectNav target classes.
    4: "chair",
    6: "table",
    7: "picture",
    8: "cabinet",
    9: "cushion",
    11: "sofa",
    12: "bed",
    14: "chest_of_drawers",
    15: "plant",
    16: "sink",
    19: "toilet",
    20: "stool",
    21: "towel",
    23: "tv_monitor",
    24: "shower",
    26: "bathtub",
    27: "counter",
    28: "fireplace",
    34: "gym_equipment",
    35: "seating",
    39: "clothes",
    # Context classes confirmed in the MP3D40 mapping table.
    5: "door",
    10: "window",
    13: "curtain",
    17: "stairs",
    22: "mirror",
    32: "shelving",
    33: "blinds",
    38: "appliances",
}

COCO80_TO_CATEGORY = {
    56: "chair",
    57: "sofa",
    58: "plant",
    59: "bed",
    60: "table",
    61: "toilet",
    62: "tv_monitor",
    69: "appliances",
    71: "sink",
    72: "refrigerator",
    73: "book",
    74: "clock",
    75: "vase",
    39: "bottle",
    41: "cup",
}

# Backward-compatible aliases for older modules. New runtime logic should use
# args.target_categories, args.semantic_categories, and args.category_to_channel.
coco_categories = [0, 3, 2, 4, 5, 1]
coco_categories_hm3d2mp3d = [0, 6, 8, 10, 13, 5]
category_to_id = HM3D_TARGET_CATEGORIES
hm3d_category = HM3D_SEMANTIC_CATEGORIES
category_to_id_mp3d = MP3D_TARGET_CATEGORIES
mp3d_context_category = MP3D_CONTEXT_CATEGORIES
category_to_id_mp3d_with_context = DATASET_SEMANTIC_CATEGORIES["mp3d"]

category_to_id_gibson = [
    "chair",
    "couch",
    "potted plant",
    "bed",
    "toilet",
    "tv",
]

mp3d_category_id = {
    "void": 1,
    "chair": 2,
    "sofa": 3,
    "plant": 4,
    "bed": 5,
    "toilet": 6,
    "tv_monitor": 7,
    "table": 8,
    "refrigerator": 9,
    "sink": 10,
    "stairs": 11,
    "fireplace": 12,
}

mp_categories_mapping = [
    4, 11, 15, 12, 19, 23, 26, 24, 28, 38, 21, 16, 14, 6, 16
]
mp_categories_mapping21 = [
    4, 6, 7, 8, 9, 11, 12, 14, 15, 16, 19, 20, 21, 23, 24, 26, 27, 28, 34, 35, 39
]
mp_categories_mapping21_context = [5, 17, 10, 13, 33, 22, 38, 32]
object_category = category_to_id_mp3d + ["background"]
coco_categories_mapping = {
    coco_id: HM3D_SEMANTIC_CATEGORIES.index(category)
    for coco_id, category in COCO80_TO_CATEGORY.items()
    if category in HM3D_SEMANTIC_CATEGORIES
}

color_palette = [
    1.0, 1.0, 1.0,
    0.6, 0.6, 0.6,
    0.95, 0.95, 0.95,
    0.96, 0.36, 0.26,
    0.12156862745098039, 0.47058823529411764, 0.7058823529411765,
    0.9400000000000001, 0.7818, 0.66,
    0.9400000000000001, 0.8868, 0.66,
    0.8882000000000001, 0.9400000000000001, 0.66,
    0.7832000000000001, 0.9400000000000001, 0.66,
    0.6782000000000001, 0.9400000000000001, 0.66,
    0.66, 0.9400000000000001, 0.7468000000000001,
    0.66, 0.9400000000000001, 0.8518000000000001,
    0.66, 0.9232, 0.9400000000000001,
    0.66, 0.8182, 0.9400000000000001,
    0.66, 0.7132, 0.9400000000000001,
    0.7117999999999999, 0.66, 0.9400000000000001,
    0.8168, 0.66, 0.9400000000000001,
    0.9218, 0.66, 0.9400000000000001,
    0.9400000000000001, 0.66, 0.8531999999999998,
    0.9400000000000001, 0.66, 0.748199999999999]
