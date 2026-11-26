"""Matching caption nouns to detector regions through the COCO object vocabulary.

Detector labels live in the 80-class COCO vocabulary (the vocabulary CHAIR scores against).
A caption word is mapped to a COCO category through a synonym table (singular forms only,
since plural nouns refer to several regions and are excluded from supervision).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

COCO_CATEGORIES: List[str] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]

# Singular synonyms (adapted from the CHAIR synonym list; only forms that are unambiguous).
_SYN_SRC: Dict[str, Sequence[str]] = {
    "person": ["person", "man", "woman", "boy", "girl", "child", "kid", "guy", "lady", "gentleman", "baby", "toddler",
               "skier", "skateboarder", "surfer", "snowboarder", "player", "rider", "cyclist", "pedestrian", "worker",
               "teenager", "batter", "catcher", "pitcher", "biker", "bride", "groom", "chef", "driver", "passenger"],
    "bicycle": ["bicycle", "bike", "cycle"], "car": ["car", "automobile", "sedan", "taxi", "cab", "suv", "van"],
    "motorcycle": ["motorcycle", "motorbike", "scooter", "moped"], "airplane": ["airplane", "plane", "jet", "aircraft", "airliner"],
    "bus": ["bus", "minibus"], "train": ["train", "locomotive", "tram", "subway"], "truck": ["truck", "pickup", "lorry"],
    "boat": ["boat", "ship", "canoe", "kayak", "sailboat", "yacht", "ferry"],
    "traffic light": ["stoplight"], "fire hydrant": ["hydrant"], "stop sign": [], "parking meter": [],
    "bench": ["bench"], "bird": ["bird", "seagull", "duck", "pigeon", "parrot", "owl", "goose", "swan"],
    "cat": ["cat", "kitten", "kitty"], "dog": ["dog", "puppy", "pup"], "horse": ["horse", "pony", "foal"],
    "sheep": ["sheep", "lamb"], "cow": ["cow", "cattle", "bull", "calf", "ox"], "elephant": ["elephant"], "bear": ["bear"],
    "zebra": ["zebra"], "giraffe": ["giraffe"], "backpack": ["backpack", "knapsack"], "umbrella": ["umbrella", "parasol"],
    "handbag": ["handbag", "purse", "bag"], "tie": ["tie", "necktie"], "suitcase": ["suitcase", "luggage"],
    "frisbee": ["frisbee"], "skis": ["ski"], "snowboard": ["snowboard"], "sports ball": ["ball", "football", "soccer", "basketball"],
    "kite": ["kite"], "baseball bat": ["bat"], "baseball glove": ["glove", "mitt"], "skateboard": ["skateboard"],
    "surfboard": ["surfboard", "board"], "tennis racket": ["racket", "racquet"], "bottle": ["bottle"],
    "wine glass": ["wineglass"], "cup": ["cup", "mug"], "fork": ["fork"], "knife": ["knife"], "spoon": ["spoon"],
    "bowl": ["bowl"], "banana": ["banana"], "apple": ["apple"], "sandwich": ["sandwich", "burger", "sub"],
    "orange": ["orange"], "broccoli": ["broccoli"], "carrot": ["carrot"], "hot dog": ["hotdog"], "pizza": ["pizza"],
    "donut": ["donut", "doughnut"], "cake": ["cake", "cupcake"], "chair": ["chair", "armchair", "stool"],
    "couch": ["couch", "sofa", "loveseat"], "potted plant": ["plant", "houseplant"], "bed": ["bed"],
    "dining table": ["table", "desk"], "toilet": ["toilet"], "tv": ["tv", "television", "monitor", "screen"],
    "laptop": ["laptop", "notebook", "computer"], "mouse": ["mouse"], "remote": ["remote", "controller"],
    "keyboard": ["keyboard"], "cell phone": ["phone", "cellphone", "smartphone"], "microwave": ["microwave"],
    "oven": ["oven", "stove"], "toaster": ["toaster"], "sink": ["sink"], "refrigerator": ["refrigerator", "fridge"],
    "book": ["book"], "clock": ["clock"], "vase": ["vase"], "scissors": ["scissors"], "teddy bear": ["teddy"],
    "hair drier": ["hairdryer", "blowdryer"], "toothbrush": ["toothbrush"],
}

SYNONYMS: Dict[str, str] = {}
for _cat, _words in _SYN_SRC.items():
    SYNONYMS[_cat] = _cat
    for _w in _words:
        SYNONYMS.setdefault(_w, _cat)

PLURAL_EXCEPTIONS = {"scissors", "skis", "glasses", "pants", "sheep", "bus", "glass", "grass", "class", "dress"}


def normalize_word(word: str) -> str:
    """Lowercase and strip surrounding punctuation."""
    return word.lower().strip(" .,;:!?\"'()[]")


def singularize(word: str) -> str:
    """Very small rule-based singulariser (used only for *mention* counting, never for eligibility)."""
    w = normalize_word(word)
    if w in PLURAL_EXCEPTIONS or w in SYNONYMS or len(w) <= 3:
        return w
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith("ves"):
        return w[:-3] + "f"
    if w.endswith(("ches", "shes", "xes", "sses")):
        return w[:-2]
    if w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def word_to_category(word: str) -> Optional[str]:
    """Map a (singular) word or phrase to a COCO category, or None."""
    w = normalize_word(word)
    if w in SYNONYMS:
        return SYNONYMS[w]
    return SYNONYMS.get(w.replace(" ", "")) if " " in w else None


def match_regions(word: str, detections: Sequence) -> List:
    """Detections whose label equals the COCO category of ``word``."""
    cat = word_to_category(word)
    if cat is None:
        return []
    return [d for d in detections if d.label == cat]
