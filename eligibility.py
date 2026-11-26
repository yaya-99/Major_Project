"""Three-gate, precision-first token eligibility filter (R05).

A caption word is eligible for the consistency loss iff ALL of:

* **P (POS gate)** - tagged ``NN`` (singular common noun) by the configured tagger, AND concrete: a COCO
  lexicon word, or a WordNet noun whose dominant senses descend from ``physical_entity``. Plural nouns
  (``NNS``) are excluded because they refer to several regions. Words that are part of a multi-word COCO term
  ("hot dog", "teddy bear", ...) are excluded (:data:`MULTIWORD_TERM`) so that "dog" in "hot dog" never gets a dog box.
  LIMITATION: WordNet cannot separate every abstract from every concrete noun; and because gate C needs a COCO
  category, only COCO-vocabulary nouns can ever pass. This is not a general semantic concrete-noun classifier.
* **C (confidence gate)** - at least one detector region of the word's COCO category scores >= ``threshold``
  (theta_c), after geometry validation (finite, positive-area, inside the image) and per-label NMS.
* **U (uniqueness gate)** - the best region must be clearly distinguishable from competitors of the same category.
  Let ``strong`` = same-category regions with score >= theta_c and ``plausible`` = same-category regions with score
  >= ``ambiguity_floor`` (default theta_c). ``strict``: exactly one strong region and no other plausible one.
  ``margin``: (top-1 score - top-2 plausible score) >= ``uniqueness_margin`` and strictly > 0 (ties are ambiguous).

Failed tokens are excluded from the consistency loss only; they still contribute to the caption loss. Every
token gets exactly one machine-readable outcome (``ACCEPTED`` or an exclusion reason) so rates can be
persisted. Ambiguity among *different* categories (e.g. chair vs couch) is not modelled.
"""
from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.grounding.detector import Detection, box_problem, nms
from src.grounding.region_matching import SYNONYMS, normalize_word, word_to_category

logger = logging.getLogger(__name__)

ACCEPTED = "ACCEPTED"
NOT_CONCRETE_NOUN = "NOT_CONCRETE_NOUN"
MULTIWORD_TERM = "MULTIWORD_TERM"
NO_DETECTION = "NO_DETECTION"
LOW_CONFIDENCE = "LOW_CONFIDENCE"
AMBIGUOUS_REGION = "AMBIGUOUS_REGION"
INVALID_ALIGNMENT = "INVALID_ALIGNMENT"
INVALID_REGION_GEOMETRY = "INVALID_REGION_GEOMETRY"
NO_VALID_NEGATIVES = "NO_VALID_NEGATIVES"
NO_GROUNDING = "NO_GROUNDING"
ALL_OUTCOMES = (ACCEPTED, NOT_CONCRETE_NOUN, MULTIWORD_TERM, NO_DETECTION, LOW_CONFIDENCE, AMBIGUOUS_REGION,
                INVALID_ALIGNMENT, INVALID_REGION_GEOMETRY, NO_VALID_NEGATIVES, NO_GROUNDING)

_NLTK_OK: Optional[bool] = None


def _ensure_nltk() -> bool:
    """Make NLTK's tagger and WordNet available; remember the outcome (False -> conservative COCO-lexicon fallback)."""
    global _NLTK_OK
    if _NLTK_OK is not None:
        return _NLTK_OK
    try:
        import nltk
        from nltk.corpus import wordnet as wn

        for res, path in (("averaged_perceptron_tagger_eng", "taggers/averaged_perceptron_tagger_eng"),
                          ("wordnet", "corpora/wordnet")):
            try:
                nltk.data.find(path)
            except LookupError:
                nltk.download(res, quiet=True)
        wn.synsets("dog")
        nltk.pos_tag(["a", "dog"])
        _NLTK_OK = True
    except (LookupError, OSError, ImportError) as exc:  # missing resources / no network
        logger.warning("NLTK resources unavailable (%s); using the built-in COCO lexicon fallback.", exc)
        _NLTK_OK = False
    return _NLTK_OK


@dataclass
class TokenRecord:
    """Eligibility decision for one caption word."""
    token: str
    token_pos: int                      # word index within the caption
    normalized_word: str
    pos_tag: str
    detector_candidates: List[Detection] = field(default_factory=list)
    selected_region: Optional[Detection] = None
    detector_confidence: float = 0.0
    uniqueness_score: float = 0.0       # score gap to the runner-up (1.0 if unique in strict mode)
    is_eligible: bool = False
    exclusion_reason: Optional[str] = None
    token_indices: List[int] = field(default_factory=list)  # sub-word positions of the word
    rule: str = ""                      # which rule produced the outcome (audit trail)

    @property
    def outcome(self) -> str:
        return ACCEPTED if self.is_eligible else (self.exclusion_reason or "UNKNOWN")


class PosTagger:
    """POS tagger: NLTK averaged perceptron when available, otherwise a conservative lexicon fallback.

    The fallback tags words found in the COCO synonym table as ``NN`` and everything else as ``XX``; it can only reduce
    eligibility, never invent it.
    """

    def __init__(self, use_nltk: bool = True):
        self.use_nltk = use_nltk and _ensure_nltk()

    def tag(self, words: Sequence[str]) -> List[Tuple[str, str]]:
        if self.use_nltk:
            import nltk

            return [(w, t) for w, t in nltk.pos_tag(list(words))]
        return [(w, "NN" if normalize_word(w) in SYNONYMS else "XX") for w in words]

    def info(self) -> Dict[str, str]:
        """Tagger identity for reproducibility records."""
        if self.use_nltk:
            import nltk

            return {"tagger": "nltk.pos_tag (averaged_perceptron_tagger_eng)", "nltk_version": nltk.__version__, "concrete_test": "WordNet physical_entity or COCO lexicon"}
        return {"tagger": "coco_lexicon_fallback", "concrete_test": "COCO lexicon only"}


def is_concrete_noun(word: str) -> bool:
    """True if ``word`` names a physical entity (WordNet dominant senses) or is a known COCO object word."""
    w = normalize_word(word)
    if w in SYNONYMS:
        return True
    if not _ensure_nltk():
        return False
    from nltk.corpus import wordnet as wn

    physical = wn.synset("physical_entity.n.01")
    for syn in wn.synsets(w, pos=wn.NOUN)[:3]:  # dominant senses only: avoids rare abstract/physical ambiguity
        if any(physical in path for path in syn.hypernym_paths()):
            return True
    return False


def words_from_token_ids(tokenizer: Any, input_ids: Sequence[int], attention_mask: Optional[Sequence[int]] = None
                         ) -> List[Tuple[str, List[int]]]:
    """Group WordPiece tokens into words.

    Returns:
        List of ``(word, [positions in input_ids])``. Padding (attention mask 0) and special tokens
        (CLS/BOS/SEP/EOS/PAD) are skipped, so they can never become noun candidates. Punctuation is its own word.
    """
    special = set(tokenizer.all_special_ids) | {getattr(tokenizer, "bos_token_id", None)}
    words: List[Tuple[str, List[int]]] = []
    for pos, tid in enumerate(list(input_ids)):
        tid = int(tid)
        if (attention_mask is not None and int(attention_mask[pos]) == 0) or tid in special:
            continue
        piece = tokenizer.convert_ids_to_tokens(tid)
        if piece.startswith("##") and words:
            words[-1] = (words[-1][0] + piece[2:], words[-1][1] + [pos])
        else:
            words.append((piece, [pos]))
    return words


def multiword_members(words: Sequence[str]) -> set:
    """Indices of words that belong to a multi-word COCO term (adjacent pair, or the pair fused: 'hot'+'dog' -> 'hotdog')."""
    norm = [normalize_word(w) for w in words]
    members = set()
    for i in range(len(norm) - 1):
        if f"{norm[i]} {norm[i + 1]}" in SYNONYMS or (norm[i] + norm[i + 1]) in SYNONYMS and norm[i] + norm[i + 1] != norm[i]:
            members.update((i, i + 1))
    return members


def _decide_uniqueness(cands: List[Detection], theta: float, floor: float, mode: str, margin: float) -> Tuple[bool, float, str]:
    strong = [d for d in cands if d.score >= theta]
    plausible = [d for d in cands if d.score >= floor]
    if mode == "strict":
        ok = len(strong) == 1 and len(plausible) == 1
        return ok, 1.0 if ok else 0.0, f"strict: {len(strong)} strong / {len(plausible)} plausible region(s)"
    top = strong[0].score
    second = plausible[1].score if len(plausible) > 1 else 0.0
    gap = top - second
    return (gap >= margin and gap > 0.0), gap, f"margin: gap {gap:.3f} vs required {margin:.3f} (ties rejected)"


def filter_tokens(words: Sequence[str], detections: Sequence[Detection], threshold: float = 0.5,
                  iou_margin: float = 0.5, uniqueness_mode: str = "strict", uniqueness_margin: float = 0.10,
                  tagger: Optional[PosTagger] = None, token_indices: Optional[Sequence[Sequence[int]]] = None,
                  image_size: Optional[Tuple[int, int]] = None, ambiguity_floor: Optional[float] = None) -> List[TokenRecord]:
    """Apply the three gates to every word of a caption (see the module docstring for the exact rules).

    Args:
        words: Caption words in order.
        detections: Detector output for the image (geometry-validated, then NMS with ``iou_margin``).
        threshold: theta_c, detector confidence threshold.
        iou_margin: IoU used by per-label non-maximum suppression.
        uniqueness_mode: ``strict`` or ``margin``.
        uniqueness_margin: Score gap for ``margin`` mode.
        tagger: POS tagger (defaults to :class:`PosTagger`).
        token_indices: Sub-word positions per word, copied into the records; an empty/missing span -> ``INVALID_ALIGNMENT``
            when ``token_indices`` is supplied.
        image_size: (width, height) used to validate box geometry.
        ambiguity_floor: score above which a same-category region is a plausible competitor (default: ``threshold``).
    """
    if uniqueness_mode not in ("strict", "margin"):
        raise ValueError("uniqueness_mode must be 'strict' or 'margin'")
    floor = threshold if ambiguity_floor is None else float(ambiguity_floor)
    if floor > threshold:
        raise ValueError("ambiguity_floor must be <= threshold")
    tagger = tagger or PosTagger()
    words = list(words)
    tagged = tagger.tag(words)
    if token_indices is not None and len(token_indices) != len(words):
        raise ValueError("token_indices must have one entry per word")
    valid, invalid = [], []
    for d in detections:
        (invalid if box_problem(d.bbox, image_size) else valid).append(d)
    dets = nms(valid, iou_margin)
    invalid_labels = {d.label for d in invalid}
    multi = multiword_members(words)
    records: List[TokenRecord] = []
    for i, (word, tag) in enumerate(tagged):
        norm = normalize_word(word)
        span = list(token_indices[i]) if token_indices is not None else []
        rec = TokenRecord(token=word, token_pos=i, normalized_word=norm, pos_tag=tag, token_indices=span)
        records.append(rec)
        # Gate P ------------------------------------------------------------------------------
        if tag != "NN" or not is_concrete_noun(norm):
            rec.exclusion_reason, rec.rule = NOT_CONCRETE_NOUN, f"tag={tag} (needs NN) and concrete"
            continue
        if i in multi:
            rec.exclusion_reason, rec.rule = MULTIWORD_TERM, "word belongs to a multi-word COCO term"
            continue
        if token_indices is not None and (not span or any(p < 0 for p in span)):
            rec.exclusion_reason, rec.rule = INVALID_ALIGNMENT, "no sub-word span for this word"
            continue
        cat = word_to_category(norm)
        cands = sorted((d for d in dets if cat is not None and d.label == cat), key=lambda d: (-d.score, tuple(d.bbox)))
        rec.detector_candidates = cands
        # Gate C ------------------------------------------------------------------------------
        if not cands:
            if cat in invalid_labels:
                rec.exclusion_reason, rec.rule = INVALID_REGION_GEOMETRY, "only geometrically invalid boxes for this category"
            else:
                rec.exclusion_reason, rec.rule = NO_DETECTION, "no detector region of the word's category"
            continue
        strong = [d for d in cands if d.score >= threshold]
        if not strong:
            rec.exclusion_reason, rec.rule = LOW_CONFIDENCE, f"best score {cands[0].score:.3f} < theta_c {threshold}"
            continue
        rec.detector_confidence = strong[0].score
        # Gate U ------------------------------------------------------------------------------
        ok, gap, rule = _decide_uniqueness(cands, threshold, floor, uniqueness_mode, uniqueness_margin)
        rec.uniqueness_score, rec.rule = gap, rule
        if not ok:
            rec.exclusion_reason = AMBIGUOUS_REGION
            continue
        rec.selected_region, rec.is_eligible = strong[0], True
    return records


def summarize_outcomes(records: Sequence[TokenRecord]) -> Dict[str, int]:
    """Counts of outcomes over records (only non-zero entries)."""
    return dict(Counter(r.outcome for r in records))


def apply_eligibility_filter(tokens: Sequence[Tuple[str, str]], detections: Sequence[Detection], threshold: float = 0.5,
                             iou_margin: float = 0.1) -> List[Tuple[str, Detection]]:
    """Convenience wrapper: ``tokens`` are ``(word, pos)`` pairs; returns ``(word, region)`` for eligible ones."""

    class _GivenTags(PosTagger):
        def __init__(self, tags):
            self.tags = tags
            self.use_nltk = False

        def tag(self, words):
            return list(self.tags)

    words = [w for w, _ in tokens]
    records = filter_tokens(words, detections, threshold=threshold, iou_margin=iou_margin, tagger=_GivenTags(tokens))
    return [(r.token, r.selected_region) for r in records if r.is_eligible and r.selected_region is not None]
