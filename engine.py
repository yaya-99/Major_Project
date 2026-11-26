"""Analysis engine of the demo: one image -> caption, per-word attention, regions, eligibility, association, semantics.

Runtime inspection pipeline (no training happens here)::

    image -> validation -> BLIP free generation (beam search, B1 = frozen pretrained BLIP)
          -> teacher-forced replay of the generated caption (prediction states h_t + last-layer cross-attention)
          -> word grouping of WordPiece tokens -> Grounding DINO regions (geometry check + per-label NMS)
          -> frozen OpenCLIP region embeddings v_r -> three-gate eligibility (P x C x U) + hard negatives
          -> class-agnostic attention association A(t, r) -> zero-shot OpenCLIP semantic inspection
          -> diagnostic per word

The Semantic Consistency Laboratory (``optimise``) is the training-side mechanism in miniature: a FRESH token projection
f_s is optimised with the repository's InfoNCE loss on ONE token, against frozen region embeddings. It never changes the
captioning model and says nothing about generalisation.
"""
from __future__ import annotations

import base64
import copy
import hashlib
import io
import logging
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from demo_app.association import associate
from src.grounding.cache import ImageGrounding, compute_grounding
from src.grounding.eligibility import (ACCEPTED, AMBIGUOUS_REGION, INVALID_ALIGNMENT, INVALID_REGION_GEOMETRY, LOW_CONFIDENCE,
                                       MULTIWORD_TERM, NO_DETECTION, NOT_CONCRETE_NOUN, NO_VALID_NEGATIVES, PosTagger,
                                       filter_tokens)
from src.grounding.pairing import build_pairs
from src.grounding.region_matching import COCO_CATEGORIES, normalize_word, singularize, word_to_category
from src.inference.predictor import CaptionPredictor
from src.inference.reliability import ReliabilityConfig, assess_caption, overall_flag
from src.losses.contrastive_loss import info_nce_with_stats
from src.models.token_projection import TokenProjection

logger = logging.getLogger(__name__)
SCHEMA_VERSION = "demo-1"
SOFTWARE_VERSION = "grounded-not-just-attending 0.2.0 / demo_app 1.0"

# Diagnostic categories shown in the interface (colour in brackets).
STRONG = "strong_semantic_agreement"          # green
INCONSISTENT = "potential_semantic_inconsistency"  # red
AMBIGUOUS = "ambiguous_evidence"              # amber
INSUFFICIENT = "insufficient_visual_evidence"  # purple/amber
INELIGIBLE = "ineligible_token"               # grey
DIAGNOSTICS = (STRONG, INCONSISTENT, AMBIGUOUS, INSUFFICIENT, INELIGIBLE)

_GATE_P_FAIL = {NOT_CONCRETE_NOUN, MULTIWORD_TERM, INVALID_ALIGNMENT}
_GATE_C_FAIL = {NO_DETECTION, LOW_CONFIDENCE, INVALID_REGION_GEOMETRY}


class DemoError(ValueError):
    """A request the demo cannot serve (shown to the user as-is; never contains local paths)."""


# --------------------------------------------------------------------------------------------------------------------------
# Frozen OpenCLIP text side (zero-shot inspection signal only)
# --------------------------------------------------------------------------------------------------------------------------
class ClipTextEncoder:
    """Text tower of the SAME frozen OpenCLIP model used for region embeddings (so image and text share one space)."""

    def __init__(self, region_encoder):
        import open_clip

        self.model = region_encoder.model
        self.device = region_encoder.device
        self.tokenizer = open_clip.get_tokenizer(region_encoder.model_name)
        self._cache: Dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def encode(self, texts: Sequence[str]) -> torch.Tensor:
        missing = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if missing:
            feats = F.normalize(self.model.encode_text(self.tokenizer(missing).to(self.device)).float(), dim=-1).cpu()
            for t, f in zip(missing, feats):
                self._cache[t] = f
        return torch.stack([self._cache[t] for t in texts]) if texts else torch.zeros(0, 1)


# --------------------------------------------------------------------------------------------------------------------------
@dataclass
class AnalysisState:
    """Server-side tensors of one analysis (needed for the counterfactual and the laboratory)."""
    public: Dict[str, Any]
    states: torch.Tensor                      # (n_words, H) prediction states h_t
    region_embeddings: torch.Tensor           # (N, D) frozen v_r
    image: Optional[Image.Image] = None       # None for cached replay
    grounding: Optional[ImageGrounding] = None
    pairs: Dict[int, Dict[str, Any]] = field(default_factory=dict)   # word index -> {positive, negatives}


def _png_b64(image: Image.Image, max_side: int = 768) -> str:
    im = image.convert("RGB").copy()
    im.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _r(x: Optional[float], n: int = 4) -> Optional[float]:
    return None if x is None or not np.isfinite(x) else round(float(x), n)


def _module_fingerprint(module: Optional[torch.nn.Module]) -> Optional[str]:
    """Cheap content hash of all parameters (sum + a strided sample of each tensor)."""
    if module is None:
        return None
    h = hashlib.sha256()
    for name, p in sorted(module.named_parameters(), key=lambda t: t[0]):
        t = p.detach().float().flatten()
        h.update(name.encode())
        h.update(np.float64(t.double().sum().item()).tobytes())
        h.update(t[:: max(1, t.numel() // 64)].cpu().numpy().tobytes())
    return h.hexdigest()


class DemoEngine:
    """Holds the models and an LRU of recent analyses. All heavy calls are serialised by one lock."""

    def __init__(self, demo_cfg: dict, model_cfg: dict, det_cfg: dict, *, blip=None, processor=None, detector=None,
                 region_encoder=None, text_encoder=None, device: Optional[torch.device] = None, tagger: Optional[PosTagger] = None):
        self.cfg, self.model_cfg, self.det_cfg = demo_cfg, model_cfg, det_cfg
        self.blip, self.processor, self.detector = blip, processor, detector
        self.region_encoder, self.text_encoder = region_encoder, text_encoder
        self.device = device or torch.device("cpu")
        self.tagger = tagger
        self.status = "ready" if blip is not None else "not_loaded"
        self.load_error: Optional[str] = None
        self.load_seconds: Optional[float] = None
        self._lock = threading.Lock()
        self._store: "OrderedDict[str, AnalysisState]" = OrderedDict()
        self._category_text: Optional[torch.Tensor] = None

    # ---------------------------------------------------------------------------------------------------------------- load
    def load(self) -> None:
        """Load BLIP, Grounding DINO and OpenCLIP (downloads on first use, then the Hugging Face cache)."""
        from src.grounding.detector import GroundingDinoDetector
        from src.grounding.region_features import RegionEncoder
        from src.models.blip_model import load_blip
        from src.utils.device import get_device

        self.status, t0 = "loading", time.time()
        try:
            self.device = get_device()
            self.blip, self.processor = load_blip(self.model_cfg["blip_model_name"], self.device)
            self.detector = GroundingDinoDetector(self.det_cfg["detector_checkpoint"], device=self.device,
                                                  text_threshold=float(self.det_cfg.get("text_threshold", 0.25)))
            self.region_encoder = RegionEncoder(self.model_cfg["clip_model_name"], self.model_cfg.get("clip_pretrained"), self.device)
            self.text_encoder = ClipTextEncoder(self.region_encoder)
            self.tagger = PosTagger()
            self._warm_up()
            self.status, self.load_seconds = "ready", round(time.time() - t0, 1)
        except Exception as exc:  # noqa: BLE001 - the real failure is shown on the page, cached replay stays available
            logger.exception("model loading failed")
            self.status, self.load_error = "error", f"{type(exc).__name__}: {exc}"

    def _warm_up(self) -> None:
        img = Image.new("RGB", (64, 64), (127, 127, 127))
        CaptionPredictor(self.blip, self.processor, self.device, None, {"num_beams": 1, "max_new_tokens": 3, "min_new_tokens": 1}).predict_caption(img)
        self._categories()

    @property
    def ready(self) -> bool:
        return self.status == "ready"

    def info(self) -> Dict[str, Any]:
        return {"status": self.status, "error": self.load_error, "device": str(self.device), "load_seconds": self.load_seconds,
                "captioner": self.model_cfg.get("blip_model_name"), "captioner_role": "B1: frozen pretrained BLIP (no fine-tuning)",
                "detector": self.det_cfg.get("detector_checkpoint"),
                "visual_semantic_encoder": f"OpenCLIP {self.model_cfg.get('clip_model_name')} ({self.model_cfg.get('clip_pretrained')}), frozen",
                "pos_tagger": self.tagger.info() if self.tagger is not None else None}

    # ---------------------------------------------------------------------------------------------------------------- helpers
    def _categories(self) -> torch.Tensor:
        if self._category_text is None:
            tpl = self.cfg["semantic"]["prompt"]
            self._category_text = self.text_encoder.encode([tpl.format(c) for c in COCO_CATEGORIES])
        return self._category_text

    def _semantic(self, word: str, category: str, v: torch.Tensor) -> Dict[str, Any]:
        """Zero-shot inspection signal s_inspect(t, r) = cos(E_text(word), E_image(r)) and its margin over the other COCO classes."""
        tpl = self.cfg["semantic"]["prompt"]
        t_word = self.text_encoder.encode([tpl.format(word)])[0]
        s = float(t_word @ v)
        cat_sims = self._categories() @ v
        own = COCO_CATEGORIES.index(category)
        others = torch.cat([cat_sims[:own], cat_sims[own + 1:]])
        best_other = int(torch.argmax(others))
        best_other = best_other if best_other < own else best_other + 1
        top_cat = COCO_CATEGORIES[int(torch.argmax(cat_sims))]
        margin = s - float(cat_sims[best_other])
        rank = 1 + int((others > s).sum())
        return {"similarity": s, "margin": margin, "rank_among_80": rank, "best_competing_category": COCO_CATEGORIES[best_other],
                "region_top_category": top_cat}

    def _diagnose(self, assoc_status: str, sem: Optional[Dict[str, Any]]) -> Tuple[str, str]:
        amb = float(self.cfg["semantic"]["ambiguous_margin"])
        if assoc_status == "no_regions":
            return INSUFFICIENT, "The detector found no region in the image, so there is no visual evidence to compare against."
        if assoc_status == "insufficient":
            return INSUFFICIENT, "The word's attention is spread outside every detected region; the object may be undetected or the attention diffuse."
        if assoc_status == "ambiguous":
            return AMBIGUOUS, "Two regions receive almost the same share of the word's attention, so no single region can be compared."
        m = sem["margin"]
        if m >= 0:
            return STRONG, f"Among the 80 COCO classes the attended region agrees best with this word (margin {m:+.3f})."
        if m >= -amb:
            return AMBIGUOUS, f"The attended region agrees almost equally with '{sem['best_competing_category']}' (margin {m:+.3f})."
        return INCONSISTENT, (f"The model attended a region that OpenCLIP matches better to '{sem['best_competing_category']}' than to this word "
                              f"(margin {m:+.3f}). This is a flag to inspect, not proof of an error: the crop may be poor, the detector "
                              "may have missed the object, or the word may be a valid synonym.")

    # ---------------------------------------------------------------------------------------------------------------- analyse
    def analyse(self, image: Image.Image, caption: Optional[str] = None, grounding: Optional[ImageGrounding] = None,
                mode: str = "live", meta: Optional[Dict[str, Any]] = None) -> AnalysisState:
        if not self.ready:
            raise DemoError(f"Models are not ready (status: {self.status}). Use cached replay or wait for loading to finish.")
        with self._lock:
            return self._analyse(image, caption, grounding, mode, meta or {})

    @torch.no_grad()
    def _analyse(self, image, caption, grounding, mode, meta) -> AnalysisState:
        timings: Dict[str, float] = {}
        warnings: List[str] = []
        t = time.time()
        pred = CaptionPredictor(self.blip, self.processor, self.device, None, self.cfg.get("decoding"), "B1")
        if caption is None:
            caption = pred.predict_caption(image)
            timings["caption_generation_s"] = round(time.time() - t, 3)
        t = time.time()
        gen = pred.analyse(image, caption)          # teacher-forced replay: states + last-layer cross-attention per word
        timings["teacher_forced_replay_s"] = round(time.time() - t, 3)
        t = time.time()
        if grounding is None:
            grounding = compute_grounding(0, image, self.detector, self.region_encoder, float(self.det_cfg.get("box_threshold", 0.3)),
                                          float(self.det_cfg.get("nms_iou", 0.5)))
        timings["detection_and_region_embedding_s"] = round(time.time() - t, 3)
        t = time.time()
        dets = grounding.detections
        if not dets:
            warnings.append("The detector returned no region above the box threshold: every object word is 'insufficient evidence'.")
        theta_c = float(self.det_cfg.get("confidence_threshold", 0.5))
        records = filter_tokens(gen.words, dets, threshold=theta_c, iou_margin=float(self.det_cfg.get("nms_iou", 0.5)),
                                uniqueness_mode=self.det_cfg.get("uniqueness_mode", "strict"),
                                uniqueness_margin=float(self.det_cfg.get("uniqueness_margin", 0.1)), tagger=self.tagger,
                                token_indices=gen.token_positions, image_size=image.size)
        pairs = build_pairs(records, grounding, int(self.det_cfg.get("num_negatives", 5)), float(self.det_cfg.get("iou_threshold", 0.5)))
        pair_by_word = {p.record.token_pos: {"positive": p.positive_index, "negatives": list(p.negative_indices)} for p in pairs}
        rel = assess_caption(gen.words, grounding, None, ReliabilityConfig(theta_c, 0.2), self.tagger)
        acfg = self.cfg["association"]
        regions = [{"region_id": i, "label": d.label, "confidence": _r(d.score), "bbox": [round(v, 1) for v in d.bbox],
                    "width": round(d.bbox[2] - d.bbox[0], 1), "height": round(d.bbox[3] - d.bbox[1], 1)} for i, d in enumerate(dets)]
        ids = self.processor.tokenizer(caption, truncation=True)["input_ids"]
        tokens: List[Dict[str, Any]] = []
        char = 0
        for i, (word, rec, rr) in enumerate(zip(gen.words, records, rel)):
            start = caption.lower().find(word.lower(), char)
            if start >= 0:
                char = start + len(word)
            grid = gen.attention[i]
            assoc = associate(grid, [d.bbox for d in dets], image.size, float(acfg["min_mass"]), float(acfg["min_margin"]))
            is_object = rr.claim_type == "object" and rr.category is not None
            sem = None
            if is_object and assoc["selected_region_id"] is not None:
                base = singularize(word) if rec.pos_tag in ("NN", "NNS") else normalize_word(word)
                sem = self._semantic(base, rr.category, grounding.embeddings[assoc["selected_region_id"]])
                sem = {k: (_r(v) if isinstance(v, float) else v) for k, v in sem.items()}
            if is_object:
                diag, why = self._diagnose(assoc["status"], sem)
            else:
                diag = INELIGIBLE
                why = ("Not an object claim the inspector can check" + (f" ({rr.note})" if rr.note else "") +
                       ". Ineligible means not assessed, never 'incorrect'.")
            outcome = rec.outcome
            m_pos = outcome not in _GATE_P_FAIL
            m_conf = None if not m_pos else outcome not in _GATE_C_FAIL
            m_uniq = None if not m_conf else outcome != AMBIGUOUS_REGION
            tokens.append({
                "word_index": i, "token": word, "token_ids": [int(ids[q]) for q in gen.token_positions[i] if q < len(ids)],
                "subword_positions": gen.token_positions[i], "char_offsets": [start, start + len(word)] if start >= 0 else None,
                "pos_tag": rec.pos_tag, "claim_type": rr.claim_type, "coco_category": rr.category,
                "eligibility": {"m_pos": m_pos, "m_confidence": m_conf, "m_uniqueness": m_uniq, "eligible": outcome == ACCEPTED,
                                "outcome": outcome, "rule": rec.rule, "detector_confidence": _r(rec.detector_confidence) if m_conf else None,
                                "uniqueness_score": _r(rec.uniqueness_score) if m_conf else None,
                                "positive_region_id": pair_by_word.get(i, {}).get("positive"),
                                "hard_negative_region_ids": pair_by_word.get(i, {}).get("negatives", []),
                                "note": "Training-time supervision gate m_t = m_POS x m_conf x m_uniq (src/grounding/eligibility.py); "
                                        "a rejected word is ineligible or ambiguous, not incorrect."},
                "attention": {"grid": np.round(grid, 6).tolist(), "grid_size": list(grid.shape), "layer": "last decoder layer",
                              "aggregation": "mean over heads", "read_at": "prediction position (the state that emits the word)"},
                "association": {**{k: v for k, v in assoc.items() if k != "candidates"},
                                "top_mass": _r(assoc["top_mass"]), "margin": _r(assoc["margin"]),
                                "candidates": [{k: _r(v) if isinstance(v, float) else v for k, v in c.items()} for c in assoc["candidates"]]},
                "semantic": sem, "score_provenance": "zero-shot frozen OpenCLIP text/image cosine (inspection signal, not f_s, not a probability)" if sem else None,
                "detector_evidence": {"status": rr.status, "rule": rr.rule, "note": rr.note, "region_id": rr.region_id},
                "diagnostic": diag, "explanation": why,
            })
        timings["analysis_s"] = round(time.time() - t, 3)
        counts = {d: sum(1 for tk in tokens if tk["diagnostic"] == d) for d in DIAGNOSTICS}
        public = {
            "schema_version": SCHEMA_VERSION, "analysis_id": uuid.uuid4().hex[:12], "mode": mode,
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "software_version": SOFTWARE_VERSION,
            "model": {"captioner": self.model_cfg.get("blip_model_name"), "system": "B1 (frozen pretrained BLIP)",
                      "revision": str(getattr(self.blip.config, "_commit_hash", None) or "unknown"), "device": str(self.device),
                      "decoding": {**{"num_beams": 3, "max_new_tokens": 30, "min_new_tokens": 5}, **(self.cfg.get("decoding") or {})},
                      "detector": self.det_cfg.get("detector_checkpoint"),
                      "visual_semantic_encoder": f"OpenCLIP {self.model_cfg.get('clip_model_name')} / {self.model_cfg.get('clip_pretrained')}"},
            "configuration": {"theta_c": theta_c, "box_threshold": self.det_cfg.get("box_threshold"), "nms_iou": self.det_cfg.get("nms_iou"),
                              "uniqueness_mode": self.det_cfg.get("uniqueness_mode"), "num_negatives": self.det_cfg.get("num_negatives"),
                              "association": acfg, "semantic": self.cfg["semantic"],
                              "heuristic_note": "association and semantic thresholds are heuristic demo settings, not validated values"},
            "image": {"width": image.size[0], "height": image.size[1], "data_url": _png_b64(image), **meta},
            "caption": caption, "regions": regions, "tokens": tokens,
            "summary": {"diagnostic_counts": counts, "eligible_words": sum(1 for tk in tokens if tk["eligibility"]["eligible"]),
                        "regions": len(regions), "detector_evidence_overall": overall_flag(rel)},
            "timings": timings, "warnings": warnings,
        }
        st = AnalysisState(public, gen.states.detach().float().cpu(), grounding.embeddings.float().cpu(), image, grounding, pair_by_word)
        self._remember(st)
        return st

    def _remember(self, st: AnalysisState) -> None:
        self._store[st.public["analysis_id"]] = st
        while len(self._store) > int(self.cfg.get("max_cached_analyses", 16)):
            self._store.popitem(last=False)

    def get(self, analysis_id: str) -> AnalysisState:
        st = self._store.get(analysis_id)
        if st is None:
            raise DemoError("Unknown or expired analysis id; run the analysis again or reload the case.")
        return st

    # ---------------------------------------------------------------------------------------------------------------- replay
    def register_case(self, case: Dict[str, Any]) -> AnalysisState:
        """Load a saved analysis for CACHED REPLAY (no model needed; the counterfactual needs live models)."""
        rs = case.get("replay_state") or {}
        public = {k: v for k, v in case.items() if k != "replay_state"}
        public = copy.deepcopy(public)
        public["mode"] = "cached_replay"
        public["analysis_id"] = "case-" + str(case.get("case_id", uuid.uuid4().hex[:6]))
        states = torch.tensor(rs.get("states", []), dtype=torch.float32)
        emb = torch.tensor(rs.get("region_embeddings", []), dtype=torch.float32)
        pairs = {int(k): v for k, v in (rs.get("pairs") or {}).items()}
        st = AnalysisState(public, states, emb, None, None, pairs)
        self._remember(st)
        return st

    @staticmethod
    def case_record(st: AnalysisState, case_id: str, source: str, license_: str, purpose: str) -> Dict[str, Any]:
        out = copy.deepcopy(st.public)
        out.update({"case_id": case_id, "mode": "cached_replay"})
        out["image"].update({"source": source, "license": license_, "purpose": purpose})
        out["replay_state"] = {"states": [[round(float(x), 6) for x in row] for row in st.states.tolist()],
                               "region_embeddings": [[round(float(x), 6) for x in row] for row in st.region_embeddings.tolist()],
                               "pairs": {str(k): v for k, v in st.pairs.items()}}
        return out

    # ---------------------------------------------------------------------------------------------------------------- counterfactual
    def counterfactual(self, analysis_id: str, word_index: int, replacement: str) -> Dict[str, Any]:
        st = self.get(analysis_id)
        if st.image is None:
            raise DemoError("The controlled perturbation re-runs BLIP on the edited caption, so it needs live models (not available in cached replay).")
        rep = replacement.strip().lower()
        if not rep.isalpha() or not 1 <= len(rep) <= 30:
            raise DemoError("The replacement must be a single word made of letters.")
        toks = st.public["tokens"]
        if not 0 <= word_index < len(toks):
            raise DemoError("word_index is out of range")
        orig = toks[word_index]
        if orig["diagnostic"] == INELIGIBLE:
            raise DemoError("Choose an object word (an assessable noun) to perturb.")
        words = [t["token"] for t in toks]
        words[word_index] = rep
        edited = " ".join(words).replace(" .", ".").replace(" ,", ",")
        new = self.analyse(st.image, caption=edited, grounding=st.grounding, mode="controlled_perturbation",
                           meta={k: v for k, v in st.public["image"].items() if k not in ("width", "height", "data_url")})
        if len(new.public["tokens"]) != len(toks):
            raise DemoError("The edited caption no longer aligns word-for-word with the original (tokenisation changed).")
        ctrl = new.public["tokens"][word_index]
        # Fixed-region counterfactual: the ORIGINAL word's attended region is held constant, only the word changes.
        fixed = None
        rid = orig["association"]["selected_region_id"]
        cat = word_to_category(singularize(rep))
        if rid is not None and cat is not None:
            with self._lock:
                sem = self._semantic(singularize(rep), cat, st.region_embeddings[rid])
            fixed = {"region_id": rid, "spatial_mass": orig["association"]["top_mass"],
                     **{k: (_r(v) if isinstance(v, float) else v) for k, v in sem.items()}}
            fixed["diagnostic"], fixed["explanation"] = self._diagnose(orig["association"]["status"], fixed)
        new.public["controlled"] = {"original_caption": st.public["caption"], "edited_word_index": word_index,
                                    "original_word": orig["token"], "replacement": rep}

        def side(t):
            return {"token": t["token"], "region_id": t["association"]["selected_region_id"], "spatial_mass": t["association"]["top_mass"],
                    "association_status": t["association"]["status"], "semantic": t["semantic"], "diagnostic": t["diagnostic"],
                    "explanation": t["explanation"]}

        return {"label": "CONTROLLED PERTURBATION - edited by the user, not a spontaneous model error",
                "original_caption": st.public["caption"], "controlled_caption": edited,
                "original": side(orig), "controlled_recomputed": side(ctrl),
                "controlled_fixed_region": fixed,
                "notes": ["'recomputed': BLIP is teacher-forced on the edited caption, so the attention of the new word is genuinely extracted "
                          "from the model when it is made to emit that word.",
                          "'fixed-region counterfactual': the original word's attended region is held constant and only the word changes, "
                          "isolating semantic disagreement from spatial association."
                          if fixed else "Fixed-region comparison unavailable: the replacement is outside the 80-class COCO vocabulary or no region was associated."],
                "controlled_analysis": new.public}

    # ---------------------------------------------------------------------------------------------------------------- laboratory
    def _lab_inputs(self, analysis_id: str, word_index: int, num_negatives: Optional[int]):
        st = self.get(analysis_id)
        toks = st.public["tokens"]
        if not 0 <= word_index < len(toks):
            raise DemoError("word_index is out of range")
        pair = st.pairs.get(word_index)
        if pair is None:
            el = toks[word_index]["eligibility"]
            raise DemoError(f"'{toks[word_index]['token']}' has no valid training pair (outcome {el['outcome']}). "
                            "Pick an eligible word: the positive must be a confident, unique region of the word's category with at least one hard negative.")
        if st.states.numel() == 0 or st.region_embeddings.numel() == 0:
            raise DemoError("This saved case has no stored representations; regenerate it with demo_app/prepare_cases.py.")
        negs = pair["negatives"][: num_negatives] if num_negatives else pair["negatives"]
        return st, toks[word_index], pair["positive"], negs

    def preview(self, analysis_id: str, word_index: int, temperature: float, num_negatives: Optional[int] = None) -> Dict[str, Any]:
        st, tok, pos, negs = self._lab_inputs(analysis_id, word_index, num_negatives)
        res = self.optimise(analysis_id, word_index, steps=0, lr=1e-3, temperature=temperature, lam=1.0, num_negatives=num_negatives)
        return {k: res[k] for k in ("token", "positive", "negatives", "initial", "temperature", "dims", "parameter_counts", "interpretation")}

    def optimise(self, analysis_id: str, word_index: int, steps: int, lr: float, temperature: float, lam: float,
                 num_negatives: Optional[int] = None) -> Dict[str, Any]:
        lab = self.cfg["laboratory"]
        if not 0 <= steps <= int(lab["max_steps"]):
            raise DemoError(f"steps must be between 0 and {lab['max_steps']}")
        if not 1e-5 <= lr <= 1.0 or not 0.01 <= temperature <= 2.0 or not 0.0 < lam <= 10.0:
            raise DemoError("learning rate must be in [1e-5, 1], temperature in [0.01, 2], lambda in (0, 10]")
        st, tok, pos, negs = self._lab_inputs(analysis_id, word_index, num_negatives)
        h = st.states[word_index].clone().detach()                       # decoder prediction state h_t (constant)
        v_pos = st.region_embeddings[pos].clone().detach()               # frozen v_r+
        v_neg = st.region_embeddings[negs].clone().detach()              # frozen same-image hard negatives
        torch.manual_seed(int(lab["seed"]))
        f_s = TokenProjection(h.shape[-1], v_pos.shape[-1], int(self.model_cfg.get("projection_layers", 1)),
                              int(self.model_cfg.get("projection_hidden", 512)))
        frozen = {"BLIP captioner": self.blip, "OpenCLIP encoder f_v": getattr(self.region_encoder, "model", None)}
        frozen = {k: v for k, v in frozen.items() if isinstance(v, torch.nn.Module)}
        before = {k: _module_fingerprint(m) for k, m in frozen.items()}
        emb_before = st.region_embeddings.clone()
        opt = torch.optim.Adam(f_s.parameters(), lr=lr)
        frozen_ids = {id(p) for m in frozen.values() for p in m.parameters()}
        in_opt = sum(1 for g in opt.param_groups for p in g["params"] if id(p) in frozen_ids)
        traj: List[Dict[str, Any]] = []
        zs: List[np.ndarray] = []

        def record(step: int, z: torch.Tensor, loss: torch.Tensor) -> None:
            with torch.no_grad():
                sp = float(z @ v_pos)
                sn = (v_neg @ z).tolist()
                logits = [sp / temperature] + [s / temperature for s in sn]
            traj.append({"step": step, "loss": float(loss), "weighted_loss": float(lam * loss), "pos_sim": sp, "neg_sims": sn,
                         "hardest_neg_sim": max(sn), "margin": sp - max(sn), "logits": logits})
            zs.append(z.detach().numpy().copy())

        with self._lock, torch.enable_grad():
            for step in range(steps + 1):
                z = f_s(h.unsqueeze(0))
                loss, stats = info_nce_with_stats(z, v_pos.unsqueeze(0), v_neg.unsqueeze(0), temperature)
                if not torch.isfinite(loss):
                    raise DemoError("Loss became non-finite; lower the learning rate or raise the temperature.")
                record(step, z[0], loss)
                if step == steps:
                    break
                opt.zero_grad()
                (lam * loss).backward()
                opt.step()
        after = {k: _module_fingerprint(m) for k, m in frozen.items()}
        n_train = sum(p.numel() for p in f_s.parameters())
        n_frozen = sum(p.numel() for m in frozen.values() for p in m.parameters())
        # 2-D PCA of the 512-D space (illustration only): token trajectory + positive + negatives.
        pts = np.vstack(zs + [v_pos.numpy()] + [v.numpy() for v in v_neg])
        c = pts - pts.mean(0, keepdims=True)
        _, _, vt = np.linalg.svd(c, full_matrices=False)
        xy = (c @ vt[:2].T).round(5).tolist() if c.shape[0] > 1 else [[0.0, 0.0]] * len(pts)
        nz = len(zs)
        regions = st.public["regions"]
        return {
            "token": {"word_index": word_index, "token": tok["token"], "pos_tag": tok["pos_tag"]},
            "positive": {**regions[pos], "role": "positive v+ (detector-assigned region of the word's category)"},
            "negatives": [{**regions[j], "role": "same-image hard negative"} for j in negs],
            "temperature": temperature, "lambda": lam, "learning_rate": lr, "steps_requested": steps, "steps_run": len(traj) - 1,
            "dims": {"h_t": int(h.shape[-1]), "z_t": int(v_pos.shape[-1]), "v_r": int(v_pos.shape[-1])},
            "token_hidden_preview": [round(float(x), 4) for x in h[:12]],
            "initial": traj[0], "final": traj[-1], "trajectory": [{k: v for k, v in s.items() if k != "logits"} for s in traj],
            "pca_2d": {"token_path": xy[:nz], "positive": xy[nz], "negatives": xy[nz + 1:],
                       "note": "2-D PCA of the 512-D embedding space for illustration only; distances are not preserved."},
            "parameter_counts": {"trainable_f_s": n_train, "frozen": n_frozen, "frozen_modules": list(frozen)},
            "frozen_check": {"fingerprints_unchanged": before == after, "frozen_params_in_optimizer": in_opt,
                             "region_embeddings_unchanged": bool(torch.equal(emb_before, st.region_embeddings)),
                             "modules_checked": list(frozen) or ["(cached replay: no model loaded; region embeddings are constants)"]},
            "objective": "L_total = L_caption + lambda * L_cons; here BLIP is frozen, so L_caption is a constant and only lambda * L_cons "
                         "(InfoNCE, src/losses/contrastive_loss.py) is optimised with respect to f_s.",
            "interpretation": "A falling loss shows the objective is differentiable and can pull this ONE projected token state towards its "
                              "positive region and away from same-image negatives. It is not a trained model, not evidence of better "
                              "generalisation, and not evidence of fewer hallucinations. The f_s head starts from a random initialisation (seed "
                              f"{lab['seed']}); its initial scores carry no meaning.",
        }
