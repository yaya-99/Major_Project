"""
Major Project Configuration
===========================

Project:
    Grounded, Not Just Attending:
    A Targeted Consistency Objective for Attention-Correct,
    Semantically Incorrect Image Captioning Errors

Purpose:
    Central configuration for the Major project.

This file contains:
    - Project paths
    - Dataset configuration
    - BLIP model configuration
    - OpenCLIP visual representation configuration
    - Region/eligibility configuration
    - LoRA configuration
    - Consistency-loss configuration
    - Caption generation configuration
    - Reproducibility settings
    - Experiment output paths
    - Basic configuration validation

Important:
    This file intentionally does NOT contain:
    - SALICON fixation/saliency paths
    - Simulated saliency
    - Minor-project model comparisons
    - Minor scenario categories
    - Evaluation results
    - Training code
    - Model loading code

Exact model/checkpoint choices that depend on the audit of the Minor
BLIP implementation should be locked after Notebook 02 is audited.
"""

from pathlib import Path


# =============================================================================
# 1. PROJECT PATHS
# =============================================================================

# Root directory of the Major repository.
PROJECT_ROOT = Path(__file__).resolve().parent

# Main project directories.
DATA_DIR = PROJECT_ROOT / "data"
SRC_DIR = PROJECT_ROOT / "src"
CONFIG_DIR = PROJECT_ROOT / "configs"
DOCS_DIR = PROJECT_ROOT / "docs"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
TESTS_DIR = PROJECT_ROOT / "tests"

# Data subdirectories.
COCO_DIR = DATA_DIR / "coco"
ANNOTATIONS_DIR = COCO_DIR / "annotations"
IMAGES_DIR = COCO_DIR / "images"

# Diagnostic data migrated from the Minor project.
# This data is NOT training data.
DIAGNOSTICS_DIR = DATA_DIR / "diagnostics"

# Experiment outputs.
OUTPUT_DIR = EXPERIMENTS_DIR / "outputs"
CHECKPOINT_DIR = EXPERIMENTS_DIR / "checkpoints"
LOG_DIR = EXPERIMENTS_DIR / "logs"
PREDICTIONS_DIR = EXPERIMENTS_DIR / "predictions"
METRICS_DIR = EXPERIMENTS_DIR / "metrics"
FIGURES_DIR = EXPERIMENTS_DIR / "figures"


# =============================================================================
# 2. DATASET CONFIGURATION
# =============================================================================

# Primary dataset.
DATASET_NAME = "MS_COCO"

# Dataset splits.
TRAIN_SPLIT = "train2014"
DEV_SPLIT = "val2014"
TEST_SPLIT = "test2014"

# Active split names used by individual scripts.
# Do not use TEST_SPLIT for training.
ACTIVE_TRAIN_SPLIT = TRAIN_SPLIT
ACTIVE_DEV_SPLIT = DEV_SPLIT
ACTIVE_TEST_SPLIT = TEST_SPLIT

# COCO annotation files.
CAPTIONS_TRAIN_JSON = ANNOTATIONS_DIR / "captions_train2014.json"
CAPTIONS_VAL_JSON = ANNOTATIONS_DIR / "captions_val2014.json"

INSTANCES_TRAIN_JSON = ANNOTATIONS_DIR / "instances_train2014.json"
INSTANCES_VAL_JSON = ANNOTATIONS_DIR / "instances_val2014.json"

# Optional test annotation path.
# COCO test images do not normally have public ground-truth captions in
# the same form as train/val. Keep this configurable rather than assuming
# a caption annotation file exists.
CAPTIONS_TEST_JSON = ANNOTATIONS_DIR / "captions_test2014.json"

# Image directories.
TRAIN_IMAGES_DIR = IMAGES_DIR / "train2014"
VAL_IMAGES_DIR = IMAGES_DIR / "val2014"
TEST_IMAGES_DIR = IMAGES_DIR / "test2014"

# Dataset limits.
#
# None means use the complete configured split.
# During development, set these to a small integer for smoke tests.
TRAIN_MAX_IMAGES = None
DEV_MAX_IMAGES = None
TEST_MAX_IMAGES = None

# Development/debugging limit.
DEBUG_MODE = False
DEBUG_MAX_IMAGES = 32


# =============================================================================
# 3. REPRODUCIBILITY
# =============================================================================

SEED = 42

# When True, training/evaluation code should attempt to use deterministic
# algorithms where supported. Exact behavior is controlled by the training
# implementation.
DETERMINISTIC = True


# =============================================================================
# 4. DEVICE / NUMERICAL PRECISION
# =============================================================================

# "auto" lets the model/training code select an available device.
# Valid explicit values include:
#     "cpu"
#     "cuda"
DEVICE = "auto"

# Mixed precision.
USE_MIXED_PRECISION = True

# Supported values depend on hardware:
#     "fp16"
#     "bf16"
#     None
MIXED_PRECISION_DTYPE = "bf16"


# =============================================================================
# 5. BASELINE CAPTIONING MODEL
# =============================================================================

# Major project is BLIP-centered.
#
# This is the BLIP checkpoint identified in the Minor configuration.
# The exact checkpoint should be re-confirmed against Notebook 02 before
# the baseline implementation is considered locked.
BLIP_MODEL_NAME = "Salesforce/blip-image-captioning-large"

# Processor/tokenizer are obtained from the same checkpoint unless the
# implementation explicitly overrides them.
BLIP_PROCESSOR_NAME = BLIP_MODEL_NAME
BLIP_TOKENIZER_NAME = BLIP_MODEL_NAME

# Model architecture is expected to provide:
#     image encoder
#     text decoder
#     contextual decoder hidden states
#
# Exact hidden-state dimensions must be verified during Notebook 02 audit.
BLIP_EXPECTED_CONTEXTUAL_REPRESENTATION = True


# =============================================================================
# 6. VISUAL-SEMANTIC REPRESENTATION
# =============================================================================

# OpenCLIP is used as the frozen visual-semantic representation source
# for the consistency objective.
#
# The exact OpenCLIP checkpoint must remain locked for the duration of
# the experiments once selected.
OPENCLIP_MODEL_NAME = "ViT-B-32"

# Initial OpenCLIP pretrained identifier.
#
# This is intentionally kept configurable because the exact checkpoint
# must be verified/locked before the first official experiment.
OPENCLIP_PRETRAINED = "laion2b_s34b_b79k"

# The visual projection is frozen.
OPENCLIP_FROZEN = True

# Normalize visual representations before contrastive similarity.
NORMALIZE_VISUAL_REPRESENTATIONS = True


# =============================================================================
# 7. REGION EXTRACTION / ELIGIBILITY
# =============================================================================

# Region extraction is part of the Major pipeline.
#
# The exact detector/checkpoint is intentionally configurable and should
# be locked before official experiments. Do not silently substitute a
# different detector between experiments.
REGION_DETECTOR_NAME = None
REGION_DETECTOR_CHECKPOINT = None

# Detector confidence threshold.
REGION_CONFIDENCE_THRESHOLD = 0.50

# Minimum region uniqueness criterion.
#
# This is a configuration placeholder for the implementation that will
# determine whether a candidate region is sufficiently unique/relevant.
REGION_UNIQUENESS_THRESHOLD = 0.50

# Maximum number of regions retained per image.
MAX_REGIONS_PER_IMAGE = 100

# Region representation source.
REGION_REPRESENTATION_SOURCE = "openclip"


# =============================================================================
# 8. TOKEN ELIGIBILITY
# =============================================================================

# Eligible caption tokens are selected using linguistic and visual criteria.
#
# The Major specification uses concrete noun tokens as the primary
# eligibility criterion.
ELIGIBLE_POS_TAGS = {
    "NN",
    "NNS",
    "NNP",
    "NNPS",
}

# Tokens that cannot be reliably mapped to a concrete visual region should
# not receive the consistency loss.
#
# They still participate in the normal captioning loss.
APPLY_CAPTION_LOSS_TO_INELIGIBLE_TOKENS = True
APPLY_CONSISTENCY_LOSS_TO_INELIGIBLE_TOKENS = False


# =============================================================================
# 9. TOKEN REPRESENTATION
# =============================================================================

# The Major uses contextual decoder representations under teacher forcing.
TOKEN_REPRESENTATION_TYPE = "contextual_decoder_hidden_state"

# Static word embeddings are NOT used as the primary semantic representation.
USE_STATIC_WORD_EMBEDDINGS = False

# Teacher forcing is required for training-time token representations.
USE_TEACHER_FORCING = True


# =============================================================================
# 10. TRAINABLE TOKEN PROJECTION
# =============================================================================

# Token representations are projected into the same representation space
# as the frozen visual-semantic representation.
TOKEN_PROJECTION_ENABLED = True

# The input dimension is intentionally not hard-coded until Notebook 02
# verifies the exact BLIP decoder hidden-state dimension.
TOKEN_PROJECTION_INPUT_DIM = None

# Output dimension must match the frozen visual representation dimension.
TOKEN_PROJECTION_OUTPUT_DIM = None

# Projection architecture.
TOKEN_PROJECTION_HIDDEN_DIM = None

# Projection dropout.
TOKEN_PROJECTION_DROPOUT = 0.10


# =============================================================================
# 11. CONSISTENCY / CONTRASTIVE OBJECTIVE
# =============================================================================

# Core Major hypothesis:
#
#     L_total = L_caption + lambda * L_object
#
# where L_object is a contrastive consistency objective between eligible
# contextual token representations and corresponding visual representations.

CONSISTENCY_LOSS_ENABLED = True

# Weight of the consistency objective.
CONSISTENCY_LOSS_WEIGHT = 1.0

# InfoNCE temperature.
CONTRASTIVE_TEMPERATURE = 0.07

# Normalize both projected token and visual representations before
# computing similarity.
NORMALIZE_CONTRASTIVE_REPRESENTATIONS = True

# Use same-image hard negatives where available.
USE_SAME_IMAGE_HARD_NEGATIVES = True

# Numerical stability epsilon.
LOSS_EPSILON = 1e-8


# =============================================================================
# 12. LORA CONFIGURATION
# =============================================================================

# LoRA is used for trainable adaptation of the captioning model.
LORA_ENABLED = True

LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

# Target modules must be verified against the exact BLIP architecture
# during Notebook 02/model implementation audit.
LORA_TARGET_MODULES = None

# Whether the original BLIP model parameters remain frozen except for
# LoRA adapters and explicitly trainable Major modules.
FREEZE_BASE_MODEL = True


# =============================================================================
# 13. EXPERIMENT CONDITIONS
# =============================================================================

# These are the Major experimental conditions.
#
# 1. Frozen pretrained BLIP
# 2. LoRA fine-tuning with caption loss only
# 3. LoRA fine-tuning with caption loss + consistency objective
EXPERIMENT_FROZEN_BASELINE = "frozen_baseline"
EXPERIMENT_LORA_BASELINE = "lora_baseline"
EXPERIMENT_CONSISTENCY = "consistency"

EXPERIMENTS = [
    EXPERIMENT_FROZEN_BASELINE,
    EXPERIMENT_LORA_BASELINE,
    EXPERIMENT_CONSISTENCY,
]


# =============================================================================
# 14. CAPTION GENERATION
# =============================================================================

# These parameters are for free-generation inference.
# They are NOT the teacher-forcing training configuration.
GENERATION_CONFIG = {
    "max_new_tokens": 80,
    "num_beams": 5,
    "length_penalty": 1.1,
    "no_repeat_ngram_size": 2,
    "early_stopping": True,
}

# Whether to return generated token IDs during inference.
RETURN_GENERATED_TOKEN_IDS = True

# Whether inference should return decoder hidden states when supported.
RETURN_GENERATED_HIDDEN_STATES = False


# =============================================================================
# 15. TRAINING CONFIGURATION
# =============================================================================

# These are initial development values and should be tuned/locked through
# the experimental protocol rather than silently changed between runs.
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4

LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.01

NUM_EPOCHS = 3

WARMUP_RATIO = 0.05

MAX_TEXT_LENGTH = 128

# Gradient clipping.
MAX_GRAD_NORM = 1.0

# Save/evaluation frequency is expressed in training steps.
SAVE_EVERY_STEPS = 500
EVAL_EVERY_STEPS = 500
LOG_EVERY_STEPS = 50


# =============================================================================
# 16. DATALOADER CONFIGURATION
# =============================================================================

NUM_WORKERS = 4

PIN_MEMORY = True
PERSISTENT_WORKERS = True

DROP_LAST_TRAIN_BATCH = False


# =============================================================================
# 17. EXPERIMENT MANIFEST
# =============================================================================

# Every official run should save a manifest containing the configuration
# used for that run.
SAVE_EXPERIMENT_MANIFEST = True

# Git commit should be recorded by the experiment runner when available.
RECORD_GIT_COMMIT = True


# =============================================================================
# 18. HELPER FUNCTIONS
# =============================================================================

def get_image_dir(split: str) -> Path:
    """
    Return the image directory corresponding to a COCO split.

    Args:
        split: "train2014", "val2014", or "test2014"

    Returns:
        Path to the corresponding image directory.
    """
    split_to_dir = {
        TRAIN_SPLIT: TRAIN_IMAGES_DIR,
        DEV_SPLIT: VAL_IMAGES_DIR,
        TEST_SPLIT: TEST_IMAGES_DIR,
    }

    if split not in split_to_dir:
        raise ValueError(
            f"Unsupported split: {split}. "
            f"Expected one of {list(split_to_dir)}"
        )

    return split_to_dir[split]


def get_caption_annotation_path(split: str) -> Path:
    """
    Return the COCO caption annotation file for a split.

    Note:
        COCO test caption annotations may not be available locally.
    """
    split_to_path = {
        TRAIN_SPLIT: CAPTIONS_TRAIN_JSON,
        DEV_SPLIT: CAPTIONS_VAL_JSON,
        TEST_SPLIT: CAPTIONS_TEST_JSON,
    }

    if split not in split_to_path:
        raise ValueError(
            f"Unsupported split: {split}. "
            f"Expected one of {list(split_to_path)}"
        )

    return split_to_path[split]


def get_instances_annotation_path(split: str) -> Path:
    """
    Return the COCO instance annotation file for a split.
    """
    split_to_path = {
        TRAIN_SPLIT: INSTANCES_TRAIN_JSON,
        DEV_SPLIT: INSTANCES_VAL_JSON,
    }

    if split not in split_to_path:
        raise ValueError(
            f"Instance annotations are not configured for split: {split}"
        )

    return split_to_path[split]


def get_image_path(
    image_id: int,
    split: str,
    file_name: str | None = None,
) -> Path:
    """
    Resolve a COCO image path.

    If file_name is provided, it is used directly.
    Otherwise, the standard COCO filename is constructed.
    """
    image_dir = get_image_dir(split)

    if file_name is not None:
        return image_dir / file_name

    # COCO image filenames follow:
    # COCO_<split>_<12-digit-image-id>.jpg
    return image_dir / f"COCO_{split}_{image_id:012d}.jpg"


def get_experiment_output_dir(experiment_name: str) -> Path:
    """
    Return and create the output directory for an experiment.
    """
    output_dir = OUTPUT_DIR / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def get_experiment_checkpoint_dir(experiment_name: str) -> Path:
    """
    Return and create the checkpoint directory for an experiment.
    """
    checkpoint_dir = CHECKPOINT_DIR / experiment_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir


def get_experiment_prediction_dir(experiment_name: str) -> Path:
    """
    Return and create the prediction directory for an experiment.
    """
    prediction_dir = PREDICTIONS_DIR / experiment_name
    prediction_dir.mkdir(parents=True, exist_ok=True)
    return prediction_dir


# =============================================================================
# 19. CONFIGURATION VALIDATION
# =============================================================================

def validate_config(verbose: bool = True) -> dict:
    """
    Validate the Major project configuration.

    This function checks configuration consistency and reports filesystem
    availability. It does not download models or datasets.

    Returns:
        Dictionary containing validation results.
    """

    checks = {}

    # -------------------------------------------------------------------------
    # Basic project paths
    # -------------------------------------------------------------------------

    checks["project_root"] = PROJECT_ROOT.exists()
    checks["data_dir"] = DATA_DIR.exists()
    checks["experiments_dir"] = EXPERIMENTS_DIR.exists()

    # -------------------------------------------------------------------------
    # Dataset paths
    # -------------------------------------------------------------------------

    checks["captions_train_json"] = CAPTIONS_TRAIN_JSON.exists()
    checks["captions_val_json"] = CAPTIONS_VAL_JSON.exists()

    checks["instances_train_json"] = INSTANCES_TRAIN_JSON.exists()
    checks["instances_val_json"] = INSTANCES_VAL_JSON.exists()

    checks["train_images_dir"] = TRAIN_IMAGES_DIR.exists()
    checks["val_images_dir"] = VAL_IMAGES_DIR.exists()

    # -------------------------------------------------------------------------
    # Configuration consistency
    # -------------------------------------------------------------------------

    checks["blip_model_configured"] = bool(BLIP_MODEL_NAME)
    checks["openclip_model_configured"] = bool(OPENCLIP_MODEL_NAME)
    checks["openclip_pretrained_configured"] = bool(OPENCLIP_PRETRAINED)

    checks["seed_valid"] = isinstance(SEED, int)

    checks["consistency_weight_valid"] = (
        CONSISTENCY_LOSS_WEIGHT >= 0.0
    )

    checks["temperature_valid"] = (
        CONTRASTIVE_TEMPERATURE > 0.0
    )

    checks["confidence_threshold_valid"] = (
        0.0 <= REGION_CONFIDENCE_THRESHOLD <= 1.0
    )

    checks["uniqueness_threshold_valid"] = (
        0.0 <= REGION_UNIQUENESS_THRESHOLD <= 1.0
    )

    checks["lora_rank_valid"] = LORA_RANK > 0

    checks["batch_size_valid"] = BATCH_SIZE > 0

    checks["learning_rate_valid"] = LEARNING_RATE > 0.0

    # -------------------------------------------------------------------------
    # Important architecture checks
    # -------------------------------------------------------------------------

    checks["teacher_forcing_enabled"] = USE_TEACHER_FORCING

    checks["static_embeddings_disabled"] = (
        USE_STATIC_WORD_EMBEDDINGS is False
    )

    checks["openclip_frozen"] = (
        OPENCLIP_FROZEN is True
    )

    # -------------------------------------------------------------------------
    # Create project output directories
    # -------------------------------------------------------------------------

    for directory in [
        OUTPUT_DIR,
        CHECKPOINT_DIR,
        LOG_DIR,
        PREDICTIONS_DIR,
        METRICS_DIR,
        FIGURES_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Verbose output
    # -------------------------------------------------------------------------

    if verbose:
        print("=" * 70)
        print("MAJOR PROJECT CONFIGURATION VALIDATION")
        print("=" * 70)

        print(f"\nProject root:")
        print(f"  {PROJECT_ROOT}")

        print("\nDataset:")
        print(f"  Name:       {DATASET_NAME}")
        print(f"  Train:      {TRAIN_SPLIT}")
        print(f"  Development:{DEV_SPLIT}")
        print(f"  Test:       {TEST_SPLIT}")

        print("\nBLIP:")
        print(f"  Model:      {BLIP_MODEL_NAME}")

        print("\nOpenCLIP:")
        print(f"  Model:      {OPENCLIP_MODEL_NAME}")
        print(f"  Pretrained: {OPENCLIP_PRETRAINED}")
        print(f"  Frozen:     {OPENCLIP_FROZEN}")

        print("\nConsistency objective:")
        print(f"  Enabled:    {CONSISTENCY_LOSS_ENABLED}")
        print(f"  Lambda:     {CONSISTENCY_LOSS_WEIGHT}")
        print(f"  Temperature:{CONTRASTIVE_TEMPERATURE}")
        print(f"  Hard negs:  {USE_SAME_IMAGE_HARD_NEGATIVES}")

        print("\nLoRA:")
        print(f"  Enabled:    {LORA_ENABLED}")
        print(f"  Rank:       {LORA_RANK}")
        print(f"  Alpha:      {LORA_ALPHA}")
        print(f"  Dropout:    {LORA_DROPOUT}")

        print("\nPath checks:")

        for name, status in checks.items():
            icon = "OK" if status else "MISSING/INVALID"
            print(f"  [{icon}] {name}")

        print("\n" + "=" * 70)

    return checks


# =============================================================================
# 20. MAIN
# =============================================================================

if __name__ == "__main__":
    validate_config(verbose=True)