"""Constants for SkillSight: standalone skill retrieval method."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT_DEFAULT = Path("./SRA-Bench")
DEFAULT_MODEL_PATH = "./models/Qwen3-Embedding-0.6B"
DEFAULT_MAX_SEQ_LENGTH = 32768
DEFAULT_DATASETS = [
    "theoremqa",
    "logicbench",
    "toolqa",
    "medcalcbench",
    "champ",
    "bigcodebench",
]
DEFAULT_TOP_K = [1, 3, 5, 10, 20, 50]
SKILLROUTER_TIERS = ("easy", "hard")

# Encoder-specific defaults follow each checkpoint's published inference setup.
ENCODER_PRESETS = {
    "qwen3": {
        "model_path": "models/Qwen3-Embedding-0.6B",
        "max_seq_length": 32768,
        "doc_batch": 4,
        "query_batch": 64,
        "score_batch": 64,
    },
    "skillrouter": {
        "model_path": "models/SkillRouter-Embedding-0.6B",
        "max_seq_length": 4096,
        "doc_batch": 16,
        "query_batch": 32,
        "score_batch": 32,
    },
    "skillret": {
        "model_path": "models/SKILLRET-Embedding-0.6B",
        "max_seq_length": 8192,
        "doc_batch": 16,
        "query_batch": 32,
        "score_batch": 32,
    },
}
DEFAULT_ENCODER_PRESET = "qwen3"

# Structural constants used by SkillSight's corpus-level text model.
# Default token cap per document; overridable via the --max-tokens-per-doc CLI flag.
MAX_TOKENS_PER_DOC = 200
MIN_GENERIC_DF = 2
TEXT_MODEL_VERSION = "skillsight-v1"

# SkillSight lexical channel: discriminativeness-weighted token evidence.
# Each in-corpus query token t is weighted by
#     w(t) = IDF(t) * (1 - DF(t)/N) ** LEXICAL_BETA
# where (1 - DF(t)/N) is the probability that matching t separates the gold
# document from a random distractor (its discriminative yield). This applies the
# same "suppress corpus-level common background" principle as the dense channel's
# generic-subspace projection, now in the discrete token space. LEXICAL_BETA
# controls the suppression sharpness; 0 recovers the plain-IDF lexical evidence.
LEXICAL_BETA = 1.0

# Retrieval variants used by the reproduction and ablation experiments.
VARIANT_DENSE = "dense"
VARIANT_SBC = "sbc"
VARIANT_LEC = "lec"
VARIANT_DENSE_LEC = "dense_lec"
VARIANT_SKILLSIGHT = "skillsight"
VARIANT_SKILLSIGHT_EQ15 = "skillsight_eq15"

ALL_VARIANTS = (
    VARIANT_DENSE,
    VARIANT_SBC,
    VARIANT_LEC,
    VARIANT_DENSE_LEC,
    VARIANT_SKILLSIGHT,
    VARIANT_SKILLSIGHT_EQ15,
)
DEFAULT_VARIANTS = (VARIANT_SKILLSIGHT,)

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "when", "while",
    "of", "to", "in", "on", "for", "from", "by", "with", "without", "as", "at",
    "is", "are", "was", "were", "be", "been", "being", "am", "do", "does", "did",
    "this", "that", "these", "those", "it", "its", "into", "about", "over", "under",
    "can", "could", "would", "should", "may", "might", "will", "shall", "must",
    "you", "your", "we", "our", "they", "their", "he", "she", "his", "her", "them",
    "i", "me", "my", "mine", "there", "here", "which", "who", "whom", "whose",
    "what", "why", "how", "where", "also", "using", "use", "used", "uses",
    "not", "all", "skill", "user", "check", "create", "code", "patterns", "name",
    "data", "type", "new", "specific", "before", "add", "only", "key", "example",
    "bash", "output", "file", "api", "reference", "run",
}
