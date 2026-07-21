# SkillSight

**SkillSight** is a skill retrieval method based on generic-subspace debiasing and discriminativeness-weighted lexical-dense fusion.

## 📋 Project Structure

```
SkillSight/
├── __init__.py              # Package initialization
├── constants.py             # Constants (datasets, stopwords, hyperparameters)
├── cli.py                   # Command-line interface
├── encoder.py               # Dense encoder wrapper (SentenceTransformer-based)
├── data.py                  # Data loading (SRA-Bench and SkillRouter formats)
├── cache.py                 # Embedding cache management
├── text_model.py            # Text model (generic token identification, IDF/DF statistics)
├── retriever.py             # SkillSight core retriever
├── fusion.py                # Lexical-dense score fusion
├── metrics.py               # Evaluation metrics (Recall@K, Hit@K, MRR@K)
├── utils.py                 # Utility functions (L2 normalization, hashing, Otsu threshold)
├── evaluation.py            # Ranking evaluation pipeline (Recall/Hit/MRR)
├── eval_skillsight_e2e.py   # End-to-end task-accuracy pipeline (retrieve→infer→evaluate)
├── vllm.sh                  # Launch a vLLM server for the end-to-end eval
├── download_datasets.py     # Dataset download script
├── requirements.txt         # Core dependencies
└── README.md                # This file
```

## 🔧 Setup

### 1. Create Conda Environment

```bash
conda create -n skillsight python=3.10
conda activate skillsight
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**Core dependencies:**
- `torch==2.6.0` - PyTorch deep learning framework
- `sentence-transformers==5.5.1` - Sentence encoder
- `numpy==2.4.6` - Numerical computation
- `huggingface-hub==1.16.1` - Dataset and model downloads
- `tqdm==4.67.1` - Progress bar display

### 3. Download Embedding Model

SkillSight uses **Qwen3-Embedding-0.6B** as the default embedding model.

#### Option A: Download from Hugging Face (Recommended)

```bash
hf download Qwen/Qwen3-Embedding-0.6B \
  --local-dir ./models/Qwen3-Embedding-0.6B
```

Or using Python:

```python
from huggingface_hub import snapshot_download

model_path = snapshot_download(
    repo_id="Qwen/Qwen3-Embedding-0.6B",
    repo_type="model",
    local_dir="./models/Qwen3-Embedding-0.6B",
)

```

#### Option B: Use Custom Path

If you already have the model downloaded elsewhere, you can specify the path using `--model-path`:

```bash
python -m cli --model-path /path/to/your/embedding/model
```

**Note**: The model should be compatible with `sentence-transformers`. Recommended models:
- Qwen3-Embedding-0.6B
- Any SentenceTransformer-compatible embedding model

## 📦 Download Datasets

SkillSight supports two benchmark datasets:

### Method 1: Use Download Script (Recommended)

```bash
python download_datasets.py
```

This will automatically download:
- **SkillRouter-Eval-Core** (~422 MB): 87 routing tasks with Easy/Hard candidate pools
- **SRA-Bench** (~242 MB): 5,400 test instances with 26,262 skill corpus

After downloading, the directory structure will be:
```
SkillSight/
├── SkillRouter-Eval-Core/
│   ├── easy/                # 78,361 skills
│   ├── hard/                # 79,141 skills
│   ├── manifest.json
│   ├── relevance.json
│   └── tasks.jsonl
└── SRA-Bench/
    ├── corpus/
    │   └── corpus.json      # Complete skill corpus
    └── instances/           # Test sets for 6 task families
        ├── theoremqa.json
        ├── logicbench.json
        ├── toolqa.json
        ├── medcalcbench.json
        ├── champ.json
        └── bigcodebench.json
```

### Method 2: Manual Download

```bash
# Install Hugging Face CLI
pip install -U huggingface-hub

# Download SkillRouter
hf download pipizhao/SkillRouter-Eval-Core \
  --repo-type dataset \
  --local-dir ./SkillRouter-Eval-Core

# Download SRA-Bench
hf download WeihangSu/SRA-Bench \
  --repo-type dataset \
  --local-dir ./SRA-Bench
```

## 🚀 Running SkillSight

### Quick Start

**Evaluate on SRA-Bench:**

```bash
python -m cli \
  --benchmark sra \
  --model-path ./models/Qwen3-Embedding-0.6B \
  --device cuda:0
```

**Evaluate on SkillRouter:**

```bash
python -m cli \
  --benchmark skillrouter \
  --model-path ./models/Qwen3-Embedding-0.6B \
  --device cuda:0
```

### Main Parameters

#### Dataset and Paths
- `--benchmark {sra,skillrouter}` - Choose benchmark dataset (default: `sra`)
- `--repo-root PATH` - SRA-Bench data path (default: `./SRA-Bench`)
- `--skillrouter-root PATH` - SkillRouter data path (default: `./SkillRouter-Eval-Core`)
- `--datasets [NAMES...]` - SRA-Bench sub-datasets (default: `theoremqa logicbench toolqa medcalcbench champ bigcodebench`)

#### Model Configuration
- `--model-path PATH` - Embedding model path (default: `./models/Qwen3-Embedding-0.6B`)
- `--device DEVICE` - Computing device (default: `cuda:0`)
- `--max-seq-length INT` - Maximum sequence length (default: `32768`)

#### Retrieval Parameters
- `--top-k [K...]` - K values for evaluation (default: `1 3 5 10 20 50`)
- `--primary-k INT` - Primary K value for reporting (default: `10`)
- `--candidate-k INT` - Candidate depth (default: `300`)

#### SkillSight Method Parameters
- `--max-tokens-per-doc INT` - Maximum tokens per document (default: `200`)
- `--lexical-beta FLOAT` - Lexical discriminativeness suppression strength (default: `1.0`)
  - `beta=0`: Plain IDF lexical evidence
  - `beta=1.0`: Full discriminativeness weighting

#### Cache and Output
- `--cache-dir PATH` - Embedding cache directory (default: `./cache`)
- `--cache-only` - Use cache only, no re-encoding
- `--output-dir PATH` - Output directory (default: `./results`)
- `--output-json PATH` - Specify output JSON file path

#### Batch Sizes
- `--doc-batch INT` - Document encoding batch size (default: `4`)
- `--query-batch INT` - Query encoding batch size (default: `64`)
- `--score-batch INT` - Scoring batch size (default: `64`)

### Complete Examples

```bash
# Full evaluation on SRA-Bench (all datasets)
python -m cli \
  --benchmark sra \
  --model-path ./models/Qwen3-Embedding-0.6B \
  --device cuda:0 \
  --datasets theoremqa logicbench toolqa medcalcbench champ bigcodebench \
  --top-k 1 3 5 10 20 50 \
  --candidate-k 300 \
  --cache-dir ./cache \
  --output-dir ./results

# Evaluate single dataset on SRA-Bench
python -m cli \
  --benchmark sra \
  --datasets theoremqa \
  --model-path ./models/Qwen3-Embedding-0.6B \
  --device cuda:0

# SkillRouter evaluation (including excluded tasks)
python -m cli \
  --benchmark skillrouter \
  --include-excluded \
  --model-path ./models/Qwen3-Embedding-0.6B \
  --device cuda:0 \
  --output-dir ./results/skillrouter

# Use existing cache (no re-encoding)
python -m cli \
  --benchmark sra \
  --cache-only \
  --cache-dir ./cache \
  --output-dir ./results
```

## 🎯 End-to-End Evaluation

The commands above only score the *ranking* (Recall/Hit/MRR). To measure **task accuracy** of an agent handed the top-K skills SkillSight retrieves, use the end-to-end pipeline. It runs three stages against SR-Agents: **retrieve → infer → evaluate**.

This requires two extra pieces: a **separate vLLM environment** to serve the task LLM, and the **SR-Agents corpus** installed under `third_party/SR-Agents-main`.

### 1. Serve the task model with vLLM

Keep vLLM in its own conda environment (its dependencies conflict with the retrieval env). The encoder and the LLM run on **different GPUs**.

```bash
# one-time: create a dedicated vLLM environment
conda create -n vllm python=3.10
conda activate vllm
pip install vllm

# launch the OpenAI-compatible server (idempotent; re-run is a no-op if already up)
VLLM_GPU=4 \
VLLM_MODEL_PATH=/path/to/Qwen3-4B-Instruct-2507 \
VLLM_SERVED_NAME=Qwen3-4B-Instruct-2507 \
VLLM_PORT=8001 \
bash vllm.sh
```

The server listens on `http://127.0.0.1:8001/v1`. Watch startup with `tail -f logs/vllm_8001.log`.

### 2. Install the SR-Agents corpus

The end-to-end pipeline reads skills from the SR-Agents checkout. Clone it into `third_party/SR-Agents-main`, then install it in **its own** virtualenv (Python 3.10–3.12):

```bash
cd third_party/SR-Agents-main

sudo apt-get install -y python3-venv unzip     # base tools
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip && pip install -e .
```

**Internal skill corpus (required).** SR-Agents ships the corpus zipped; unzip it once (the eval script also auto-extracts it on first use):

```bash
unzip -o data/bench/corpus/corpus.json.zip -d data/bench/corpus/
test -f data/bench/corpus/corpus.json && echo "corpus ready"
```

**External ToolQA corpus (only for ToolQA).** TheoremQA, LogicBench, CHAMP, MedCalc-Bench, and BigCodeBench do **not** need it. If you evaluate ToolQA, download the preprocessed *External Corpus* (not raw data) from [Google Drive](https://drive.google.com/file/d/1zRbHzPW2x4dDcfmphBWlan8cxUCRNmqk/view?usp=drive_link) and place it so that `agenda/`, `scirex/`, `flights/`, etc. sit **directly** under `data/external/toolqa/`:

```bash
pip install --upgrade gdown
gdown --fuzzy 'https://drive.google.com/file/d/1zRbHzPW2x4dDcfmphBWlan8cxUCRNmqk/view?usp=drive_link' -O /tmp/toolqa.zip
unzip -q /tmp/toolqa.zip -d /tmp/toolqa_unpack

# copy the directory that directly contains agenda/ (avoid an extra nesting level)
SRC="$(dirname "$(dirname "$(find /tmp/toolqa_unpack -path '*/agenda/agenda_descriptions_merged.jsonl' -print -quit)")")"
rm -rf data/external/toolqa && mkdir -p data/external/toolqa
cp -a "$SRC"/. data/external/toolqa/
```

The result must be `data/external/toolqa/agenda/…`, **not** `.../toolqa/external_corpus/agenda/…`. Ubuntu is case-sensitive, so file names like `Airbnb_Open_Data.csv` must match exactly. Note: SR-Agents loads the DBLP `.pkl` files via `pickle.load` — only use the official ToolQA files, never untrusted pickles.

### 3. Run the pipeline

Point the eval at your served model and encoder. The LLM must match `--served-model-name` from step 1, and `--device` should be a **different GPU** than vLLM's:

```bash
python -m SkillSight.eval_skillsight_e2e \
  --model Qwen3-4B-Instruct-2507 \
  --api-base http://127.0.0.1:8001/v1 \
  --embed-model-path ./models/Qwen3-Embedding-0.6B \
  --device cuda:0 \
  --top-ks 3 \
  --no-start-vllm \
  --overwrite
```

Key flags: `--datasets` selects task families (default: all six), `--top-ks` sets how many retrieved skills to inject, `--no-start-vllm` assumes the server is already up (drop it to let the script launch `vllm.sh`), and `--reuse-retrieval` skips re-retrieval. Results land in `results/skillsight_e2e/` (`summary_skillsight_e2e.csv` and `per_sample_skillsight_e2e.csv`).

## 📊 Output

After running, the following will be generated:

### 1. Console Output
- Recall@K, Hit@K, MRR@K metrics for each dataset
- Overall micro-averaged metrics
- Latency statistics

### 2. JSON Results File
Default path: `./results/performance_metrics.json`

```json
{
  "config": { ... },           // Run configuration
  "method": {                  // Method description
    "name": "SkillSight",
    ...
  },
  "per_category": {            // Detailed metrics per dataset
    "theoremqa": { ... },
    ...
  },
  "overall": { ... },          // Overall metrics
  "summary_tables": { ... }    // Summary tables
}
```

### 3. Cache Files
- `./cache/*.npy` - Embedding cache files
- `./cache/*.meta.json` - Cache metadata
- `./results/cache/*.npy` - Text model cache

## 📈 Evaluation Metrics

- **Recall@K**: Fraction of gold skills retrieved in top-K results
- **Hit@K**: Proportion of queries with at least one gold skill in top-K
- **MRR@K**: Mean reciprocal rank of the first gold skill within top-K

## 💡 FAQ

### Q1: CUDA out of memory error?
Reduce batch sizes:
```bash
python -m cli --doc-batch 2 --query-batch 32 --score-batch 32
```

### Q2: How to use CPU?
```bash
python -m cli --device cpu
```

### Q3: How to use my own embedding model?
Ensure your model is compatible with `sentence-transformers`, then:
```bash
python -m cli --model-path /path/to/your/model
```

### Q4: Cache files too large?
Caching significantly speeds up repeated experiments. To clean up:
```bash
rm -rf ./cache/*
rm -rf ./results/cache/*
```

### Q5: Wrong number of SkillRouter tasks?
- By default, `generic_only` tasks are excluded (75 tasks)
- Add `--include-generic-only` to get all 87 tasks
- Add `--include-excluded` to include excluded tasks
```

## 📚 Citation

If you use SkillSight, please cite the relevant papers.

**Dataset Citations:**

- **SkillRouter**: [SkillRouter: Retrieve-and-Rerank Skill Selection for LLM Agents at Scale](https://github.com/zhengyanzhao1997/SkillRouter)
- **SRA-Bench**: [Skill Retrieval Augmentation (SRA-Bench)](https://github.com/oneal2000/SR-Agents)

## 📄 License

Please refer to the licenses of the original papers and datasets.

## 🤝 Contributing

Issues and Pull Requests are welcome!

---

**Technical Support**: If you encounter issues, please check:
1. Python version >= 3.8
2. PyTorch and CUDA versions are compatible
3. Dataset paths are correct
4. Sufficient disk space (at least 2 GB)
5. Embedding model is downloaded and accessible
