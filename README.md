# SkillSight

Official implementation of **SkillSight**, a training-free calibration method for skill retrieval.

SkillSight can be applied on top of an existing dense retriever without retriever fine-tuning or an additional reranking model. This README focuses on running the code and reproducing the main results; please refer to the paper for the full motivation and method.

## Quick Start

The commands below assume that the repository is cloned as `SkillSight/` and that retrieval commands are run from the repository root.

### 1. Create the environment

Python 3.11 and a CUDA-enabled environment are recommended. The pinned
`numpy==2.4.6` release requires Python 3.11 or newer.

```bash
conda create -n skillsight python=3.11 -y
conda activate skillsight
pip install -r requirements.txt
```

### 2. Download the embedding model

The paper uses [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B).

```bash
mkdir -p models
hf download Qwen/Qwen3-Embedding-0.6B \
  --local-dir ./models/Qwen3-Embedding-0.6B
```

Any local SentenceTransformer-compatible model can also be supplied through `--model-path`.

### 3. Download the datasets

```bash
python download_datasets.py
```

This downloads the two retrieval benchmarks used in the paper:

| Benchmark | Corpus scale | Expected directory |
| --- | ---: | --- |
| SRA-Bench | 5,400 queries; approximately 26k skills | `./SRA-Bench` |
| SkillBench-Supp | approximately 77k skills | `./SkillRouter-Eval-Core` |

If automatic downloading is unavailable, run:

```bash
hf download WeihangSu/SRA-Bench \
  --repo-type dataset \
  --local-dir ./SRA-Bench

hf download pipizhao/SkillRouter-Eval-Core \
  --repo-type dataset \
  --local-dir ./SkillRouter-Eval-Core
```

### 4. Run retrieval evaluation

SRA-Bench:

```bash
python -m cli \
  --benchmark sra \
  --model-path ./models/Qwen3-Embedding-0.6B \
  --device cuda:0
```

SkillBench-Supp uses `skillrouter` as its internal benchmark identifier:

```bash
python -m cli \
  --benchmark skillrouter \
  --model-path ./models/Qwen3-Embedding-0.6B \
  --device cuda:0
```

The default retrieval results are written to:

```text
results/performance_metrics.json
```

Embeddings and corpus statistics are cached under `cache/` and `results/cache/`. Repeated runs can reuse these files.

## Reproducing the Paper Configuration

The main experiments use a candidate pool of 300, lexical coefficient `beta=1`, and evaluation cutoffs of 1, 3, 5, and 10:

```bash
python -m cli \
  --benchmark sra \
  --top-k 1 3 5 10 \
  --candidate-k 300 \
  --lexical-beta 1.0 \
  --model-path ./models/Qwen3-Embedding-0.6B \
  --device cuda:0 \
  --output-dir ./results
```

To evaluate selected SRA-Bench task families:

```bash
python -m cli \
  --benchmark sra \
  --datasets theoremqa logicbench toolqa \
  --top-k 1 3 5 10 \
  --model-path ./models/Qwen3-Embedding-0.6B \
  --device cuda:0
```

To evaluate all available SkillBench-Supp groups, including generic-only and normally excluded tasks:

```bash
python -m cli \
  --benchmark skillrouter \
  --include-generic-only \
  --include-excluded \
  --model-path ./models/Qwen3-Embedding-0.6B \
  --device cuda:0
```

To reuse cached embeddings:

```bash
python -m cli \
  --benchmark sra \
  --cache-only \
  --cache-dir ./cache \
  --output-dir ./results
```

### Main arguments

| Argument | Description | Default |
| --- | --- | --- |
| `--benchmark` | `sra` or `skillrouter` | `sra` |
| `--encoder-preset` | Published model/path, sequence length, and batch defaults | `qwen3` |
| `--model-path` | Local embedding model | `./models/Qwen3-Embedding-0.6B` |
| `--device` | Encoding and scoring device | `cuda:0` |
| `--datasets` | SRA-Bench task families | all six families |
| `--skillrouter-tiers` | SkillRouter candidate tiers | `easy hard` |
| `--top-k` | Evaluation cutoffs | `1 3 5 10 20 50` |
| `--candidate-k` | Candidate depth before calibration | `300` |
| `--lexical-beta` | Lexical calibration strength | `1.0` |
| `--variants` | Retrieval variants evaluated together | `skillsight` |
| `--max-tokens-per-doc` | Maximum tokens per skill document | `200` |
| `--doc-batch` | Document encoding batch size | `4` |
| `--query-batch` | Query encoding batch size | `64` |
| `--score-batch` | Scoring batch size | `64` |
| `--cache-dir` | Embedding cache directory | `./cache` |
| `--output-dir` | Evaluation output directory | `./results` |

Run `python -m cli --help` for the complete interface.

### Retrieval Ablations

The reproduction fork can evaluate multiple retrieval variants in one pass over
the cached corpus embeddings:

```bash
python -m SkillSight.cli \
  --benchmark sra \
  --repo-root ./third_party/SR-Agents-main \
  --model-path ./models/Qwen3-Embedding-0.6B \
  --variants dense sbc lec dense_lec skillsight skillsight_eq15
```

The variants are defined as follows:

- `dense`: raw dense score `q^T d`.
- `sbc`: background-calibrated dense score `q_perp^T d_perp`.
- `lec`: lexical-only diagnostic ranking over the dense top-K candidates.
- `dense_lec`: dense and lexical local fusion without background projection.
- `skillsight`: the repository's reported SkillSight fusion
  (`skillsight_fusion(..., lam=None)`).
- `skillsight_eq15`: the paper's equation 15 fusion
  (`skillsight_fusion(..., lam=1.0)`).

The default remains `skillsight`, so existing commands reproduce the original
reported behavior. Corpus and query embeddings are reused when their caches are
present, so adding variants does not require re-encoding the corpus.

### Cross-Encoder Evaluation

`DenseEncoder` also supports Transformer-only embedding checkpoints that do not
provide a SentenceTransformer configuration. This is required for the public
`pipizhao/SkillRouter-Embedding-0.6B` checkpoint, which uses last-token pooling,
left padding, BF16 inference, and the query instruction documented by its model
card.

Use the checkpoint presets to apply their published maximum sequence length,
model path, and recommended batch sizes:

```bash
python -m SkillSight.cli \
  --benchmark sra \
  --repo-root ./third_party/SR-Agents-main \
  --encoder-preset skillrouter \
  --variants dense skillsight
```

### Background-Energy Analysis

After caching corpus and query embeddings, reproduce the query/document
background-energy diagnostics without running either encoder again:

```bash
python -m SkillSight.analysis.background_energy \
  --encoder-preset qwen3 \
  --cache-dir ./cache \
  --text-cache-dir ./results/ablations/cache \
  --output-dir ./analysis_output/background_energy_qwen3_sra
```

The analysis writes:

- `summary.json`;
- `query_energy.csv`;
- `document_energy.csv`;
- `figures/background_energy_distribution.png`;
- `figures/background_energy_gap.png`.

```bash
python -m SkillSight.cli \
  --benchmark sra \
  --repo-root ./third_party/SR-Agents-main \
  --encoder-preset skillret \
  --variants dense skillsight
```

## End-to-End Agent Evaluation

The end-to-end experiment retrieves the top three skills, inserts them into the agent context, and evaluates task execution on SRA-Bench. It requires a separate vLLM environment and the SR-Agents evaluation code.

### 1. Start the task model

Use a GPU different from the one used by the embedding model.

```bash
conda create -n vllm python=3.10 -y
conda activate vllm
pip install vllm

VLLM_GPU=4 \
VLLM_MODEL_PATH=/path/to/Qwen3-4B-Instruct-2507 \
VLLM_SERVED_NAME=Qwen3-4B-Instruct-2507 \
VLLM_PORT=8001 \
bash vllm.sh
```

The server should be available at `http://127.0.0.1:8001/v1`.

### 2. Prepare SR-Agents

```bash
git clone https://github.com/oneal2000/SR-Agents.git third_party/SR-Agents-main
cd third_party/SR-Agents-main
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
unzip -o data/bench/corpus/corpus.json.zip -d data/bench/corpus/
cd ../..
```

TheoremQA, LogicBench, CHAMP, MedCalc-Bench, and BigCodeBench use the internal SR-Agents corpus. ToolQA additionally requires its official preprocessed corpus, placed directly under:

```text
third_party/SR-Agents-main/data/external/toolqa/
```

Do not add an extra `external_corpus/` directory level.

### 3. Run the end-to-end pipeline

From the directory containing the `SkillSight/` repository, run:

```bash
python -m SkillSight.eval_skillsight_e2e \
  --model Qwen3-4B-Instruct-2507 \
  --api-base http://127.0.0.1:8001/v1 \
  --embed-model-path ./SkillSight/models/Qwen3-Embedding-0.6B \
  --device cuda:0 \
  --top-ks 3 \
  --no-start-vllm \
  --overwrite
```

Useful options:

| Argument | Description |
| --- | --- |
| `--datasets` | Evaluate selected SRA-Bench task families |
| `--top-ks` | Number of retrieved skills supplied to the agent |
| `--no-start-vllm` | Reuse an already running vLLM server |
| `--reuse-retrieval` | Reuse cached rankings and skip retrieval |

Outputs are saved under `results/skillsight_e2e/`:

```text
summary_skillsight_e2e.csv
per_sample_skillsight_e2e.csv
```

## Representative Results

All retrieval results below use Qwen3-Embedding-0.6B. Latency is measured in milliseconds per query.

| Benchmark | Method | Hit@5 | Recall@10 | MRR@10 | Latency |
| --- | --- | ---: | ---: | ---: | ---: |
| SRA-Bench | Dense | 67.70 | 66.02 | 57.09 | 0.13 |
| SRA-Bench | **SkillSight** | **86.04** | **86.23** | **74.02** | 1.17 |
| SkillBench-Supp | Dense | 75.34 | 56.56 | 66.67 | 0.57 |
| SkillBench-Supp | **SkillSight** | **76.71** | **64.24** | **67.28** | 2.57 |

The following table reports representative end-to-end results on SRA-Bench. Overall is averaged over all instances; latency is measured in seconds, and tokens are the average number of input tokens per instance.

| Agent model | Method | Overall | Latency | Tokens |
| --- | --- | ---: | ---: | ---: |
| Llama-3.1-8B | LLM Selection | 33.83 | 21.65 | 8,525 |
| Llama-3.1-8B | **SkillSight** | **38.80** | 19.91 | 7,703 |
| GPT-5.4-mini | LLM Selection | 67.83 | 17.02 | 10,591 |
| GPT-5.4-mini | **SkillSight** | **69.67** | 6.12 | 9,205 |
| Qwen3-4B | LLM Selection | 56.85 | 26.85 | 8,829 |
| Qwen3-4B | **SkillSight** | **61.13** | 20.51 | 8,234 |

See the paper for the full baseline comparison, per-task results, ablations, and hyperparameter analysis.

## Output Metrics

- **Recall@K:** proportion of gold skills retrieved within the top-k results.
- **Hit@K:** proportion of queries for which at least one gold skill appears in the top-k results.
- **MRR@K:** reciprocal rank of the first gold skill, averaged across queries.

## Repository Layout

```text
SkillSight/
├── cli.py                       # Retrieval evaluation entry point
├── retriever.py                 # SkillSight retrieval pipeline
├── evaluation.py                # Retrieval evaluation
├── eval_skillsight_e2e.py       # End-to-end evaluation
├── download_datasets.py         # Dataset downloader
├── vllm.sh                      # vLLM launcher
├── requirements.txt
└── README.md
```

## Reproducibility Notes

- Use Qwen3-Embedding-0.6B and the dependency versions in `requirements.txt` to reproduce the reported retrieval results.
- Keep the embedding model and vLLM task model on different GPUs during end-to-end evaluation.
- Reduce `--doc-batch`, `--query-batch`, or `--score-batch` if GPU memory is limited.
- CPU execution is available through `--device cpu`, but full-corpus evaluation is substantially slower.
- SkillBench-Supp task counts can change when `--include-generic-only` or `--include-excluded` is enabled.

## Citation

If you find this repository useful, please cite the SkillSight paper. The BibTeX entry will be added when the paper is publicly available.

Please also cite the benchmark repositories used in the experiments:

- [SRA-Bench / SR-Agents](https://github.com/oneal2000/SR-Agents)
- [SkillRouter-Eval-Core](https://github.com/zhengyanzhao1997/SkillRouter)

## License

Please follow the licenses and terms of use of this repository's dependencies, pretrained models, and benchmark datasets.
