# Enhancing the Multi-Turn Teaching Domain of Large Language Models using Theory of Mind

This repository contains the experimental code for the Master's thesis:

> **Enhancing the Multi-Turn Teaching Domain of Large Language Models using Theory of Mind**  
> Tomás dos Santos Pedrosa  
> Instituto Superior Técnico, Universidade de Lisboa, 2026

The project studies **multi-turn model-to-model teaching** between a stronger teacher LLM and a weaker student LLM. It builds on the teacher-student framework of Saha et al. and evaluates whether Theory-of-Mind-inspired adaptation improves student performance over several rounds of explanation-based teaching.

The code implements the baseline multi-turn teaching regimes and the three strategies evaluated in the thesis:

- **NE — No Explanations Baseline**: selected training examples are added to the student context without rationales.
- **SE — Student Explanations Baseline**: selected examples are explained by the student itself.
- **MTTR — Multi-Turn Teacher Rationales**: selected examples are explained by the teacher.
- **FES — Fixed Example Scaling**: varies the number of teacher-explained examples added per round.
- **DMMU — Dynamic Mental Model Updates**: updates the teacher's mental model of the student after each round.
- **EDPE — Error-Driven Prioritization of Explanations**: biases example selection toward the student's recurring error principles.

All final thesis experiments are inference-only: the student model is not fine-tuned. Improvements come from the multi-turn accumulation and selection of in-context demonstrations.

---

## Repository structure

```text
.
├── data/
│   └── datasets/
│       └── strategyqa/
│           ├── train.json
│           ├── validation.json
│           ├── test.json
│           └── harmful_explanations.json
├── scripts/
│   └── run_multiturn_experiments_slurm.sh
├── src/
│   ├── multiturn_mm_experiments.py
│   ├── machine_teaching/
│   │   ├── error_prior/
│   │   │   ├── monitor.py
│   │   │   ├── principles.py
│   │   │   └── utility.py
│   │   └── models/
│   │       ├── model.py
│   │       └── vllm/
│   │           ├── model_vllm.py
│   │           ├── student_model_vllm.py
│   │           ├── teacher_model_vllm.py
│   │           ├── teacher_mental_model_vllm.py
│   │           ├── teacher_static_mental_model_vllm.py
│   │           └── teacher_dynamic_mental_model_vllm.py
│   └── utilities/
│       ├── dataset_tasks_utils.py
│       └── prompts.py
└── README.md
```

Important files:

| File | Purpose |
|---|---|
| `src/multiturn_mm_experiments.py` | Main entry point for all final thesis experiments. Implements the multi-turn loop, NE/SE/MTTR, FES, DMMU, EDPE, evaluation, and runtime logging. |
| `src/utilities/dataset_tasks_utils.py` | Loads StrategyQA into the internal dataframe format used by the experiments. |
| `src/utilities/prompts.py` | Centralizes student, teacher, mental-model, and error-principle prompts. |
| `src/machine_teaching/models/vllm/student_model_vllm.py` | Student model wrapper using vLLM/OpenAI-compatible endpoints. |
| `src/machine_teaching/models/vllm/teacher_model_vllm.py` | Teacher model wrapper; also includes error-principle classification for EDPE. |
| `src/machine_teaching/models/vllm/teacher_static_mental_model_vllm.py` | Static mental model used by the baseline MTTR selection policy. |
| `src/machine_teaching/models/vllm/teacher_dynamic_mental_model_vllm.py` | Dynamic mental model used by DMMU. |
| `src/machine_teaching/error_prior/principles.py` | Error-principle taxonomy used by EDPE. |
| `src/machine_teaching/error_prior/monitor.py` | Counts observed error principles and converts frequencies into additive EDPE coefficients. |
| `src/machine_teaching/error_prior/utility.py` | Adds the EDPE coefficient to the base expected-utility score. |
| `scripts/run_multiturn_experiments_slurm.sh` | Optional SLURM wrapper for launching teacher/student vLLM servers and running experiments. |

---

## Experimental setting reproduced by this repository

The final thesis experiments use:

- Dataset: **StrategyQA**
- Candidate teaching pool: StrategyQA training split
- Evaluation set: fixed 200-sample subset of the StrategyQA validation split
- Teaching rounds: `R = 5`
- Default examples per round: `k = 2`
- FES budgets: `k ∈ {1, 2, 3, 4, 5}`
- Student context: cumulative across rounds
- Decoding: deterministic, `temperature = 0`
- Backend: vLLM, exposed through OpenAI-compatible endpoints
- Metrics: round-wise accuracy and wall-clock runtime

The model pairs used in the thesis follow the form:

```text
teacher model  →  student model
```

with matched model families and larger teachers than students. Replace the model identifiers in the commands below with the exact Hugging Face model IDs used in your run.

---

## Installation

### 1. Create a Python environment

A Python 3.10+ environment is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 2. Install dependencies

For the vLLM-based thesis experiments:

```bash
pip install \
  pandas \
  numpy \
  torch \
  transformers \
  accelerate \
  sentencepiece \
  protobuf \
  openai \
  vllm \
  tqdm
```

Depending on the model family and vLLM version, additional model-specific dependencies may be required. Follow the model card instructions for gated or instruction-tuned models.

### 3. Configure Python path

Run all commands from the repository root and expose `src/` on `PYTHONPATH`:

```bash
export PYTHONPATH="$PWD/src:$PYTHONPATH"
```

### 4. Prepare output directory

```bash
mkdir -p results
```

---

## Data

The StrategyQA files should be available under:

```text
data/datasets/strategyqa/
├── train.json
├── validation.json
├── test.json
└── harmful_explanations.json
```

The final thesis experiments use `train.json` as the candidate teaching pool and `validation.json` as the evaluation source. The script samples a fixed validation subset internally using the repository-level random seed.

---

## Running with vLLM endpoints

The main experiment script expects a teacher endpoint and a student endpoint when using `--remote`.

In one terminal, launch the teacher model:

```bash
export API_KEY="local-vllm-token"
export TEACHER_MODEL="<teacher-model-id>"

CUDA_VISIBLE_DEVICES=0 vllm serve "$TEACHER_MODEL" \
  --host 127.0.0.1 \
  --port 15051 \
  --api-key "$API_KEY" \
  --dtype auto \
  --gpu-memory-utilization 0.90 \
  --max-model-len 4096 \
  --disable-log-requests
```

In another terminal, launch the student model:

```bash
export API_KEY="local-vllm-token"
export STUDENT_MODEL="<student-model-id>"

CUDA_VISIBLE_DEVICES=1 vllm serve "$STUDENT_MODEL" \
  --host 127.0.0.1 \
  --port 15050 \
  --api-key "$API_KEY" \
  --dtype auto \
  --gpu-memory-utilization 0.90 \
  --max-model-len 4096 \
  --disable-log-requests
```

Then run an experiment from the repository root.

---

## Common arguments

Most thesis runs share the following arguments:

```bash
COMMON_ARGS="\
  --llm-lib vllm \
  --remote \
  --use-instruct \
  --use-explanations \
  --use-gold-label \
  --task strategy_qa \
  --data-dir data/datasets/strategyqa \
  --train-filename train.json \
  --test-filename test.json \
  --val-filename validation.json \
  --teacher-model $TEACHER_MODEL \
  --student-model $STUDENT_MODEL \
  --teacher-model-url http://127.0.0.1:15051/v1 \
  --student-model-url http://127.0.0.1:15050/v1 \
  --api-key $API_KEY \
  --n-ic-samples 4 \
  --n-test-samples 200 \
  --n-rounds 5 \
  --temperature 0.0 \
  --n-beams 1 \
  --n-logprobs 5 \
  --mm-type mm_both \
  --intervention-utility mm_both \
  --teacher-explanation-type useful_teacher \
  --student-explanation-type cot"
```

Use the same five seeds for every condition in a model pair. For example:

```bash
SEEDS="40 41 42 43 44"
```

If the thesis used a different exact seed list, replace `SEEDS` with that list before running.

---

## Reproducing baseline regimes

### NE — No Explanations Baseline

The selected examples are inserted without explanations:

```bash
python -u src/multiturn_mm_experiments.py $COMMON_ARGS \
  --k-expl 2 \
  --sample-expl none \
  --seeds $SEEDS \
  --results-path results/ne_k2.json
```

### SE — Student Explanations Baseline

The selected examples are explained by the student model:

```bash
python -u src/multiturn_mm_experiments.py $COMMON_ARGS \
  --k-expl 2 \
  --sample-expl student \
  --seeds $SEEDS \
  --results-path results/se_k2.json
```

### MTTR — Multi-Turn Teacher Rationales

The selected examples are explained by the teacher model:

```bash
python -u src/multiturn_mm_experiments.py $COMMON_ARGS \
  --k-expl 2 \
  --sample-expl teacher \
  --seeds $SEEDS \
  --results-path results/mttr_k2.json
```

---

## Reproducing FES

FES varies the number of newly explained examples added per round:

```bash
for K in 1 2 3 4 5; do
  python -u src/multiturn_mm_experiments.py $COMMON_ARGS \
    --k-expl "$K" \
    --sample-expl teacher \
    --seeds $SEEDS \
    --results-path "results/fes_k${K}.json"
done
```

`K=2` corresponds to the standard MTTR explanation budget.

---

## Reproducing DMMU

DMMU keeps `k = 2` but updates the teacher's mental model after each round:

```bash
python -u src/multiturn_mm_experiments.py $COMMON_ARGS \
  --k-expl 2 \
  --sample-expl teacher \
  --mm-dynamic \
  --seeds $SEEDS \
  --results-path results/dmmu_k2.json
```

---

## Reproducing EDPE

EDPE keeps the static mental model by default and adds the error-principle prior to the expected-utility score:

```bash
python -u src/multiturn_mm_experiments.py $COMMON_ARGS \
  --k-expl 2 \
  --sample-expl teacher \
  --error-prioritisation \
  --seeds $SEEDS \
  --results-path results/edpe_k2_x03.json
```

The EDPE coefficient range is currently fixed in code as:

```text
βp ∈ [-0.30, +0.30]
```

This is implemented in:

```text
src/machine_teaching/error_prior/monitor.py
```

---

## Optional: EDPE + DMMU

This combination is supported by the code, although the thesis reports EDPE primarily as an error-prioritized selection strategy over the MTTR-style expected-utility computation.

```bash
python -u src/multiturn_mm_experiments.py $COMMON_ARGS \
  --k-expl 2 \
  --sample-expl teacher \
  --mm-dynamic \
  --error-prioritisation \
  --seeds $SEEDS \
  --results-path results/edpe_dmmu_k2_x03.json
```

---

## Output format

Each run writes a JSON file at `--results-path`.

The output contains one entry per seed, with fields such as:

```json
{
  "40": {
    "round_1": 0.61,
    "round_1_seconds": 1234.56,
    "round_2": 0.63,
    "round_2_seconds": 1201.33,
    "round_3": 0.64,
    "round_3_seconds": 1198.45,
    "round_4": 0.65,
    "round_4_seconds": 1210.78,
    "round_5": 0.66,
    "round_5_seconds": 1222.90,
    "total_seconds": 6068.02
  }
}
```

Accuracy is stored as a fraction, e.g. `0.66` means `66%`.

---

## Suggested full experiment grid

For each teacher-student pair:

```bash
# Baselines
python -u src/multiturn_mm_experiments.py $COMMON_ARGS --k-expl 2 --sample-expl none    --seeds $SEEDS --results-path results/ne_k2.json
python -u src/multiturn_mm_experiments.py $COMMON_ARGS --k-expl 2 --sample-expl student --seeds $SEEDS --results-path results/se_k2.json
python -u src/multiturn_mm_experiments.py $COMMON_ARGS --k-expl 2 --sample-expl teacher --seeds $SEEDS --results-path results/mttr_k2.json

# FES
for K in 1 2 3 4 5; do
  python -u src/multiturn_mm_experiments.py $COMMON_ARGS \
    --k-expl "$K" \
    --sample-expl teacher \
    --seeds $SEEDS \
    --results-path "results/fes_k${K}.json"
done

# DMMU
python -u src/multiturn_mm_experiments.py $COMMON_ARGS \
  --k-expl 2 \
  --sample-expl teacher \
  --mm-dynamic \
  --seeds $SEEDS \
  --results-path results/dmmu_k2.json

# EDPE
python -u src/multiturn_mm_experiments.py $COMMON_ARGS \
  --k-expl 2 \
  --sample-expl teacher \
  --error-prioritisation \
  --seeds $SEEDS \
  --results-path results/edpe_k2_x03.json
```

---

## SLURM usage

The repository includes an optional SLURM wrapper:

```bash
scripts/run_multiturn_experiments_slurm.sh
```

Example:

```bash
sbatch scripts/run_multiturn_experiments_slurm.sh \
  -d strategy_qa \
  -t "<teacher-model-id>" \
  -s "<student-model-id>" \
  -u mm_both \
  --mm mm_both \
  --rounds 5 \
  --expl 2 \
  --samples 200 \
  --se useful_teacher \
  --te useful_teacher \
  --ics 4 \
  --lib vllm \
  --remote \
  --sample-expl teacher \
  --seed 40 \
  --seed 41 \
  --seed 42 \
  --seed 43 \
  --seed 44 \
  --results mttr_k2.json
```

Before using the SLURM script on a new cluster, update:

- partition and QoS settings;
- GPU and memory settings;
- conda environment name;
- cache directory;
- data directory;
- vLLM port allocation;
- API key handling.

Do not commit private cluster paths, personal emails, or real API tokens.

---

## Reproducibility notes

1. The validation subset is sampled deterministically inside the main script using the repository-level random seed.
2. Use the same seed list across all strategies for fair comparison.
3. Use the same teacher/student model IDs across NE, SE, MTTR, FES, DMMU, and EDPE.
4. Use deterministic decoding with `temperature = 0.0`.
5. Runtime depends on GPU type, vLLM version, server load, and tensor parallelism. For thesis-level comparison, compare runs executed under the same hardware and serving configuration.
6. EDPE is significantly more expensive because it performs a diagnostic pass over the training pool to assign error principles before selection.

---

## Acknowledgment

This work builds on the multi-turn teacher-student framework introduced by:

```bibtex
@article{saha2023can,
  title={Can language models teach weaker agents? Teacher explanations improve students via theory of mind},
  author={Saha, Swarnadeep and Hase, Peter and Bansal, Mohit},
  journal={arXiv preprint arXiv:2306.09299},
  year={2023}
}
```
