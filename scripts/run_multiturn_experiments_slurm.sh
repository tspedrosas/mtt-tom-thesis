#!/bin/bash

#SBATCH --mail-type=BEGIN,END,FAIL         # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=your.email@example.com
#SBATCH --job-name=mtt_tom_experiment
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --time=3-00:00:00
#SBATCH --mem-per-cpu=200G
#SBATCH --qos=gpu
#SBATCH --output="job-%x-%j.out"
#SBATCH --partition=gpu

date;hostname;pwd
options=$(getopt -o d:,s:,t:,u:,b: -l mm-dynamic,error-prioritisation,sample-expl:,mm:,rounds:,expl:,se:,te:,ics:,lib:,key:,seed:,sgpu:,tgpu:,shost:,thost:,sport:,tport:,temp:,lp:,usage:,results:,train-samples:,samples:,remote -- "$@")
# Override these paths when running on a cluster, for example:
#   export CLUSTER_CACHE_DIR=/path/to/model/cache
#   export CLUSTER_DATA_DIR=/path/to/data/root
cache_dir="${CLUSTER_CACHE_DIR:-./cache}"
data_dir="${CLUSTER_DATA_DIR:-./data}"

eval set -- "$options"
seeds=()

while [ $# -gt 0 ]
do
  case $1 in
    -d) dataset=${2}; shift ;;
    -s) student_model=${2}; shift ;;
    -t) teacher_model=${2}; shift ;;
    -u) utility=${2}; shift ;;
    --rounds) n_rounds=${2}; shift ;;
    --expl) k_expl=${2}; shift ;;
    --samples) n_test_samples=${2}; shift ;;
    --seed) seeds+=("${2}"); shift ;;
    --lib) lib=${2}; shift ;;
    --mm) mental_model=${2}; shift ;;
    --mm-dynamic) mm_dynamic=1 ;;
    --error-prioritisation) error_prior=1 ;;
    --sample-expl) sample_expl=${2}; shift ;;
    --se) student_expl=${2}; shift ;;
    --te) teacher_expl=${2}; shift ;;
    --ics) n_ic_samples=${2}; shift ;;
    --remote) remote_model=1 ;;
    --key) api_key=${2}; shift ;;
    --sgpu) n_student_gpus=${2}; shift ;;
    --tgpu) n_teacher_gpus=${2}; shift ;;
    --shost) student_host=${2}; shift ;;
    --thost) teacher_host=${2}; shift ;;
    --sport) student_port=${2}; shift ;;
    --tport) teacher_port=${2}; shift ;;
    --temp) gen_temperature=${2}; shift ;;
    --lp) num_logprobs=${2}; shift ;;
    --usage) gpu_usage=${2}; shift ;;
    --results) results_path="$data_dir"results/${2}; shift ;;
    --train-samples) n_train_samples=${2}; shift ;;
    (--) shift; break ;;
    (-*) echo "$0: error - unrecognized option $1" 1>&2; exit 1 ;;
    (*) break ;;
    esac
    shift
done

if [ ${#seeds[@]} -eq 0 ]; then
  seeds=("1" "2")
fi

if [ -z "$n_rounds" ]; then
  n_rounds=5
fi

if [ -z "$k_expl" ]; then
  k_expl=2
fi

if [ -z "$dataset" ]; then
  dataset="strategy_qa"
fi

if [ -z "$lib" ]; then
  lib="vllm"
fi

if [ -z "$student_model" ]; then
  student_model="google/flan-t5-large"
fi

if [ -z "$teacher_model" ]; then
  teacher_model="google/flan-t5-xl"
fi

if [ -z "$mental_model" ]; then
  mental_model="mm_both"
fi

if [ -z "$utility" ]; then
  utility="mm_both"
fi

if [ -z "$student_expl" ]; then
  student_expl="cot"
fi

if [ -z "$teacher_expl" ]; then
  teacher_expl="useful_teacher"
fi

if [ -z "$n_ic_samples" ]; then
  n_ic_samples=5
fi

if [ -z "$remote_model" ]; then
    remote_model=0
fi

if [ -z "$api_key" ]; then
    api_key="${VLLM_API_KEY:-local-vllm-token}"
fi

if [ -z "$n_student_gpus" ]; then
    n_student_gpus="1"
fi

if [ -z "$n_teacher_gpus" ]; then
    n_teacher_gpus="1"
fi

if [ -z "$student_host" ]; then
    student_host="localhost"
fi

if [ -z "$mm_dynamic" ]; then
    mm_dynamic=0
fi

if [ -z "$error_prior" ]; then
    error_prior=0
fi

if [ -z "$sample_expl" ]; then
    sample_expl="teacher"
fi

if [ -z "$teacher_host" ]; then
    teacher_host="localhost"
fi

if [ -z "$student_port" ]; then
    student_port=15050
fi

if [ -z "$teacher_port" ]; then
    teacher_port=15051
fi

if [ -z "$gen_temperature" ]; then
    gen_temperature=0.0
fi

if [ -z "$num_logprobs" ]; then
    num_logprobs=5
fi

if [ -z "$n_test_samples" ]; then
    n_test_samples=10
fi

n_train_samples_arg=""
if [ -n "$n_train_samples" ]; then
    n_train_samples_arg="--n-train-samples $n_train_samples"
fi

if [ -z "$gpu_usage" ]; then
    gpu_usage=1
fi

if [ -z "$log_level" ]; then
  log_level="INFO"
fi

student_model_url="http://$student_host:$student_port/v1"
teacher_model_url="http://$teacher_host:$teacher_port/v1"

if [ -n "${SLURM_JOB_ID:-}" ] ; then
  IFS=' '
  read -ra newarr <<< "$(scontrol show job "$SLURM_JOB_ID" | awk -F= '/Command=/{print $2}')"
  script_path=$(dirname "${newarr[0]}")
else
  script_path="$( cd -- "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 ; pwd -P )"
fi

# module load python cuda
export LD_LIBRARY_PATH="/opt/cuda/lib64:$LD_LIBRARY_PATH"
export PATH="/opt/cuda/bin:$PATH"
conda_dir="${CONDA_HOME:-${CONDA_PREFIX:-}}"
if [ -z "$conda_dir" ]; then
  echo "ERROR: set CONDA_HOME or activate conda before submitting this job."
  exit 1
fi

conda_env="${CONDA_ENV_NAME:-mtt_tom}"
source "$conda_dir"/bin/activate "$conda_env"

cd "$script_path" || exit
cd .. || exit

if [ "$dataset" = "ec_qa" ]; then
  dataset_dir="datasets/ecqa"
  train_file="data_train.csv"
  test_file="data_test.csv"
  val_file="data_val.csv"
elif [ "$dataset" = "gsm8k" ]; then
  dataset_dir="datasets/gsm8k"
  train_file="train.jsonl"
  test_file="test.jsonl"
  val_file=""
else
  dataset_dir="datasets/strategyqa"
  train_file="train.json"
  test_file="test.json"
  val_file="validation.json"
fi

s_name=$(sed 's/-/_/g' <<< "$(sed 's/\//_/g' <<< "$student_model")")
t_name=$(sed 's/-/_/g' <<< "$(sed 's/\//_/g' <<< "$teacher_model")")
out_file=multi_turn_"$mental_model"_"$t_name"_"$utility"_"$s_name"_"$dataset"_"$(date '+%Y-%m-%d_%H-%M-%S')".out

if [ -z "$results_path" ]; then
    results_path="$data_dir"results/multi_turn_"$mental_model"_"$t_name"_"$utility"_"$s_name"_"$dataset".txt
fi

# export VLLM_LOGGING_LEVEL=DEBUG
# export CUDA_LAUNCH_BLOCKING=1
# export NCCL_DEBUG=TRACE
# export VLLM_TRACE_FUNCTION=1

# For running with more than 1 GPU

readarray -d "," -t gpus_avail <<< "$CUDA_VISIBLE_DEVICES"
student_gpus="${gpus_avail[@]:0:$n_student_gpus}"
teacher_gpus="${gpus_avail[@]:$n_student_gpus:$n_teacher_gpus}"

# For running with a single GPU

# readarray -d "," -t gpus_avail <<< "${CUDA_VISIBLE_DEVICES:-0}"
# if [ "${#gpus_avail[@]}" -eq 1 ] && [ "$n_student_gpus" -eq 1 ] && [ "$n_teacher_gpus" -eq 1 ]; then
#   student_gpus="${gpus_avail[0]}"
#   teacher_gpus="${gpus_avail[0]}"   # share the same GPU
# else
#   student_gpus="${gpus_avail[@]:0:$n_student_gpus}"
#   teacher_gpus="${gpus_avail[@]:$n_student_gpus:$n_teacher_gpus}"
# fi

student_gpus="${student_gpus// /,}"
teacher_gpus="${teacher_gpus// /,}"

if [ -z "$remote_model"  ]; then
  python src/multiturn_mm_experiments.py --data-dir "$data_dir"/"$dataset_dir" --cache-dir "$cache_dir" --train-filename "$train_file" --test-filename "$test_file" \
                                    --val-filename "$val_file" --results-path "$results_path" --task "$dataset" --student-model "$student_model" --n-test-samples "$n_test_samples" \
                                    --teacher-model "$teacher_model" --max-new-tokens 100 --n-beams 1 --n-ic-samples "$n_ic_samples" --mm-type "$mental_model" \
                                    --intervention-utility "$utility" --teacher-explanation-type "$teacher_expl" --student-explanation-type "$student_expl" --use-explanations \
                                    --use-gold-label --llm-lib "$lib" --temperature "$gen_temperature" --n-logprobs "$num_logprobs" --n-rounds "$n_rounds" --k-expl "$k_expl" \
                                    $n_train_samples_arg --use-instruct --mm-dynamic --sample-expl "$sample_expl" --log-level "$log_level" > "$out_file"


else
  if [ "$lib" = "vllm" ]; then
    echo "Serving teacher model using vLLM"
    echo "Model is located at http://$teacher_host:$teacher_port/v1"
    vllm_dtype="${VLLM_DTYPE:-auto}"
    CUDA_VISIBLE_DEVICES="$teacher_gpus" vllm serve "$teacher_model" --download-dir "$cache_dir" --dtype "$vllm_dtype" --api-key "$api_key" --gpu-memory-utilization "$gpu_usage" \
                                  --tensor-parallel-size "$n_teacher_gpus" --host "$teacher_host" --max-model-len 4096 --max-parallel-loading-workers 1 --swap-space 2 --port "$teacher_port" --disable-log-requests &
    teacher_id=$!

    # scontrol show job $SLURM_JOB_ID | egrep -i 'TRES|MinMemory|NumCPUs|QOS|Partition'
    # cat /sys/fs/cgroup/memory.max 2>/dev/null || true


    # wait_ready () {
    #   local url=$1
    #   for i in {1..60}; do
    #     if curl -sf "http://$url/health" | grep -q "ok"; then return 0; fi
    #     sleep 5
    #   done
    #   echo "ERROR: $url never became healthy"; exit 1
    # }
    # wait_ready "${teacher_host}:${teacher_port}"

    sleep 5m

    echo "Serving student model using vLLM"
    echo "Model is located at http://$student_host:$student_port/v1"
    vllm_dtype="${VLLM_DTYPE:-auto}"
    CUDA_VISIBLE_DEVICES="$student_gpus" vllm serve "$student_model" --download-dir "$cache_dir" --dtype "$vllm_dtype" --api-key "$api_key" --gpu-memory-utilization "$gpu_usage" \
                                  --tensor-parallel-size "$n_student_gpus" --host "$student_host" --max-model-len 4096 --port "$student_port" --disable-log-requests &
    student_id=$!
    sleep 5m
    # wait_ready "${student_host}:${student_port}"
  fi
  echo "Launching the Multi-Turn experiment script"

  if [ "$mm_dynamic" = 1 ]; then
    if [ "$error_prior" = 1 ]; then
      python -u src/multiturn_mm_experiments.py --data-dir "$data_dir"/"$dataset_dir" --cache-dir "$cache_dir" --train-filename "$train_file" --test-filename "$test_file" \
                                    --val-filename "$val_file" --results-path "$results_path" --task "$dataset" --student-model "$student_model" \
                                    --teacher-model "$teacher_model" --max-new-tokens 100 --n-beams 4 --n-ic-samples "$n_ic_samples" --n-test-samples "$n_test_samples" --mm-type "$mental_model" \
                                    --intervention-utility "$utility" --teacher-explanation-type "$teacher_expl" --student-explanation-type "$student_expl" --use-explanations \
                                    --use-gold-label --llm-lib "$lib" --remote --student-model-url "$student_model_url" --teacher-model-url "$teacher_model_url" --api-key "$api_key" \
                                    --temperature "$gen_temperature" --n-logprobs "$num_logprobs" --n-rounds "$n_rounds" --k-expl "$k_expl" \
                                    $n_train_samples_arg --use-instruct --mm-dynamic --error-prioritisation --sample-expl "$sample_expl" --log-level "$log_level" > "$out_file"
    else
      python -u src/multiturn_mm_experiments.py --data-dir "$data_dir"/"$dataset_dir" --cache-dir "$cache_dir" --train-filename "$train_file" --test-filename "$test_file" \
                                    --val-filename "$val_file" --results-path "$results_path" --task "$dataset" --student-model "$student_model" \
                                    --teacher-model "$teacher_model" --max-new-tokens 100 --n-beams 4 --n-ic-samples "$n_ic_samples" --n-test-samples "$n_test_samples" --mm-type "$mental_model" \
                                    --intervention-utility "$utility" --teacher-explanation-type "$teacher_expl" --student-explanation-type "$student_expl" --use-explanations \
                                    --use-gold-label --llm-lib "$lib" --remote --student-model-url "$student_model_url" --teacher-model-url "$teacher_model_url" --api-key "$api_key" \
                                    --temperature "$gen_temperature" --n-logprobs "$num_logprobs" --n-rounds "$n_rounds" --k-expl "$k_expl" \
                                    $n_train_samples_arg --use-instruct --mm-dynamic --sample-expl "$sample_expl" --log-level "$log_level" > "$out_file"
    fi
  else
    if [ "$error_prior" = 1 ]; then
      python -u src/multiturn_mm_experiments.py --data-dir "$data_dir"/"$dataset_dir" --cache-dir "$cache_dir" --train-filename "$train_file" --test-filename "$test_file" \
                                    --val-filename "$val_file" --results-path "$results_path" --task "$dataset" --student-model "$student_model" \
                                    --teacher-model "$teacher_model" --max-new-tokens 100 --n-beams 4 --n-ic-samples "$n_ic_samples" --n-test-samples "$n_test_samples" --mm-type "$mental_model" \
                                    --intervention-utility "$utility" --teacher-explanation-type "$teacher_expl" --student-explanation-type "$student_expl" --use-explanations \
                                    --use-gold-label --llm-lib "$lib" --remote --student-model-url "$student_model_url" --teacher-model-url "$teacher_model_url" --api-key "$api_key" \
                                    --temperature "$gen_temperature" --n-logprobs "$num_logprobs" --n-rounds "$n_rounds" --k-expl "$k_expl" \
                                    $n_train_samples_arg --use-instruct --error-prioritisation --sample-expl "$sample_expl" --log-level "$log_level" > "$out_file"
    else
      python -u src/multiturn_mm_experiments.py --data-dir "$data_dir"/"$dataset_dir" --cache-dir "$cache_dir" --train-filename "$train_file" --test-filename "$test_file" \
                                    --val-filename "$val_file" --results-path "$results_path" --task "$dataset" --student-model "$student_model" \
                                    --teacher-model "$teacher_model" --max-new-tokens 100 --n-beams 4 --n-ic-samples "$n_ic_samples" --n-test-samples "$n_test_samples" --mm-type "$mental_model" \
                                    --intervention-utility "$utility" --teacher-explanation-type "$teacher_expl" --student-explanation-type "$student_expl" --use-explanations \
                                    --use-gold-label --llm-lib "$lib" --remote --student-model-url "$student_model_url" --teacher-model-url "$teacher_model_url" --api-key "$api_key" \
                                    --temperature "$gen_temperature" --n-logprobs "$num_logprobs" --n-rounds "$n_rounds" --k-expl "$k_expl" \
                                    $n_train_samples_arg --use-instruct --sample-expl "$sample_expl" --log-level "$log_level" > "$out_file"
    fi
  fi
  if [ "$lib" = "vllm" ]; then
    kill -9 "$student_id" "$teacher_id"
  fi
fi

conda deactivate
date