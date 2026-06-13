set -u
cd "$(dirname "$0")"
mkdir -p logs

STAGES=("$@")
if [ ${#STAGES[@]} -eq 0 ]; then
    STAGES=(0 1 2 3 4)
fi

PIDS=()
for s in "${STAGES[@]}"; do
    nohup python learn.py --stage "$s" \
        > "logs/stage${s}.out" 2>&1 &
    PIDS+=("$!")
    echo "stage $s  ->  PID $!   (log: logs/stage${s}.out)"
done

echo
echo "${#STAGES[@]} trainings running. Stop all with:  kill ${PIDS[*]}"
echo "When done, chain them:  python test.py --order 4 2 3 1 5"
