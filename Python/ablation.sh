#/usr/bin/bash

ARGS=("all_inc" "none_inc" "use_per" "use_burnin" "n_step_return" "standardise_rewards" "num_executors")


echo "Args set"

for _ in {1..3}; do
for arg in "${ARGS[@]}"; do
	python3 main.py "$arg" &
	last_pid=$!
	wait "$last_pid"
	echo "Python script with argument '$arg' terminated."
done 
done
