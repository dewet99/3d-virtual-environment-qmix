source Python/MLAgents_Portal/bin/activate
while true; do
timeout -sHUP 1m tensorboard --logdir=results;
done
