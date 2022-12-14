#!/bin/bash
# modified version of https://github.com/adap/flower/blob/main/examples/quickstart_pytorch/run.sh


NUM_CLIENTS=5

set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/


echo "Starting server"
python run_fedavg_server.py &
sleep 3  # Sleep for 3s to give the server enough time to start

for i in $(seq 0 $(expr $NUM_CLIENTS - 1)); do
    echo "Starting client $i"
    python run_client.py cid=$i &
done

# Enable CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait