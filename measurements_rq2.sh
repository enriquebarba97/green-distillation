#!/bin/bash
# Number of possible values
VALUES=20

# Number of repetitions for each value
REPETITIONS=20

# Name of experiment and path
#NAME="CodeT5_Summarization"
#TASK_PATH="CodeT5+/Summarization"

NAME="GCB-Vulnerability-Detection"
TASK_PATH="GraphCodeBERT/Vulnerability-Detection/Morph"


# Initialize an associative array to track counts
declare -A COUNTS

# Create directories for storing results for each value
for ((i = 0; i < VALUES; i++)); do
    #mkdir -p "energy/model-$i/training"
    #mkdir -p "energy/model-$i/evaluation"
    mkdir -p "energy/${NAME}/energy-op/model-$i/evaluation"
    mkdir -p "energy/${NAME}/flops-op/model-$i/evaluation"
done

# Initialize all counts to 0
for ((i = 0; i < 2*VALUES; i++)); do
    COUNTS[$i]=0
done

# Trap sigint, stop docker and break the loop
function cleanup {
    echo "Stopping Docker containers..."
    docker stop $(docker ps -q)
    exit 0
}
trap cleanup SIGTERM SIGINT

# Generate a random sequence of values and test the script
while true; do
    # Create a list of values that are not fully tested
    NOT_FULLY_TESTED=()
    for ((i = 0; i < 2*VALUES; i++)); do
        if [[ ${COUNTS[$i]} -lt $REPETITIONS ]]; then
            NOT_FULLY_TESTED+=($i)
        fi
    done

    # Exit the loop if all values have been tested 15 times
    if [[ ${#NOT_FULLY_TESTED[@]} -eq 0 ]]; then
        break
    fi

    # Select a random value from the list of not fully tested values
    RANDOM_INDEX=$((RANDOM % ${#NOT_FULLY_TESTED[@]}))
    RANDOM_VALUE=${NOT_FULLY_TESTED[$RANDOM_INDEX]}


    # Increment the count for this value
    COUNT=${COUNTS[$RANDOM_VALUE]}
    COUNT=$((COUNT + 1))
    COUNTS[$RANDOM_VALUE]=$COUNT

    USE_FLOPS=""
    OP_ROUTE="energy-op"
    REAL_VALUE=$RANDOM_VALUE

    if [[ $RANDOM_VALUE -ge $VALUES ]]; then
        USE_FLOPS="--use-flops"
        OP_ROUTE="flops-op"
        REAL_VALUE=$(($RANDOM_VALUE - $VALUES))
    fi

    # Call the script with the random value as an argument
    echo "Running with argument: $REAL_VALUE $USE_FLOPS. Count: ${COUNTS[$RANDOM_VALUE]}"
    #$SCRIPT_TO_TEST "$RANDOM_VALUE"


    #CMD="/home/enrique/EnergiBridge/target/release/energibridge --gpu -o energy/model-${RANDOM_VALUE}/training/${COUNT}.csv docker run -a stdout --gpus all --rm -v $(pwd):/root/Morph morph_env /bin/bash -c 'cd /root/Morph/CodeBERT/Clone-Detection/Morph/; python3 train_energy_test.py --model $RANDOM_VALUE'"
    #eval $CMD
    #sleep 60
    CMD="/home/enrique/EnergiBridge/target/release/energibridge --gpu -o energy/${NAME}/${OP_ROUTE}/model-${REAL_VALUE}/evaluation/${COUNT}.csv docker run -a stdout --gpus all --rm -v $(pwd):/root/green green_env /bin/bash -c 'cd /root/green/${TASK_PATH}; python3 evaluation_energy_test.py --model $REAL_VALUE $USE_FLOPS'"
    eval $CMD
    sleep 30

done
