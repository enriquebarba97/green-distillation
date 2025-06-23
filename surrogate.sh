# Loop from 70 to 80 (80 is exclusive)
for i in {70..79}
do
    CMD="docker run -a stdout --gpus all --rm -v $(pwd):/root/green green_env /bin/bash -c 'cd /root/green/CodeT5+/Summarization/; python3 create_surrogate_model.py --start-from $i --single"