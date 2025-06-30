# usage: ./run.sh config/substract.py
config=$1

base=$(basename $config .py)
python -u main.py --config $config 1> log/$base.log 2>&1 &
PID=$!
date=$(date)
echo "$date [$base] started, PID: $PID" >> pid.log
echo "$date [$base] started, PID: $PID"
