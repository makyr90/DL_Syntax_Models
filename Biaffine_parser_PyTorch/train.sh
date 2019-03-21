stdbuf -oL python3 -u main.py --cuda | while IFS= read -r line
do
tee -a model_log.txt
done
