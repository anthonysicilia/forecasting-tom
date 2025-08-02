model='openai/gpt-3.5-turbo-0125'
frac=1
n=10
temp=1.0
p=1.0

# w/o demographics
python3 -m src.predict --model=$model --data='candor-fo' --mc_samples=$n --frac=$frac --temp=$temp --top_p=$p
python3 -m src.predict --model=$model --data='candor-so' --mc_samples=$n --frac=$frac --temp=$temp --top_p=$p
python3 -m src.predict --model=$model --data='casino' --mc_samples=$n --frac=$frac --temp=$temp --top_p=$p
python3 -m src.predict --model=$model --data='multiwoz' --mc_samples=$n --frac=$frac --temp=$temp --top_p=$p
# with demographics [START here]
python3 -m src.predict --model=$model --data='candor-fo' --mc_samples=$n --frac=$frac --temp=$temp --top_p=$p --demographics=1
python3 -m src.predict --model=$model --data='candor-so' --mc_samples=$n --frac=$frac --temp=$temp --top_p=$p --demographics=1
python3 -m src.predict --model=$model --data='casino' --mc_samples=$n --frac=$frac --temp=$temp --top_p=$p --demographics=1
# feature cache update
python3 -m src.tasks.make_cache 