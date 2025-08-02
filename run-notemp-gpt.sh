model='openai/gpt-3.5-turbo-0125'
frac=1
n=1
temp=0
p=1.0

python3 -m src.predict --model=$model --data='multiwoz' --mc_samples=$n --frac=$frac --temp=$temp --top_p=$p