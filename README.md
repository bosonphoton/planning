Run with openai:

```python
python main.py --provider openai --model_name gpt-3.5-turbo --num_samples 200 &> output_gpt35turbo.resultlog
```

Run with together:

```python
python main.py --provider together --model_name deepseek-ai/DeepSeek-V3 --num_samples 200 &> output_together_deepseek_v3.resultlog
```

Run the baseline with openai:
```python
python main24_baseline.py --provider openai --model_name gpt-4o --num_samples 5
```


* Before you start *

1. Install requirements via ```pip install -r requirements.txt```
2. Ensure you have the system variables OPENAI_API_KEY_1, ... OPENAI_API_KEY_4 (for openai) and likewise TOGETHER_API_KEY_1, ... TOGETHER_API_KEY_4 (for together)



- msg Chelsea to get env file if u dont have the api keys, or check our discord for messages by Varun and others
