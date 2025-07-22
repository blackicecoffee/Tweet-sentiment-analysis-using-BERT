# Tweet sentiment analysis by finetuning BERT

## How to run

1. Create virtual environment: `python -m venv <your_environment_name>`

2. Activate virtual environment:
- On Windows, using **Command Prompt**: `.\<your_environment_name>\Scripts\activate`

- On Macs/Linux, using **Bash**: `source <your_environment_name>/bin/activate`

3. Install required packages: `pip install -r requirements.txt`

4. Training model: `python src/train.py`

5. Model inference: `python src/infer.py`
