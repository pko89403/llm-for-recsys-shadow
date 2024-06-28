## MACRec: a Multi-Agent Collaboration Framework for Recommendation

- This is shadow implementation of [MACRec](https://github.com/wzf2000/MACRec). ( fix several bugs )
- Implement OpenAI API and ml-100k dataset only.
- Not implemented for train and feedback

## Install Requirements
```sh
pip install -r requirements.txt # include unecessary also sorry about that
```

## Run with the command line

### Run Analyst Agent demo
```sh
python macrec/agents/analyst.py
# item 0
# user 0
```
### Run Interpreter Agent demo
```sh
python macrec/agents/interpreter.py
# I'm very interested in watching movie. But recently I couldn't find a movie that satisfied me very much. Do you know how to solve this?
```
### Run Searcher Agent demo
```sh
python macrec/agents/searcher.py
# KOREA
```

### Run Converational System demo
```sh
python macrec/systems/chat.py 
# 1. Hello! How are you today?
# 2. I have watched the movie Schindler's List recently. I am very touched by the movie. I wonder what other movies can teach me about history like this?
```
### Run Reacy System demo
```sh
python macrec/systems/react.py
```


### Run with the web demo
```sh
streamlit run web_demo.py
```
Then open the browser and visit `http://localhost:8501/` to use the web demo.