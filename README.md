
# Sentiment Analysis Pipeline for Call Center Conversations 

**Author:** Álvaro Rivera Arcelus

### Relevant aspects
*   This model is specifically developed for processing **Spanish conversations** this includes precise **mexican jargon**. Nevertheless, this same workflow can be applied for any language by only implementing subtle changes to the code. 


## Index

* [Data Pipeline](#data-pipeline)
* [Storage Architecture](#storage-architecture)
* [Models](#models)
* [Libraries and packages](#libraries-and-packages)
* [Functions](#functions) 
* [Relevant variables](#relevant-variables)
* [Limitations](#limitations)

## Data Pipeline
<img style='margin-top:20px; margin-bottom:20px; ' src='images/workflow_calls.png'>


**Note:** take into consideration that this pipeline is developed specifically for proecessing files formated according to Five9's specifications. 

# Storage Architecture
<img style='margin-top:20px; margin-bottom:20px; ' src='images/storage_architecture.png'>

## Models

### **Speech diarization model**
For differenciating which speaker spoke throughout the conversation, we used a *diarization model* known as `pyannote.audio` found at [Hugging Face - Pyannote](https://huggingface.co/pyannote)
```
from pyannote.audio import Pipeline
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1",
                                    use_auth_token="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")
```

**How does it work?** 

<img style='margin-top:20px; margin-bottom:20px; max-width:800px;' src='images/asr_sd_diagram.png'>

### **Speech-to-text model**
For transforming all audio files to text, we used Chat GPT's python library `whisper.ai`
```
#function that transcribes an audio
model = whisper.load_model('medium') 

def transcribe_me(audio):
    result = model.transcribe(audio, language='es', fp16=False)
    return result['text']

```


### **Transformer models**
This pipeline uses a set of pre-trained neural network models known as Transformer models. 
In this project we used the `transformers` library available at [Hugging Face - Transformers](https://huggingface.co/learn/nlp-course/chapter0/1?fw=pt)

Transformer models used: 
*   Text classification pipeline $\rightarrow$ [link](https://huggingface.co/docs/transformers/main_classes/pipelines)
    ```
    sentiment = pipeline('sentiment-analysis', model='pysentimiento/robertuito-sentiment-analysis', truncation=True)
    ```
*   Zero-shot classification $\rightarrow$ [link](https://huggingface.co/learn/nlp-course/chapter1/3?fw=pt)
    ```
    classifier = pipeline("zero-shot-classification", model='facebook/bart-large-mnli')
    ```

### **Word frequency model**
This model was developed manually by following all standard procedures for text analysis such as tokenization, stop words removal and so forth. Then, we simply determine whether either speaker said a word inside any of our two following lists: 
1. *lenguaje_inapropiado.txt*
2. *molestias.txt*

Both lists include common spanish expressions as well as "mexican slangs" commonly used to express distress or any sort of nuissance. 

## Libraries and packages 
Amongst multiple open-source libraries used, some of the most relevant are the following: 
*   NLTK
*   PyAnnote 
*   TensorFlow
*   Whisper
*   Transformers

Fot the entire list of libraries & packages used, read the file `libraries.py` found inside this same repository. 

## Functions 
All local functions are saved and organized based on their purpose. 
*   Chat conversations preprocessing `chatPreprocessing.py` 
*   Audio data cleansing `cleansing.py`
*   Data preprocessing `preprocessing.py`
*   All NLP and transformer models `myTransformers.py`

## Relevant variables 
Here is the entire list of all output variables calculated through the entire pipeline:

*   *init_res* $\rightarrow$ initial reponse time of the first person  to start the conversation
*   *avg_res_client* $\rightarrow$ average response time from client (minutes)
*   *avg_res_empl* $\rightarrow$ average response time from employee (minutes)
*   *first_res* $\rightarrow$ whether the employee, agent or bot initialized the conversatoin
*   *webhook* $\rightarrow$ if the first message was an image
    *   Useful to filter out onversations which the first message was an automatic screenshot sent to the debt collection agent only as a *proof of payment*. 
*   *client_sent* $\rightarrow$ client sentiment (POS,NEU,NEG)
*   *client_scr* $\rightarrow$ client sentiment classification probabilty 
*   *employee_sent* $\rightarrow$ employee sentiment (POS,NEU,NEG)
*   *employee_scr* $\rightarrow$ employee sentiment classification probabilty 
*   *client_em* $\rightarrow$ client's main emotion in the conversation (emotion classification)
*   *employee_em* $\rightarrow$ employee's main emotion in the conversation
*   *client_swear* $\rightarrow$ whether the client spoke innapropiately or not
*   *employee_swear* $\rightarrow$ whether the employee spoke innapropiately or not
*   *depto* $\rightarrow$ department to wich the agent belongs
*   *client_disturb* $\rightarrow$ whether the client explicitly said a word that expresses a disconfort with the comopany's services. 

## Limitations 
*   The diarization model performs well but fails to segment speakers correctly in some cases. 
*   Using Chat GPT's AI library `whisper` for transcribing every conversation, there are many cases in which the model does not perform correctly given the low quality of the call. 
*   Some calls exported from CRM's tend to record voicemail calls. They could be easily filtered by adding an aditional step in the preprosesing job. 

