import logging
import json
import azure.functions as func
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline


tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
nlp = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    text = req.params.get('text')
    if not text:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            text = req_body.get('text')

    sentiment = nlp(text)
    if text:
        return func.HttpResponse(json.dumps({"text": text, "sentiment": sentiment}), mimetype="application/json")
    else:
        return func.HttpResponse(
             "Function executed successfully. Pass a text in the query string or in the request body for sentiment analysis.",
             status_code=400
        )