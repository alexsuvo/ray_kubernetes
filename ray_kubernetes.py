from ray import serve
from starlette.requests import Request
from transformers import pipeline


@serve.deployment
class Summariser:

    def __init__(self, model_name):
        self.model = pipeline("summarization", model=model_name, tokenizer=model_name)

    def summarize(self, text: str) -> str:
        return self.model(text, min_length=5, max_length=10)

    async def __call__(self, http_request: Request) -> str:
        text = await http_request.json()
        return self.summarize(text['text'])


sum_app = Summariser.bind('t5-small')
