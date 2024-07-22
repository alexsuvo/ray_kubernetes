import logging
import os

from ray import serve
from starlette.requests import Request
from vllm import LLM, SamplingParams


@serve.deployment(ray_actor_options={"num_gpus": 1.0, "num_cpus": 4})
class QPilotVllm:

    def __init__(self):
        self.logger = logging.getLogger("ray.serve")

        self.model = LLM('meta-llama/Meta-Llama-3-8B-Instruct')
        self.logger.info('### Model loaded')

        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.logger.info('### Tokenizer loaded')

    async def __call__(self, http_request: Request) -> str:
        req = await http_request.json()
        self.logger.info(f'### Request body: {req}')

        question = req['question']
        self.logger.info(f'### Question:{question}')

        conversation = self.tokenizer.apply_chat_template([{'role': 'system', 'content': 'You are a powerful Q language question-answering system.'},
                                                           {'role': 'system', 'content': 'Your job is to provide a single and concise Q language answer to a Q language question.'},
                                                           {'role': 'system', 'content': 'Do not add Notes or examples'},
                                                           {'role': 'user', 'content': f'Q language question:{question}'}], tokenize=False)

        output = self.model.generate([conversation], SamplingParams(temperature=0,
                                                                    top_p=0.9,
                                                                    max_tokens=int(os.environ.get('CUTOFF_LEN')),
                                                                    stop_token_ids=[self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|eot_id|>")]))

        answer = output[0].outputs[0].text.replace('Q language answer:', '')
        self.logger.info(f'### Answer:{answer}')

        return answer


app = QPilotVllm.bind()
