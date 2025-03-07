import os

import torch
from peft import AutoPeftModelForCausalLM
from ray import serve
from starlette.requests import Request
from transformers import AutoTokenizer, GenerationConfig, BitsAndBytesConfig


@serve.deployment(ray_actor_options={"num_gpus": 1.0, "num_cpus": 8})
class QPilot:

    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'### Device:{self.device}')

        quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=False)
        self.model = AutoPeftModelForCausalLM.from_pretrained(os.environ.get('MDL_LOC'), token=os.environ.get('HF_TOKEN'), trust_remote_code=True, quantization_config=quant_config)
        print('### Model loaded')

        self.tokenizer = AutoTokenizer.from_pretrained(os.environ.get('BASE_MDL_NAME'), token=os.environ.get('HF_TOKEN'), model_max_length=int(os.environ.get('CUTOFF_LEN')), padding_side="right")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        print('### Tokenizer loaded')

    def get_output(self, text):
        torch.cuda.empty_cache()

        inputs = self.tokenizer.encode_plus(text, return_tensors="pt", padding="longest", truncation=True, max_length=int(os.environ.get('CUTOFF_LEN')))

        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        output = self.model.model.generate(input_ids, attention_mask=attention_mask, max_length=int(os.environ.get('CUTOFF_LEN')),
                                           generation_config=GenerationConfig(temperature=0.001, top_p=1.0, num_beams=1, pad_token_id=self.tokenizer.eos_token_id))

        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    async def __call__(self, http_request: Request) -> str:
        req = await http_request.json()
        print(f'### Request body: {req}')

        question = req['question']
        print(f'### Question:{question}')

        prompt = f"""<|begin_of_text|>
        <|start_header_id|>system<|end_header_id|>You are a powerful Q language question-answering system.<|eot_id|>
        <|start_header_id|>system<|end_header_id|>Your job is to provide a single and concise Q language answer to a Q language question.<|eot_id|>
        <|start_header_id|>system<|end_header_id|>Do not add Notes or examples.<|eot_id|>
        <|start_header_id|>user<|end_header_id|>Q language question:{question}<|eot_id|>"""
        answer = self.get_output(prompt)
        print(f'### Answer:{answer}')

        return answer


app = QPilot.bind()
