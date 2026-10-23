import argparse
from fastapi import FastAPI, Request
import torch
import uvicorn
import json
import datetime
import os

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

app = FastAPI()

# GPU visibility is controlled by the operator via CUDA_VISIBLE_DEVICES.
# Global variables for model and tokenizer
model = None
tokenizer = None

@app.post("/v1/chat/completions")
async def create_item(request: Request):
    global model, tokenizer
    try:
        json_post_raw = await request.json()
        max_length = json_post_raw.get('max_tokens')
        top_p = json_post_raw.get('top_p')
        temperature = json_post_raw.get('temperature')
        messages = json_post_raw.get('messages')
        repetition_penalty = json_post_raw.get('repetition_penalty')

        sampling_params = SamplingParams(temperature=temperature, top_p=top_p, repetition_penalty=repetition_penalty, max_tokens=max_length)

        inputs = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        outputs = model.generate(inputs, sampling_params)
        response = outputs[0].outputs[0].text

        now = datetime.datetime.now()
        time = now.strftime("%Y-%m-%d %H:%M:%S")
        answer = {
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response,
                }
            }],
        }
        log = f"[{time}] prompt: {messages[0]['content']}, response: {repr(response)}"
        print(log)
        return answer["choices"][0]["message"]["content"]

    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        print(error_message)
        return {"error": error_message}

def parse_args():
    parser = argparse.ArgumentParser(description="Run FastAPI server with custom port and model")
    parser.add_argument('--model', type=str, required=True, help="Model to load (e.g., 'microsoft/Phi-3-mini-4k-instruct')")
    parser.add_argument('--port', type=int, default=4000, help="Port to run the server on")
    parser.add_argument('--host', default='127.0.0.1', help='Interface to bind')
    return parser.parse_args()

if __name__ == '__main__':
    # Command line arguments
    args = parse_args()

    # Model and tokenizer loading based on provided model name
    model_dir = args.model
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = LLM(model=model_dir, tensor_parallel_size=torch.cuda.device_count(), trust_remote_code=True)

    uvicorn.run(app, host=args.host, port=args.port, workers=1)
