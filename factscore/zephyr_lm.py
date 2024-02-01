from lm import LM
import torch
from transformers import pipeline


class Zephyr(LM):
    def __init__(self, cache_file=None):
        self.save_interval = 100
        if cache_file:
            super().__init__(cache_file)

    def load_model(self):
        self.model = pipeline("text-generation", model="HuggingFaceH4/zephyr-7b-beta", torch_dtype=torch.bfloat16,
                              device_map="auto")

    def _generate(self, prompt, max_sequence_length=2048, max_output_length=None):
        if self.add_n % self.save_interval == 0:
            self.save_cache()

        message = [{"role": "user", "content": prompt}]

        token_prompt = self.model.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        response = self.model(token_prompt, max_new_tokens=max_sequence_length, do_sample=True, temperature=0.7,
                               top_k=50, top_p=0.95)

        output = response[0]["generated_text"]
        cutline = "<|assistant|>"
        idx_output = output.find(cutline)
        refined_output = output[idx_output + len(cutline):]


        return refined_output, response
