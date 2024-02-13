import os
os.environ["HF_HOME"] = "/nlp/cache/huggingface/"
os.environ["TRANSFORMERS_CACHE"] = "/nlp/cache/huggingface/"
os.environ["HF_DATASETS_CACHE"] = "/nlp/cache/huggingface/"

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

        # system_message = """
        # You are a smart, helpful research assistant.
        # You are tasked with verifying whether a statement is True or False.
        # Utilize only the given context and make sure to NOT use your own knowledge.
        # If the statement is true, respond with "True". If the statement is false, respond with "False". If the context does not contain enough information to verify the statement, respond with "False".
        #
        # VERY IMPORTANT: Answer using a single word, either "True" or "False". No other words are allowed!
        #  """

        system_message = """
        You are a smart, helpful research assistant.
        "Do not reply using a complete sentence, only give the answer in the following format: {False} or {True}."
        """
        message = [{
        "role": "system",
        "content": system_message},
        {"role": "user", "content": prompt}
        ]

        token_prompt = self.model.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        tries = 0
        while True:
            response = self.model(token_prompt, max_new_tokens=max_sequence_length, do_sample=True, temperature=0.7,
                                   top_k=50, top_p=0.95)

            output = response[0]["generated_text"]
            cutline = "<|assistant|>"
            idx_output = output.find(cutline)
            refined_output = output[idx_output + len(cutline):]
            refined_output = refined_output.strip()

            if "True" in refined_output or "False" in refined_output:
                break
            tries += 1
            if tries == 5:
                break
        with open("/home/palfib/factscore/responses.txt", "a") as f:
            f.write(f"Response: -S- {refined_output} -E-\n")
            f.write("\n")

        return refined_output, response
