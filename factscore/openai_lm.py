from lm import LM
import openai
import sys
import time
import os
import numpy as np
import logging


class OpenAIModel(LM):

    def __init__(self, model_name, cache_file=None, key_path="api.key"):
        self.model_name = model_name
        self.key_path = key_path
        self.temp = 0.7
        self.save_interval = 100
        super().__init__(cache_file)

    def load_model(self):
        # load api key
        key_path = self.key_path
        assert os.path.exists(key_path), f"Please place your OpenAI APT Key in {key_path}."

        with open(key_path, 'r') as f:
            api_key = f.readline()

        openai.api_key = api_key.strip()
        if self.model_name == "ChatGPT":
            openai.api_type = "azure"
            openai.api_base = 'https://topicpages.openai.azure.com/'
            openai.api_version = "2023-03-15-preview"
        elif self.model_name == "GPT4":
            openai.api_type = "azure"
            openai.api_base = "https://els-sdanswers-innovation.openai.azure.com/"
            openai.api_version = "2023-07-01-preview"
            os.environ["OPENAI_API_KEY"] = api_key.strip()

        self.model = self.model_name

    def _generate(self, prompt, max_sequence_length=2048, max_output_length=128):
        if self.add_n % self.save_interval == 0:
            self.save_cache()
        # return a tuple of string (generated text) and metadata (any format)
        # This should be about generating a response from the prompt, no matter what the application is
        if self.model_name == "ChatGPT":
            # Construct the prompt send to ChatGPT
            message = [{"role": "user", "content": prompt}]
            # Call API
            response = call_ChatGPT(message, temp=self.temp, max_len=max_sequence_length)
            # Get the output from the response
            try:
                output = response["choices"][0]["message"]["content"]
            except:
                output = "I cannot answer this question (not allowed)."
                print(response)
            return output, response
        elif self.model_name == "InstructGPT":
            # Call API
            response = call_GPT3(prompt, temp=self.temp)
            # Get the output from the response
            output = response["choices"][0]["text"]
            return output, response
        elif self.model == "GPT4":
            # Construct the prompt send to GPT4
            message = [{"role": "user", "content": prompt}]
            # Call API
            response = call_ChatGPT(message, model_name="gpt4", temp=self.temp, max_len=max_sequence_length)
            # Get the output from the response
            try:
                output = response["choices"][0]["message"]["content"]
            except:
                output = "I cannot answer this question (not allowed)."
                print(response)
            return output, response
        else:
            raise NotImplementedError()


def call_ChatGPT(message, model_name="gpt-3.5-turbo", max_len=1024, temp=0.7, verbose=False):
    # call GPT-3 API until result is provided and then return it
    response = None
    received = False
    num_rate_errors = 0
    while not received:
        try:
            if model_name == "gpt-3.5-turbo":
                response = openai.ChatCompletion.create(engine="topicpages-qa",
                                                        model=model_name,
                                                        messages=message,
                                                        max_tokens=max_len,
                                                        temperature=temp)
            else:
                response = openai.ChatCompletion.create(engine="gpt-4",
                                                        messages=message,
                                                        max_tokens=max_len,
                                                        temperature=temp)
            received = True
        except Exception as e:
            print(e)
            print(message)
            num_rate_errors += 1
            error = sys.exc_info()[0]
            if error == openai.error.InvalidRequestError:
                # something is wrong: e.g. prompt too long
                logging.critical(f"InvalidRequestError\nPrompt passed in:\n\n{message}\n\n")
                # assert False
                response = {"choices": [{"message": {"content": "I cannot answer this question (not allowed)."}}]}
                break

            logging.error("API error: %s (%d). Waiting %dsec" % (error, num_rate_errors, np.power(2, num_rate_errors)))
            time.sleep(np.power(2, num_rate_errors))
    return response


def call_GPT3(prompt, model_name="text-davinci-003", max_len=512, temp=0.7, num_log_probs=0, echo=False, verbose=False):
    # call GPT-3 API until result is provided and then return it
    response = None
    received = False
    num_rate_errors = 0
    while not received:
        try:
            response = openai.Completion.create(engine=model_name,
                                                model=model_name,
                                                prompt=prompt,
                                                max_tokens=max_len,
                                                temperature=temp,
                                                logprobs=num_log_probs,
                                                echo=echo)
            received = True
        except:
            error = sys.exc_info()[0]
            print(error)
            num_rate_errors += 1
            if error == openai.error.InvalidRequestError:
                # something is wrong: e.g. prompt too long
                logging.critical(f"InvalidRequestError\nPrompt passed in:\n\n{prompt}\n\n")
                assert False
            logging.error("API error: %s (%d)" % (error, num_rate_errors))
            time.sleep(np.power(2, num_rate_errors))
    return response
