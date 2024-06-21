import os
import re
import json
import sys
import argparse
import openai
from openai import AzureOpenAI, OpenAI
from abc import ABC, abstractmethod
from tqdm import tqdm
import time
import random
import base64
from mimetypes import guess_type

from os import getenv
from dotenv import load_dotenv
load_dotenv()
API_BASE = getenv("API_BASE")
API_KEY = getenv("API_KEY")


class BaseAPIWrapper(ABC):
    @abstractmethod
    def get_completion(self, user_prompt, system_prompt=None):
        pass

class OpenAIAPIWrapper(BaseAPIWrapper):
    def __init__(self, caller_name="default", api_base="https://api.openai.com",  key_pool=[], temperature=0, model="gpt-4-vision-preview", time_out=5):
        if API_BASE != "":
            api_base = API_BASE
        key_pool = [API_KEY]
        self.temperature = temperature
        self.model = model
        self.time_out = time_out
        self.api_key = random.choice(key_pool)
        self.api_base = api_base
        self.client = OpenAI(
            api_key=self.api_key,
            api_base=self.api_base,
        )

    def request(self, usr_question, system_content, image_path=None):
        
        if "vision" in self.model:
            data_url = self.local_image_to_data_url(image_path)
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    # {"role": "system", "content": f"{system_content}"},
                    {"role": "user", "content": [
                        {
                            "type": "text",
                            "text": f"{usr_question}",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": data_url
                            }
                        }
                    ]}
                ],
                max_tokens=300,
            )
        else:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    # {"role": "system", "content": f"{system_content}"},
                    {"role": "user", "content": f"{usr_question}"}
                ],
            )

        # resp = response.choices[0]['message']['content']
        # total_tokens = response.usage['total_tokens']
        resp = response.choices[0].message.content
        total_tokens = response.usage.total_tokens
        

        return resp, total_tokens
    
    def get_completion(self, user_prompt=None, system_prompt=None, image_path=None, max_try=10):
        gpt_cv_nlp = ""
        key_i = 0
        total_tokens = 0
        # TODO: improve naive cache
        cached = 0
        cache_dir = "cache_dir/gpt4v_cache"
        if not os.path.exists(cache_dir): os.makedirs(cache_dir)
        if image_path is not None:
            cache_path = cache_dir + "/" +user_prompt.replace(" ","").replace("/","").replace(".","").replace("'","").replace("\n","")+image_path.replace("/","").replace(".","")+".json"
            if os.path.exists(cache_path):
                cache = json.load(open(cache_path))
                gpt_cv_nlp = cache["response"]
                total_tokens = cache["tokens"]
                cached = 1
        while max_try > 0 and cached == 0:
            try:
                gpt_cv_nlp, total_tokens = self.request(user_prompt, system_prompt, image_path)
                res = {
                    "response": gpt_cv_nlp,
                    "tokens": total_tokens
                }
                if image_path is not None:
                    cache_path = cache_dir+"/"+user_prompt.replace(" ","").replace("/","").replace(".","").replace("'","").replace("\n","")+image_path.replace("/","").replace(".","")+".json"        
                    json.dump(res, open(cache_path, "w"))
                max_try = 0
                break
            except Exception as e:
                # if e.code == "content_filter":
                #     gpt_cv_nlp, total_tokens = "", 0
                #     break
                print(f"encounter error: {e}")
                print("fail ", max_try)
                # key = self.key_pool[key_i%2]
                # openai.api_key = key
                # key_i += 1
                time.sleep(self.time_out)
                max_try -= 1
    
        return gpt_cv_nlp, total_tokens


    def local_image_to_data_url(self, image_path):
        # Guess the MIME type of the image based on the file extension
        mime_type, _ = guess_type(image_path)
        if mime_type is None:
            mime_type = 'application/octet-stream'  # Default MIME type if none is found

        # Read and encode the image file
        with open(image_path, "rb") as image_file:
            base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')

        # Construct the data URL
        return f"data:{mime_type};base64,{base64_encoded_data}"

