import os
from llm import get_response

user_msg = "Hello"
print(get_response([{"role": "user", "content": user_msg}]))
