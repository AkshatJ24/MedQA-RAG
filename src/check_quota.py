import requests, os
from dotenv import load_dotenv
from config import get_groq_api_key

load_dotenv('../.env')

r = requests.post(
    'https://api.groq.com/openai/v1/chat/completions',
    headers={'Authorization': f'Bearer {get_groq_api_key()}'},
    json={
        'model': 'openai/gpt-oss-20b',
        'messages': [{'role': 'user', 'content': 'hi'}],
        'max_completion_tokens': 1,
        'temperature': 1,
        'top_p': 1,
        'reasoning_effort': 'medium',
        'stream': False,
        'stop': None
    }
)

print('Status:', r.status_code)
print()
for k, v in r.headers.items():
    if 'ratelimit' in k.lower() or 'retry' in k.lower():
        print(f'{k}: {v}')