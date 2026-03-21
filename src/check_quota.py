import requests, os
from dotenv import load_dotenv

load_dotenv('../.env')

r = requests.post(
    'https://api.groq.com/openai/v1/chat/completions',
    headers={'Authorization': f'Bearer {os.getenv("GROQ_API_KEY")}'},
    json={
        'model': 'llama-3.1-8b-instant',
        'messages': [{'role': 'user', 'content': 'hi'}],
        'max_tokens': 1
    }
)

print('Status:', r.status_code)
print()
for k, v in r.headers.items():
    if 'ratelimit' in k.lower() or 'retry' in k.lower():
        print(f'{k}: {v}')