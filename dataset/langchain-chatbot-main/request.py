import requests

PORT=7000

url = f"http://localhost:{PORT}"

# ping = requests.get(f'{url}/ping')
# print(ping.text)

def get_stream(url):
    s = requests.Session()
    with s.post(url, headers=None, stream=True, json={'question': 'Jelaskan apa itu PKM-KI?', 'sid': '123c'}) as resp:
        for line in resp.iter_lines():
            if line:
              yield line.decode('utf-8')

url = f'{url}/ask/stream'
# url = 'https://jsonplaceholder.typicode.com/posts/1'
data_rcv = ''
for line in get_stream(url):
    data_rcv += line[6:]
    print(f"{data_rcv}\n")