import sys
import requests

def fed_avg_done():
    payload = {'status': "success"}
    r = requests.get('http://localhost:3000/fed_avg_done', params=payload)

fed_avg_done()
