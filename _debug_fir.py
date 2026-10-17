from app import app

client = app.test_client()
client.post('/login', data={'username': 'field_officer', 'password': 'officer123'})

cases = [
    ('whitespace only', {'case_text': '   \n\t  '}),
    ('armed robbery', {'case_text': 'Robbery with pistol in Bengaluru IPC 392'}),
    ('special', {'case_text': "FIR: suspect used knife and gun at MG Road, Bengaluru. IPC 307/392. Alias Vicky."}),
]

for name, data in cases:
    r = client.post('/analyze_case', data=data)
    t = r.data.decode('utf-8', errors='replace')
    err = any(x in t for x in ['Traceback', 'UndefinedError', 'Internal Server Error', '500 Internal'])
    print(name, r.status_code, 'ERROR' if err else 'OK', 'results' if 'THREAT SEVERITY' in t else 'no-results')




