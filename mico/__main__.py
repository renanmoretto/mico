import uvicorn

if __name__ == '__main__':
    uvicorn.run('mico.web:app', host='127.0.0.1', port=8787, reload=False)
