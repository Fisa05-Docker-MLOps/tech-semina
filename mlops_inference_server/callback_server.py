from fastapi import FastAPI, Request

app = FastAPI(title="Callback Receiver")

@app.post("/result")
async def result(req: Request):
    payload = await req.json()
    print("[CALLBACK] payload:", payload, flush=True)
    return {"ok": True}
