from fastapi import FastAPI, HTTPException, UploadFile, File
from app.face_comparison import face_comparison
app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/compare")
async def compare(
    reference_image: UploadFile = File(...), # ... = required, = File -> look in multipart/form-data
    current_image: UploadFile = File(...)
):
    reference_bytes = await reference_image.read()
    current_bytes = await current_image.read()

    if not reference_bytes:
        raise HTTPException(status_code=400, detail="Reference image is empty") # bad request. Request was invalid

    if not current_bytes:
        raise HTTPException(status_code=400, detail="Current image is empty")

    result = compare_faces(reference_bytes, current_bytes)

    return {
        "match": result["match"],
        "confidence": result["confidence"],
        "message": result["message"],
    }

def compare_faces(reference_image: bytes, current_image: bytes) -> dict:
    '''
    return {
        "match": True,
        "confidence": 0.87,
        "message": "Fixed response. Face comparison is not implemented yet."
    }
    '''
    return face_comparison(reference_image, current_image)