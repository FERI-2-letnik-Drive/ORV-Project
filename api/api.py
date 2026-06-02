from fastapi import FastAPI, HTTPException, UploadFile, File

from app.face_comparison import face_comparison

app = FastAPI(title="ORV Face Verification API")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/v1/face-verifications")
async def face_verification(
    reference_image: UploadFile = File(...),  # ... = obvezno; File -> multipart/form-data
    current_image: UploadFile = File(...),
):
    reference_bytes = await reference_image.read()
    current_bytes = await current_image.read()

    if not reference_bytes:
        raise HTTPException(status_code=400, detail="Reference image is empty")
    if not current_bytes:
        raise HTTPException(status_code=400, detail="Current image is empty")

    result = compare_faces(reference_bytes, current_bytes)

    return {
        "match": result["match"],
        "confidence": result["confidence"],
        "message": result["message"],
    }


def compare_faces(reference_image: bytes, current_image: bytes) -> dict:
    """Tanka ovojnica nad modelom: pretvori napake v 400 Bad Request."""
    try:
        return face_comparison(reference_image, current_image)
    except ValueError as error:
        # npr. neveljavni/nedekodljivi bajti slike
        raise HTTPException(status_code=400, detail=str(error))
