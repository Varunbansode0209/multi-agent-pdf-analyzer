
import logging
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.concurrency import run_in_threadpool

from backend.core.config import settings
from backend.ingestion.pdf_loader import InvalidPDFError
from backend.services.document_service import (
    document_service,
    sanitize_filename,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/documents",
    tags=["Documents"],
)


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="A filename is required.",
        )

    original_filename = sanitize_filename(file.filename)

    if not original_filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported.",
        )

    upload_dir = Path(settings.UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)

    temporary_path = upload_dir / f"{uuid4().hex}.pdf"

    try:
        total_size = 0

        with temporary_path.open("wb") as destination:
            while True:
                content = await file.read(1024 * 1024)

                if not content:
                    break

                total_size += len(content)

                if total_size > settings.MAX_UPLOAD_MB * 1024 * 1024:
                    raise HTTPException(
                        status_code=413,
                        detail="PDF exceeds the upload size limit.",
                    )

                destination.write(content)

        if total_size == 0:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty.",
            )

        result = await run_in_threadpool(
            document_service.ingest_document,
            str(temporary_path),
            original_filename,
        )

        return result

    except HTTPException:
        raise

    except InvalidPDFError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        logger.exception("Document ingestion failed.")
        raise HTTPException(
            status_code=500,
            detail="Document ingestion failed.",
        ) from exc
    finally:
        await file.close()

        if temporary_path.exists():
            temporary_path.unlink()
