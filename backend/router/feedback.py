from fastapi import APIRouter

from models import FeedbackRequest, FeedbackResponse, new_id, utc_now

router = APIRouter()


@router.post("/api/feedback", response_model=FeedbackResponse)
async def feedback(payload: FeedbackRequest) -> FeedbackResponse:
    return FeedbackResponse(ok=True, id=new_id("feedback"), created_at=utc_now())
