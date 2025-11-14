import base64
from fastapi import APIRouter
from pydantic import BaseModel
import httpx
import hashlib
import os

router = APIRouter()

# Simple cache for screenshots - could be improved with Redis
SCREENSHOT_CACHE_DIR = "screenshot_cache"
if not os.path.exists(SCREENSHOT_CACHE_DIR):
    os.makedirs(SCREENSHOT_CACHE_DIR)


def bytes_to_data_url(image_bytes: bytes, mime_type: str) -> str:
    base64_image = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{base64_image}"


async def capture_screenshot(
    target_url: str, api_key: str, device: str = "desktop"
) -> bytes:
    api_base_url = "https://api.screenshotone.com/take"

    params = {
        "access_key": api_key,
        "url": target_url,
        "full_page": "true",
        "device_scale_factor": "1",
        "format": "png",
        "block_ads": "true",
        "block_cookie_banners": "true",
        "block_trackers": "true",
        "cache": "false",
        "viewport_width": "342",
        "viewport_height": "684",
    }

    if device == "desktop":
        params["viewport_width"] = "1280"
        params["viewport_height"] = "832"

    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.get(api_base_url, params=params)
        if response.status_code == 200 and response.content:
            return response.content
        else:
            raise Exception("Error taking screenshot")


class ScreenshotRequest(BaseModel):
    url: str
    apiKey: str


class ScreenshotResponse(BaseModel):
    url: str


@router.post("/api/screenshot")
async def app_screenshot(request: ScreenshotRequest):
    # Extract the URL from the request body
    url = request.url
    api_key = request.apiKey

    # Simple caching mechanism for screenshots
    cache_key = hashlib.md5(url.encode()).hexdigest()
    cache_file = os.path.join(SCREENSHOT_CACHE_DIR, f"{cache_key}.png")
    
    # Check if cached version exists
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            image_bytes = f.read()
        print(f"Using cached screenshot for {url}")
    else:
        # TODO: Add error handling
        image_bytes = await capture_screenshot(url, api_key=api_key)
        
        # Cache the screenshot
        with open(cache_file, "wb") as f:
            f.write(image_bytes)
        print(f"Cached new screenshot for {url}")

    # Convert the image bytes to a data url
    data_url = bytes_to_data_url(image_bytes, "image/png")

    return ScreenshotResponse(url=data_url)
