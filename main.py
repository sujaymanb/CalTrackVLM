import os
from fastapi import FastAPI, File, UploadFile, Request
from fastpi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI()
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")\

images = []

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "home.html", {"request": request, "videos": videos})

@app.post("/uploadimage/")
async def upload_image(image: UploadFile = File(...)):
    new_image = process_image(image.filename)
    images.append(new_image)
    return RedirectResponse(url="/", status_code=303)