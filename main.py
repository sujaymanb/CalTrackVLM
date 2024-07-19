import os
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import RedirectResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from model import Model

app = FastAPI()
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

responses = []

model = Model()

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "home.html", {"request": request, "response": responses})

@app.post("/uploadimage/")
async def upload_image(image: UploadFile = File(...)):
    file_loc = f"static/{image.filename}"
    with open(file_loc, "wb+") as file:
        file.write(image.file.read())
    
    # inference and parse
    meal_data = model("", image.filename)

    # TODO process output and log meal
    
    # debug output just print the parsed data
    output = str(meal_data)

    # display response
    responses.append(output)
    
    return RedirectResponse(url="/", status_code=303)
