# Simple VLM Nutrition Tracker

Uses a Multimodal Language model with vision capability to estimate the nutrition content of food in given image.

# Libraries

* HuggingFace
* Langchain

# VLM model

I used Bunny as the Vision Language Model (VLM) or Multimodal Large Language Model (MLLM) because it is lightweight but still powerful. 

Still there are some hallucinations. So using a better model might improve the usability of this application.

Compared to the typical LLMs, bunny is small and is based on phi-2 with 'only' 2.7B params.
It uses SigLIP vision encoder with 0.4B params.

[Bunny Repo](https://github.com/BAAI-DCAI/Bunny?tab=readme-ov-file)

# Project Structure

Scripts
- main
- bunny
- analyzer
- retriever
- extractor
- database
- knowledge\_scraper

Knowledge base
- Scraped Articles
- Nutrition tables
- Food Images with captions

Diet Journal
- DietDB
- Journal Files (summary)

# Langchain

* Analyzer: takes the initial description prompt text and image to figure out what the meal is.
* Retriever: RAG to get the nutrition information from the nutrition knowledge base (KB)
* Extractor: Takes analyzer output and returns meal nutrition info
* Chronicler: takes extractor output and summarizes the meals and saves it into the diet journal

# TODO

## Main Application

* Get image and text prompt from user.
* Does the image contain food?
* What food items are in the image?
* Retrieve KB to find nutrition info about each food item.
* What are the portion sizes of each item in the image?
* Based on this estimate the calories and amount of macronutrients
* Extract discrete diet data (Calories, Protein, Carbs, Fat, Fiber, list of Vitamins and Minerals were present)
* Save this as meal info in diet journal.

## Diet Journal
* each day is divided into meals
* each meal is divided into food items
* each food item has properties:
** calories (number)
** protein (n)
** carbs (n)
** fat (n)
** fiber (n)
** other (list of strings)
* the Total nutrition is calculated and updated for each day when a meal is added
* the information will be stored and summarized in human readable format so the user can view the journal independent of the application if they want to
** main database will have the daily diet objects and is linked to the summaries
** corresponding journal summary file as text/md file.

## Diet Assistant 
Analyses the diet journal to give insights about your nutrition based on your age, height, weight, activity, and in future possibly health conditions and other factors.

* Collect user data (height, weight, activity, age are the main ones to start with)
* Select time period (last week, last month, all/max)
* Retrieve the journals for the time period using RAG
* Prompt from user asking questions about their nutrition journey.

## Scraping additional data to improve model performance
* Scrape recipes and health articles with nutrition info (For RAG)
* Gather nutrition data from other sources (For RAG)
* Scrape recipes with food image and caption (for finetuning)
* Scrape reddit food subs where people post their meal photo with title as caption (for finetuning)

## Finetuning
* Do finetuning on food image and caption dataset

## Instruction tuning
* For assistant

# Conclusion

Proof of concept for now.

