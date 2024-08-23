from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import time

SCRAPE_PARENT_URLS = False
SCRAPE_FOOD = True

if SCRAPE_PARENT_URLS:
    driver = webdriver.Chrome() 

    url = "https://www.nutritionix.com/brands/grocery"
    driver.get(url)
    wait = WebDriverWait(driver, timeout=2)

    xpath = f"//button[text()='Refuse']"

    try:
        button = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, xpath))
        )
        button.click()
    except:
        print(f"Button with text '{xpath}' not found")

    with open("./data/food_urls.txt", "w") as f:
        while(True):
            raw = driver.find_elements(By.TAG_NAME, "a")

            for link in raw:
                href = link.get_attribute('href')
                if href is not None and '/brand/' in href:
                    f.write(href)
                    f.write('\n')
            
            next_page_button = driver.find_element(By.LINK_TEXT, "Next")

            if next_page_button is None:
                break
            else:
                wait.until(lambda d : next_page_button.is_displayed())
                next_page_button.click()

if SCRAPE_FOOD:
    driver = webdriver.Chrome() 

    with open("./data/food_urls.txt", "r") as f:
        for line in f:
            url = line
            driver.get(url)
            wait = WebDriverWait(driver, timeout=2)

            xpath = f"//button[text()='Refuse']"

            try:
                button = WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.XPATH, xpath))
                )
                button.click()
            except:
                pass

            # TODO: go into each table row and extract nutrient info