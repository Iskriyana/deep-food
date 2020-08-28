"""Helper functions for scraping Google image search.
"""

import selenium
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from selenium.webdriver.common.action_chains import ActionChains

import io
import PIL
from PIL import Image
from base64 import b64decode
import numpy as np
import time
import requests
import os


def scroll_to_bottom(driver):
    """Scroll to bottom of the page using Selenium
    """
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(1)


def google_query(query, images_to_download=600, verbose=1):
    """Start a Google query using Selenium and get the URLs of all images

    Args:
        query (str): Input string for search query
        verbose (int): How many print statements to execute

    Returns:
        image_urls (set): Set of all URLs that could be retrieved
    """

    search_url = "https://www.google.com/search?safe=off&site=&tbm=isch&source=hp&q={q}&oq={q}&gs_l=img"

    # intialize empty set
    image_urls = set()

    driver = webdriver.Firefox()
    driver.maximize_window()
    driver.get(search_url.format(q=query))

    image_count = len(image_urls)
    delta = 0 # difference counter
    count = 0 # image counter
    while image_count < images_to_download:

        for i in range(6):
            # go to bottom of Google search page and show all images
            scroll_to_bottom(driver)

        images = []
        error_count = 0
        while True:
            try:
                img = driver.find_element_by_xpath(
                    f'//*[@id="islrg"]/div[1]/div[{count+1}]/a[1]/div[1]/img'
                )
                images.append(img)
                count += 1

            except NoSuchElementException:
                count += 1
                error_count += 1

                if verbose >= 10:
                    print(f"Can't find image: {count}")

                if error_count > 25:
                    if verbose >= 10:
                        print(f"More than {error_count} failed attempts. \
                        It is likely that there are not more images available.")
                    break

            except:
                print("Error during selenium interaction")

        if verbose >= 1:
            print(f"{len(images)} images found.")

        for img in images:
            image_urls.add(img.get_attribute('src'))

        # detect how many new images were discovered
        delta = len(image_urls) - image_count
        image_count = len(image_urls)

        if delta == 0:
            if verbose >= 1:
                print("Can't find more images")
            break

        # click "Show more results button"
        fetch_more_button = driver.find_element_by_xpath(
            '//*[@id="islmp"]/div/div/div/div/div[5]/input'
        )
        if fetch_more_button.is_displayed():
            actions = ActionChains(driver)
            actions.move_to_element(fetch_more_button)
            actions.click(fetch_more_button)
            actions.perform()
            scroll_to_bottom(driver)
            time.sleep(5)
        else:
            if verbose >= 1:
                print("Looks like you've reached the end.")

    driver.quit()

    return image_urls


def numpy_from_base64(s64):
    """Given a base64 decoded image from Google search, convert to a numpy array.

    Args:
        s64 (str): base64 decoded image, starting with header 'data:image/jpeg;base64'
    """

    header, imgdata = s64.split('base64,')
    imgdata = b64decode(imgdata)

    im = Image.open(io.BytesIO(imgdata))
    return np.array(im)


def numpy_from_url(image_url, verbose=False):
    """Given an URL, get the image as a np.array
    """

    # Open the url image, set stream to True, this will return the stream content.
    r = requests.get(image_url, stream = True)

    # Check if the image was retrieved successfully
    if r.status_code == 200:
        # Set decode_content value to True, otherwise the downloaded image file's size will be zero.
        r.raw.decode_content = True

        img = Image.open(io.BytesIO(r.content))
        img = np.array(img)

        if verbose:
            print('Image sucessfully Downloaded: ',filename)
    else:
        img = None
        if verbose:
            print('Image Couldn\'t be retreived')

    return img


def get_numpy_image_from_query(s):
    """Download or decode image as np.array from url or base64 string

    Args:
        s (str): Output string from google_query / URL or base64 encoded image

    Returns:
        img (np.array): Image as numpy array
    """
    is_url = 'http' in s
    if is_url:
        img = numpy_from_url(s)
    else:
        img = numpy_from_base64(s)
    return img


def download_images(image_urls, parent_path, prefix='img', formatter='{:03d}'):
    """Download all images and save in a folder as numbered JPEG's

    Args:
        image_urls (set): Image URLS (or base64 encoded images)
        parent_path (string): Folder to store the images

    Returns:
        failed (list): List of failed items
    """

    if not os.path.isdir(parent_path):
        os.mkdir(parent_path)

    count = 1
    failed = []
    for s in list(image_urls):

        if s:
            try:
                img = get_numpy_image_from_query(s)
                img = Image.fromarray(img)
                filename = os.path.join(parent_path,
                                        prefix + \
                                        formatter.format(count) + \
                                        '.jpeg')
                img.save(filename)
                count += 1

            except:
                failed.append(s)

    return failed



def download_images(image_urls, parent_path, prefix='img', formatter='{:03d}'):
    """Download all images and save in a folder as numbered JPEG's

    Args:
        image_urls (set): Image URLS (or base64 encoded images)
        parent_path (string): Folder to store the images

    Returns:
        failed (list): List of failed items
    """

    if not os.path.isdir(parent_path):
        os.mkdir(parent_path)

    count = 1
    failed = []
    for s in list(image_urls):

        if s:
            try:
                img = get_numpy_image_from_query(s)
                img = Image.fromarray(img)
                filename = os.path.join(parent_path,
                                        prefix + \
                                        formatter.format(count) + \
                                        '.jpeg')
                img.save(filename)
                count += 1

            except:
                failed.append(s)

    return failed
