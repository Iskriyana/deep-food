#Thankyou very much for the code https://github.com/deepanprabhu/duckduckgo-images-api

import requests
import re
import json
import time
import logging
from sys import exit
from pathlib import Path
import os

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.disable = True # comment out to enable logging or set False

class ImageURLDownloader:
    url = 'https://duckduckgo.com/'
    filenames = []
    def __init__(self, max_results=None):
        self.max_results = max_results
        self.dest = Path().absolute()/'data'
        self.filename = None
        if not os.path.isdir(self.dest):
            os.mkdir(self.dest)

    def get(self, keywords):
        self.keywords, self.num_images = keywords, 0
        logger.debug("Hitting DuckDuckGo for Token")

        # First make a request to above URL, and parse out the 'vqd'
        # This is a special token, which should be used in the subsequent request
        self.res = requests.post(self.url, data={'q': self.keywords})
        self.searchObj = re.search(r'vqd=([\d-]+)\&', self.res.text, re.M|re.I)

        if not self.searchObj:
            logger.error("Token Parsing Failed !")
            return -1;

        logger.debug("Obtained Token")

        headers = {
            'authority': 'duckduckgo.com',
            'accept': 'application/json, text/javascript, */*; q=0.01',
            'sec-fetch-dest': 'empty',
            'x-requested-with': 'XMLHttpRequest',
            'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_4) ' + \
            'AppleWebKit/537.36 (KHTML, like Gecko) ' + \
            'Chrome/80.0.3987.163 Safari/537.36',
            'sec-fetch-site': 'same-origin',
            'sec-fetch-mode': 'cors',
            'referer': 'https://duckduckgo.com/',
            'accept-language': 'en-US,en;q=0.9',
        }

        params = (
            ('l', 'us-en'),
            ('o', 'json'),
            ('q', self.keywords),
            ('vqd', self.searchObj.group(1)),
            ('f', ',,,'),
            ('p', '1'),
            ('v7exp', 'a'),
        )

        requestUrl = self.url + "i.js"

        logger.debug("Hitting Url : %s", requestUrl)

        while True:
            while True:
                try:
                    res = requests.get(requestUrl, headers=headers,
                            params=params)
                    data = json.loads(res.text)
                    break
                except ValueError as e:
                    logger.debug("Hitting Url Failure - Sleep and Retry: %s",
                            requestUrl)
                    time.sleep(5)
                    continue

            logger.debug("Hitting Url Success : %s", requestUrl)

            self.append_url_to_file(data["results"])

            if self.max_results is not None and \
                    self.num_images >= self.max_results:
                break

            if "next" not in data:
                logger.debug("No Next Page - Exiting")
                break

            requestUrl = self.url + data["next"]

        self.filename = None

    def append_url_to_file(self, objs, verbose=False):
        if self.filename == None:
            self.filename = self.keywords.replace(" ", "_") + ".csv"
            if self.filename not in self.filenames:
                self.filenames.append(self.filename)
        for obj in objs:
            if self.num_images >= self.max_results: break

            if verbose:
                print("Width {0}, Height {1}".format(obj["width"],
                    obj["height"]))
                print("Thumbnail {0}".format(obj["thumbnail"]))
                print("Url {0}".format(obj["url"]))
                print("Title {0}".format(obj["title"].encode('utf-8')))
                print("Image {0}".format(obj["image"]))
                print("__________")

            with open(str(self.dest) + "/" + self.filename, "a") as myfile:
                myfile.write(obj["image"] + "\n")

            self.num_images += 1

