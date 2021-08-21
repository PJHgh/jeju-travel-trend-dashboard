import os
import re
import time
import logging
import datetime
import argparse
import numpy as np
import pandas as pd

import requests
from bs4 import BeautifulSoup
from selenium import webdriver

class InstaKeywordCrawling():
    def __init__(self, email, password):
        self.email = email
        self.password = password
        
        if not os.path.exists('./data'):
            os.mkdir('./data')

        # options = webdriver.ChromeOptions()
        # options.add_argument("headless")
        # self.driver = webdriver.Chrome('./etc/chromedriver.exe', options=options)
        self.driver = webdriver.Chrome('./etc/chromedriver.exe')
        
    def login(self):
        self.driver.get("https://www.instagram.com")
        time.sleep(3)

        input_id = self.driver.find_elements_by_css_selector('input._2hvTZ.pexuQ.zyHYP')[0]
        input_id.clear()
        input_id.send_keys(self.email)

        input_pw = self.driver.find_elements_by_css_selector('input._2hvTZ.pexuQ.zyHYP')[1]
        input_pw.clear()
        input_pw.send_keys(self.password)
        input_pw.submit()
        time.sleep(3)

    def get_insta_url(self, keyword):
        return f'https://www.instagram.com/explore/tags/{str(keyword)}'

    def search_first(self):
        first = self.driver.find_element_by_css_selector("div._9AhH0")
        first.click()
        time.sleep(5)

    def get_content(self):
        html = self.driver.page_source
        soup = BeautifulSoup(html, 'lxml')

        try:
            contents = soup.select('div.C4VMK > span')
            content = contents[0].text
            rewiew = list(map(lambda x:x.text, contents[1:]))
        except:
            content = ' '
            rewiew = []

        try:
            tags = re.findall(r'#[^\s#,\\]+', content)
        except:
            tags = ''

        try:
            date = soup.select('time._1o9PC.Nzb55')[0]['datetime'][:10]
        except:
            date = ''
        
        try:
            like = int(soup.select('div.Nm9Fw > a.zV_Nj > span')[0].text)
        except:
            like = 0
        
        try: 
            place = soup.select('div.M30cS')[0].text
        except:
            place = ''

        return [content, rewiew, date, like, place, tags]

    def move_next(self):
        right = self.driver.find_element_by_css_selector("body > div._2dDPU.CkGkG > div.EfHg9 > div > div > a._65Bje.coreSpriteRightPaginationArrow")
        right.click()
        time.sleep(5)

    def crawling(self, keyword, num_of_sample=300):
        self.keyword = keyword
        self.num_of_sample = num_of_sample

        url = self.get_insta_url(keyword)
        self.driver.get(url)
        time.sleep(5)

        for i in range(num_of_sample):
            if i == 0:
                self.search_first()
                data = self.get_content()
                self.result = [data]
            else:
                try:
                    self.move_next()
                    data = self.get_content()
                    self.result.append(data)
                except:
                    time.sleep(2)
                    self.move_next()

    def save_result(self):
        now = datetime.datetime.now()
        now = str(now.strftime('%Y%m%d_%H'))
        
        results_df = pd.DataFrame(self.result)
        results_df.columns = ['content', 'rewiew', 'date', 'like', 'place', 'tags']
        results_df.to_csv(f'./data/{self.keyword}_{now}.csv', index=False)

    def new_crawling(self, keyword, num_of_scroll=100):
        self.keyword = keyword
        print(keyword)
        url = self.get_insta_url(keyword)
        self.driver.get(url)
        time.sleep(5)
        
        paths = self.get_paths(num_of_scroll)
        print('crawling success')

        self.result = []
        self.num_of_sample = len(paths)
        print('num_of_sample : ', self.num_of_sample)
        for path in paths:
            self.driver.get(path)
            time.sleep(5)
            
            try:
                data = self.get_content()
                self.result.append(data)
            except:
                time.sleep(2)
        print('scraping success')
            
    def get_paths(self, num_of_scroll):
        SCROLL_PAUSE_SEC = 3

        html = self.driver.page_source
        soup = BeautifulSoup(html, 'lxml')
        paths = self.update_paths([], soup)

        last_height = self.driver.execute_script("return document.body.scrollHeight")
        for count in range(num_of_scroll):
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(SCROLL_PAUSE_SEC)

            html = self.driver.page_source
            soup = BeautifulSoup(html, 'lxml')
            paths = self.update_paths(paths, soup)
            
            now_height = self.driver.execute_script("return document.body.scrollHeight")
            new_height = now_height
            while now_height == last_height:
                new_height -= 100
                self.driver.execute_script(f"window.scrollTo(0, {new_height});")
                time.sleep(1)
                self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                now_height = self.driver.execute_script("return document.body.scrollHeight")
            last_height = new_height
        return list(set(paths))

    def update_paths(self, paths, soup):
        for row in soup.select('div.Nnq7C.weEfm'):
            for column in row.select('div.v1Nh3.kIKUG._bz0w > a'):
                path = 'https://www.instagram.com'+str(column).split(' ')[1][6:-1]
                paths.append(path)
        return paths

def main(keyword_list):
    email = '*********'
    password = '*********'

    insta = InstaKeywordCrawling(email, password)
    insta.login()
    for keyword in list(keyword_list.split(',')):
        insta.new_crawling(keyword)
        insta.save_result()
    insta.driver.quit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--keyword', required=True, type=str, help='tag keyword')
    args = parser.parse_args()

    main(args.keyword)