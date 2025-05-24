import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import time
import threading

def __loading_bar_and_info__(
    start: bool, number_of_steps: int, total_steps: int, number_of_thread: int
) -> None:
    '''¬ывод инфомрации о прогрессе выполнени€ программы.
    start - нужно ли вывести начальную строку;
    number_page - количество спаршенных страниц;
    total_pages - всего стнраниц, которые нужно спарсить;
    miss_count - число новостей, которые не удалось спарсить;
    whitour_whole_content - число новостей, у которых не получилось полностью спарсить контент.'''
    done = int(number_of_steps / total_steps * 100) if int(
        number_of_steps / total_steps * 100
    ) < 100 or number_of_steps == total_steps else 99
    stars = int(
        40 / 100 * done
    ) if int(20 / 100 * done) < 20 or number_of_steps == total_steps else 39
    tires = 40 - stars

    if start:
        stars = 0
        tires = 40
        done = 0

    print("thread{0} <".format(number_of_thread), end="")
    for i in range(stars):
        print("*", end="")

    for i in range(tires):
        print("-", end="")
    print("> {0}% ||| {1} / {2}".format(done, number_of_steps, total_steps))

def __getPage__(url: str, file_name: str) -> None:
    '''ѕолучение html файла страницы.
    url - ссылка на страницу;
    file_name - им€ файла, в который будет сохранена страница.'''
    r = requests.get(url=url)

    with open(file_name, "w", encoding="utf-8") as file:
        file.write(r.text)

def __parse_news__(url: str) -> str:
    '''ѕолучиние полного контента новости.
    url - ссылка но новость.
    ‘ункци€ возвращает полный текст новости.'''
    news_file_name = "news.html"
    __getPage__(url, news_file_name)

    with open(news_file_name, encoding="utf-8") as file:
        src = file.read()

    content = BeautifulSoup(src, "lxml").find("div", class_="main").find(
        "div", class_="post__text"
    ).text.strip()

    return content

def __parse_page__(page_file_name: str, news_container: pd.DataFrame) -> None:
    '''ѕарсинг информации с новостной страницы: ссылка на новость + коротка€ информаци€ о ней.
    page_file_name - им€ файла, в который сохранЄн код страницы;
    news_container - таблица, в которую заноситс€ информаци€ о новости.
    ‘ункци€ также возвращает количество новостей, которые не удалось спарсить
    и количество новостей, полный контент которых спарсить не удалось.'''
    with open(page_file_name, encoding="utf-8") as file:
        src = file.read()

    soup = BeautifulSoup(src, "lxml")

    news = soup.find("div", class_="post")
    for i in range(10):
        try:
            news_day = news.find("div", class_="post-meta__day").text.strip()
        except:
            news_day = ""

        try:
            news_month = news.find("div",
                                   class_="post-meta__month").text.strip()
        except:
            news_month = ""

        try:
            news_year = news.find("div", class_="post-meta__year").text.strip()
        except:
            news_year = ""

        news_date = news_day + "." + news_month + "." + news_year

        try:
            news_name = news.find("h2",
                                  class_="first_child").find("a").text.strip()
        except:
            news_name = ""

        try:
            news_short_content = news.find("p", class_="first_child"
                                          ).find_next_sibling("p").text.strip()
        except:
            news_short_content = ""

        try:
            link = news.find("h2", class_="first_child").find("a").get("href")
            if not link.startswith("https://"):
                link = 'https://www.hse.ru' + link
        except:
            link = ""

        try:
            if link.startswith("https://www.hse.ru/news/"):
                news_content = __parse_news__(link)
        except:
            news_content = ""

        if len(
            news_day + news_month + news_year + news_name + news_short_content +
            news_content
        ) > 0:
            news_container.loc[len(news_container.index)] = [
                link, news_date, news_name, news_short_content, news_content
            ]

        news = news.find_next_sibling("div", class_="post")