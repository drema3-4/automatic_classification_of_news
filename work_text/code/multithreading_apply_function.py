def crawling_pages(off_pc: bool, pages: int) -> None:
    columns = ["url", "date", "title", "summary", "content"]
    # создание контейнеров под каждый из потоков
    news_container1 = pd.DataFrame(columns=columns)
    news_container2 = pd.DataFrame(columns=columns)
    # создание потоков
    thread1 = threading.Thread(target=__crawling_pages__, args=(0, pages // 2, news_container1, 1))
    thread2 = threading.Thread(target=__crawling_pages__, args=(pages // 2, pages, news_container2, 2))
    # запуск потоков
    thread1.start()
    thread2.start()
    # ожидание завершения работы потоков
    thread1.join()
    thread2.join()
    # объединение содержимого контейнеров потоков в один
    try:
        news = pd.concat([news_container1, news_container2], ignore_index=True)
        news.to_excel("./news.xlsx")
    except:
        print("Не получилось!")
