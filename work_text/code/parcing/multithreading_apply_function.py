def crawling_pages(off_pc: bool, pages: int) -> None:
    columns = ["url", "date", "title", "summary", "content"]
    news_container1 = pd.DataFrame(columns=columns)
    news_container2 = pd.DataFrame(columns=columns)
    thread1 = threading.Thread(target=__crawling_pages__, args=(0, pages // 2, news_container1, 1))
    thread2 = threading.Thread(target=__crawling_pages__, args=(pages // 2, pages, news_container2, 2))
    thread1.start()
    thread2.start()
    thread1.join()
    thread2.join()
    try:
        news = pd.concat([news_container1, news_container2], ignore_index=True)
        news.to_excel("./news.xlsx")
    except:
        print("Не получилось!")