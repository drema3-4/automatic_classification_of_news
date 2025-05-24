def __getPage__(url: str, file_name: str) -> None:
    # получение html кода страницы с помощью библиотеки requests
    r = requests.get(url=url)
    # сохранение полученного кода в текстовый файл
    with open(file_name, "w", encoding="utf-8") as file:
        file.write(r.text)