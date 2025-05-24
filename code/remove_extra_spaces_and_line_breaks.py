def __remove_extra_spaces_and_line_breaks__(self, text: str) -> str:
    processed = ""
    # если ячейка текста пуста возвращаем пустую строку
    if type(text) != str or len(text) == 0:
        return ""
    flag = True
    for symb in text:
        # при встрече пробела или разрыва строки пропускаем все
        # последующие пробелы или разрывы строки
        if flag and (symb == " " or symb == "\n"):
            processed += " "
            flag = False
        # пока не встретим отличающийся символ
        if symb != " " and symb != "\n":
            flag = True
        if flag:
            processed += symb
    return processed.strip()