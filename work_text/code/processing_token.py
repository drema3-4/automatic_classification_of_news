def __time_processing__(self, text: str) -> str:
    if re.match(r"\d{2}:\d{2}", text):
        return "time"
    else:
        return text.replace(":", "")

def __processing_token__(self, token_lemma: str) -> str:
    return self.__time_processing__(
        self.__remove_extra_spaces_and_line_breaks__(token_lemma)
    )