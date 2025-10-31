def log_info(content: str):
    with open("log.txt", "a") as file:
        file.write(content + '\n')
