import logging
import os


class Logger:
    def __init__(self, file_path):
        self.file_path = file_path

        self.logger = logging.getLogger(file_path)
        self.logger.setLevel(logging.INFO)

        if not self.logger.handlers:
            handler = logging.FileHandler(file_path, encoding='utf-8')

            formatter = logging.Formatter('%(asctime)s - %(message)s',
                                          datefmt='%Y-%m-%d %H:%M:%S')
            handler.setFormatter(formatter)

            self.logger.addHandler(handler)

    def log(self, msg):
        self.logger.info(msg)


# logger = Logger("app.log")
