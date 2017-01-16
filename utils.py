import time


class Timer(object):

    def __init__(self, start=True):
        self.end_time = None
        if start:
            self.start()
        else:
            self.st_time = None

    def start(self):
        self.st_time = time.perf_counter()

    def end(self):
        self.end_time = time.perf_counter()
        return self.time

    @property
    def time(self):
        return self.end_time - self.st_time
