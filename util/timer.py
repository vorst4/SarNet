from time import time
import settings


class Timer:
    def __init__(self, log):
        self.timer = time()
        self.log = log

    def start(self):
        self.timer = time()
        return self

    def time(self):
        return time() - self.timer

    def stop(self, msg):
        if settings.log_timer:
            self.log.logprint('timer % 8.2f sec | %s' % (self.time(), msg))
        return self
