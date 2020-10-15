import os
from datetime import datetime
from pathlib import Path

import psutil


class Log:
    """
    This class is used to print messages to both the console and a log file.
    The filename of the log is: "Log_" + date-stamp + time-stamp + ".txt"
    Use the method log_print whenever printing something to both console and
    logfile.
    """

    def __init__(self, directory, prefix='log_', save_log=True):

        # skip initialization if log does not have to be saved
        self.save_log = save_log
        if not save_log:
            return

        # get current date and time as string
        time_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        # define log path
        if isinstance(directory, str):
            directory = Path(directory)
        self.log = directory.joinpath(prefix + time_str + '.txt')

        # create log dir if it does not exist yet
        if not directory.exists():
            directory.mkdir()

        # sanity check that log file does not exist yet
        if self.log.exists():
            raise Exception('ERROR: log-file already exists')

        # create file
        self.logprint('log created')

        # sanity check that it now exists
        if not self.log.exists():
            raise Exception('ERROR: could not create log file, '
                            'check permissions')

    @staticmethod
    def _timestamp():
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3] + ' |   '

    @staticmethod
    def _indent():
        return ' ' * 23 + ' |   '

    def logprint(self, msg):

        # print msg to console
        print(msg)

        # return if log does not to be saved
        if not self.save_log:
            return

        # open log
        with open(self.log, 'a') as log:

            # set flag to print the time stamp
            flag_timestamp = True

            # loop through each line in the msg
            for line in msg.splitlines():

                # skip line when empty
                if len(line) is 0:
                    continue

                # write either the timestamp or an indent
                if flag_timestamp:
                    log.write(self._timestamp())
                    flag_timestamp = False
                else:
                    log.write(self._indent())

                # write the line
                log.write(line + '\n')

    @staticmethod
    def memory_usage() -> str:
        mem = psutil.virtual_memory()
        used_sys = '{:,}MB'.format(int(mem.used / 1024 ** 2))
        used_py = '{:,}MB'.format(
            int(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)
        )
        free = '{:,}MB'.format(int(mem.free / 1024 ** 2))
        pct = '%.1f' % mem.percent
        string = 'used_sys=%s, used_py=%s, free=%s, pct=%s' % \
                 (used_sys, used_py, free, pct)
        return string
