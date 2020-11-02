import torch
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

    class Settings:
        def __init__(self,
                     directory: [Path, str, None],
                     save_log: bool = False,
                     filename_prefix: str = 'log_'):
            self.directory = directory
            self.save_log = save_log
            self.filename_prefix = filename_prefix

    def __init__(self, settings: [Settings, None]):

        # use default settings if None is given
        settings = self.Settings(None) if settings is None else settings

        # ATTRIBUTE
        self.settings = settings

        # skip initialization if log does not have to be saved
        if not settings.save_log:
            return

        # get current date and time as string
        time_str = self.date_time()

        # define log path
        if isinstance(settings.directory, str):
            settings.directory = Path(settings.directory)
        log = settings.directory.joinpath(
            settings.filename_prefix + time_str + '.txt'
        )

        # create log dir if it does not exist yet
        if not settings.directory.exists():
            settings.directory.mkdir()

        # sanity check that log file does not exist yet
        if log.exists():
            raise Exception('ERROR: log-file already exists')

        # ATTRIBUTE
        self.log = log

        # create file
        self('log created')

        # sanity check that it now exists
        if not self.log.exists():
            raise Exception('ERROR: could not create log file, '
                            'check permissions')

    @staticmethod
    def _timestamp() -> str:
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3] + ' |   '

    @staticmethod
    def date_time(time_format: str = '%Y-%m-%d_%H-%M-%S.%f') -> str:
        return datetime.now().strftime(time_format).replace('.', '_')

    @staticmethod
    def _indent():
        return ' ' * 23 + ' |   '

    def __call__(self, msg):

        # print msg to console
        print(msg)

        # return if log does not to be saved
        if not self.settings.save_log:
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
    def memory_usage(indent: str = '') -> str:
        s = indent + 'cpu: %s\n' % Log.cpu_memory_usage()
        s += indent + 'gpu: %s' % Log.gpu_memory_usage()
        return s

    @staticmethod
    def cpu_memory_usage() -> str:
        f = Log.format_byte
        mem = psutil.virtual_memory()
        rss = psutil.Process(os.getpid()).memory_info().rss
        s = 'total=%s, used_sys=%s, used_py=%s, free=%s, pct=%.1f%%' % \
            (f(mem.total), f(mem.used), f(rss), f(mem.free), mem.percent)
        return s

    @staticmethod
    def gpu_memory_usage() -> str:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            properties = torch.cuda.get_device_properties(device)
            name = properties.name
            total = properties.total_memory
            alloc = torch.cuda.memory_allocated(device)
            cache = torch.cuda.memory_reserved(device)
            free_cache = total - cache  # free for cache
            free_alloc = total - alloc  # free for tensors
            f = Log.format_byte
            s = ''
            s += '(%s) ' % name
            s += 'total=%s, ' % f(total)
            s += 'allocated=%s (%.1f%%), ' % (f(alloc), 100 * alloc / total)
            s += 'cached=%s (%.1f%%), ' % (f(cache), 100 * cache / total)
            s += 'free_cache=%s (%.1f%%), ' % \
                 (f(free_cache), 100 * free_cache / total)
            s += 'free_alloc=%s (%.1f%%)' % \
                 (f(free_alloc), 100 * free_alloc / total)
        else:
            s = '(not using CUDA)'
        return s

    @staticmethod
    def format_byte(size):
        """
        Format number (bytes) to string such that it is easily readible for
        humans.
        """
        units = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']
        unit = units[0]
        for unit_new in units[1:]:
            if size < 1024:
                break
            unit = unit_new
            size /= 1024
        return '%.2f %s' % (size, unit)

    @staticmethod
    def format_num(number):
        return '{:,}'.format(number)
