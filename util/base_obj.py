
class BaseObj:
    def __init__(self, indent: str):
        self._indent = indent

    def __str__(self):
        s = ''
        for att in dir(self):
            if att[0] != '_':
                s += '\n%s%s: %s' % (self._indent, att, getattr(self, att))
        return s
