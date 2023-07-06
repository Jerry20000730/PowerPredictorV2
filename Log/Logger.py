import sys
import os
 
class Logger(object):
    def __init__(self, fileN="Default.log"):
        self.terminal = sys.stdout
        self.logpath = os.path.dirname(os.path.abspath(__file__))
        self.log = open(os.path.join(self.logpath, fileN), "w")
 
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
 
    def flush(self):
        pass
