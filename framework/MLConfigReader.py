import os
from configparser import SafeConfigParser


class MLConfigReader():
    def __init__(self):
        self.cfg = None

    def getCfg(self,config_path=None):
        if self.cfg is None:
            if not os.path.exists(config_path):
                raise Exception("Unable to find config file at path - %s", config_path)
        self.cfg = SafeConfigParser(os.environ)
        self.cfg.read(config_path)
        return self.cfg

MLConfigReaderObj = MLConfigReader()

