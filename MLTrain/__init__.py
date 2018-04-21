import os
import sys
import logging
import getopt
from framework.MLConfigReader import MLConfigReaderObj
import pandas as pd
import numpy as np


def main(appentrypoint):
    try:
        opts, args = getopt.getopt(sys.argv[1:], "C:py", ["config="])
    except getopt.GetoptError as err:
        # print help information and exit:
        print str(err)  # will print something like "option -a not recognized"
        sys.exit(2)
    config = None
    for o, a in opts:
        if o in ('-C', '--config'):
            config = a
        else:
            assert False, "Unhandled Option"
    print("Loading Config File for the ML Pipeline")
    cfg = MLConfigReaderObj.getCfg(config)
    print("Config file loaded successfully")
    logging_level = (cfg.get('LOGGING', 'level', fallback='INFO'))
    if logging_level == 'INFO':
        lg_level = 'INFO'
    else:
        lg_level = 'DEBUG'

    logging.basicConfig(level=lg_level,
                        format="%(asctime)s.%(msecs)d - %(module)s - %(funcName)s : %(message)s"
                        )
    logger = logging.getLogger(__name__)
    logger.info("Logger is successfully initialized to %s", lg_level)
    logger.info("Calling the entry point module")

    #Calling appentrypoint
    appentrypoint(cfg)
