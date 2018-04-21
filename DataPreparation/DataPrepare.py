from Connectors.PyCsvConnector import csvConnector
import logging


def getData(datapath):
    logger = logging.getLogger(__name__)
    logger.info("Fetching Data for Training")
    if datapath is None:
        raise Exception("No Data Path provided in the config")
    data = csvConnector(datapath)
    return data