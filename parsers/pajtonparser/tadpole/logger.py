import logging


logging.basicConfig(
    filename='pond.log',
    filemode="a",
    format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s", 
    datefmt='%H:%M:%S',
    level=logging.DEBUG,
)

MAIN_LOGGER = logging.getLogger('MAIN-PONDLOG')
POND_LOGGER = logging.getLogger('POND-PONDLOG')
MODEL_LOGGER = logging.getLogger('MODEL-PONDLOG')