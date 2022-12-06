import logging


class ColoredLogger(logging.Formatter):
    """
    Colored logs. No messing with prints for annoucing things. Peace.
    Color scheme:
        - grey: DEBUG
        - blue: INFO
        - yellow: WARNING
        - red: ERROR
        - red-bold: CRITICAL
    Usage:
        import custom_logger
        custom_logger.logger.info("message")
    """
    # color scheme
    grey = '\x1b[38;21m'
    blue = '\x1b[38;5;39m'
    yellow = '\x1b[38;5;226m'
    red = '\x1b[38;5;196m'
    bold_red = '\x1b[31;1m'
    reset = '\x1b[0m'

    def __init__(self, fmt):
        super().__init__()
        self.fmt = fmt
        self.FORMATS = {
            logging.DEBUG: self.grey + self.fmt + self.reset,
            logging.INFO: self.blue + self.fmt + self.reset,
            logging.WARNING: self.yellow + self.fmt + self.reset,
            logging.ERROR: self.red + self.fmt + self.reset,
            logging.CRITICAL: self.bold_red + self.fmt + self.reset
        }

    def format(self, record):
        log = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log)
        return formatter.format(record)


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

fmt = '%(levelname)8s >>> %(message)s'
stdout_handler = logging.StreamHandler()
stdout_handler.setLevel(logging.DEBUG)
stdout_handler.setFormatter(ColoredLogger(fmt))

logger.addHandler(stdout_handler)