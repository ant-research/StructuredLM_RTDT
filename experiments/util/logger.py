import logging
LOGGING_NAMESPACE = "span-rep"
def configure_logger(log_file):
    """
    Simple logging configuration.
    """

    # Create logger.
    logger = logging.getLogger(LOGGING_NAMESPACE)
    logger.setLevel(logging.INFO)

    # Create file handler.
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)

    # Also log to console.
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.propagate = False
    # HACK: Weird fix that counteracts other libraries (i.e. allennlp) modifying the global logger.
    # if len(logger.parent.handlers) > 0:
    #     logger.parent.handlers.pop()

    return logger


def get_logger():
    return logging.getLogger(LOGGING_NAMESPACE)