import logging
import datetime

def log_init(root_path = "./logger/"):
    logger = logging.getLogger()
        
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
            '[%(levelname)1.1s %(asctime)s %(module)s:%(lineno)d] %(message)s',
            datefmt='%Y%m%d %H:%M:%S')

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.ERROR)
    console_handler.setFormatter(formatter)

    log_filename = datetime.datetime.now().strftime(root_path + "%Y-%m-%d.log")
    
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
if __name__ == "__main__":
    log_init()
    logging.debug('debug')
    logging.info('info')
    logging.warning('warning')
    logging.error('error')
    logging.critical('critical')
