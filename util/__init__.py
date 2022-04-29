from . import logger
from . import tool
from . import canvas
from . import gui
import configparser

cf = configparser.ConfigParser()
cf.read("./util/config/config.ini")

logger.log_init(root_path = cf.get("root", "main_root_logger"))

