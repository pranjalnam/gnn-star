import logging
import logging.config
import os


class Logger(object):
    def __init__(self, id: str, loggername: str = None) -> None:
        super().__init__()
        self.id: str = id
        self.loggername = loggername
        self.log_dir: str = f"{os.path.dirname(os.path.abspath(__file__))}/logs"
        log_name: str = f"monitor_{self.id}.log"
        os.makedirs(os.path.join(self.log_dir), exist_ok=True)
        filepath = os.path.join(self.log_dir, log_name)
        logging.config.fileConfig(
            fname=os.path.join(os.path.dirname(os.path.abspath(__file__)), "logger.conf"),
            defaults={"logfilename": filepath},
        )
        self._logger = logging.getLogger(self.loggername)

    def update_id(self, id: str):
        log_name: str = f"monitor_{self.id}.log"
        filepath = os.path.join(self.log_dir, log_name)
        logging.config.fileConfig(
            fname=os.path.join(os.path.dirname(os.path.abspath(__file__)), "logger.conf"),
            defaults={"logfilename": filepath},
        )
        self._logger = logging.getLogger(self.loggername)

    def debug(self, event: str, msg: str) -> None:
        self._logger.debug(f"{self.id},{event},{msg}")

    def info(self, event: str, msg: str) -> None:
        self._logger.info(f"{self.id},{event},{msg}")

    def warn(self, event: str, msg: str) -> None:
        self._logger.warning(f"{self.id},{event},{msg}")

    def error(self, event: str, msg: str) -> None:
        self._logger.error(f"{self.id},{event},{msg}")

    def critical(self, event: str, msg: str) -> None:
        self._logger.critical(f"{self.id},{event},{msg}")
