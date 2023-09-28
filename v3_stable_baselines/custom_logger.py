from stable_baselines3.common.logger import Logger
from stable_baselines3.common import utils
from v3_stable_baselines.custom_callback import CustomCallback
from typing import Any, Optional, Tuple, Union

INFO = 20


def configure_custom_logger(verbose, tensorboard_log, tb_log_name, reset_num_timesteps, callback):
    lg = utils.configure_logger(
        verbose, tensorboard_log, tb_log_name, reset_num_timesteps)
    return CustomLogger(wrapee=lg, callback=callback)


class CustomLogger(Logger):
    def __init__(self, wrapee, callback):
        self.wrapee: Logger = wrapee
        self.callback: CustomCallback = callback

    def record(self, key: str, value: Any, exclude: Optional[Union[str, Tuple[str, ...]]] = None) -> None:
        self.wrapee.record(key=key, value=value, exclude=exclude)
        self.callback.notify_new_record(record={key: value})

    def record_mean(self, key: str, value: Optional[float], exclude: Optional[Union[str, Tuple[str, ...]]] = None) -> None:
        self.wrapee.record_mean(key=key, value=value, exclude=exclude)

    def dump(self, step: int = 0) -> None:
        self.wrapee.dump(step=step)

    def log(self, *args, level: int = INFO) -> None:
        self.wrapee.log(*args, level=level)

    def debug(self, *args) -> None:
        self.wrapee.debug(*args)

    def info(self, *args) -> None:
        self.wrapee.info(*args)

    def warn(self, *args) -> None:
        self.wrapee.warn(*args)

    def error(self, *args) -> None:
        self.wrapee.error(*args)

    # Configuration
    # ----------------------------------------
    def set_level(self, level: int) -> None:
        self.wrapee.set_level(level)

    def get_dir(self) -> Optional[str]:
        return self.wrapee.get_dir()

    def close(self) -> None:
        self.wrapee.close()
