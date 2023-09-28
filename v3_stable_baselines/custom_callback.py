from stable_baselines3.common.callbacks import BaseCallback, EvalCallback


class CustomCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, verbose=0, n_steps=1):
        super(CustomCallback, self).__init__(verbose)
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseAlgorithm
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = None  # type: Dict[str, Any]
        # self.globals = None  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger = None  # stable_baselines3.common.logger
        # # Sometimes, for event callback, it is useful
        # # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]
        self.__steps = n_steps
        self.__counter = 0
        self.__last_value_loss = 100
        self.__last_explained_variance = 100

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        has_trained_enough = self.__last_explained_variance > 0.9 and self.__last_explained_variance < 1 and self.__last_value_loss > 0 and self.__last_value_loss < 0.005 and self.num_timesteps > 700000
        return not has_trained_enough

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass

    def notify_new_record(self, record) -> None:
        # print(record)
        explained_variance_key="train/explained_variance"
        value_loss_key="train/value_loss"

        if explained_variance_key in record:
            self.__last_explained_variance = record[explained_variance_key]
            return
        
        if value_loss_key in record:
            self.__last_value_loss = record[value_loss_key]

        # if explained_variance_key in record:
        #     value = record[explained_variance_key]
        #     if value >= 0.98 and value < 1:
        #         self.__counter += 1
            # else:
            #     self.__counter = 0