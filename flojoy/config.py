class FlojoyConfig:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = FlojoyConfig()
        return cls._instance

    def __init__(self):
        self.is_offline = False
        self.print_on = True


def set_offline():
    """
    Sets the is_offline flag to True, which means that results will not be sent to the backend via HTTP.
    Mainly used for precompilation
    """
    FlojoyConfig.get_instance().is_offline = True


def set_online():
    """
    Sets the is_offline flag to False, which means that results will be sent to the backend via HTTP.
    """
    FlojoyConfig.get_instance().is_offline = False


def set_print_on():
    """
    Sets the print_on flag to True, which means that the print statements will be executed.
    """
    FlojoyConfig.get_instance().print_on = True


def set_print_off():
    """
    Sets the print_on flag to False, which means that the print statements will not be executed.
    """
    FlojoyConfig.get_instance().print_on = False
