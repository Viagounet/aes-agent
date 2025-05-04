import importlib.resources

class Environment:
    def __init__(self):
        self.turn = 0
        self._mcp_server_script = str(importlib.resources.files("aes_agent").joinpath("mcp/servers/default.py"))

    @property
    def is_running(self) -> bool:
        """
        This property is used to tell us if the environment should be kept alive or not"
        """
        return True

    def __repr__(self):
        return self.__class__.__name__

class BrowsingEnvironment(Environment):
    def __init__(self, **kwargs):
        super().__init__()
        self.max_turns = 5
        if "max_turns" in kwargs:
            self.max_turns = kwargs["max_turns"]
        self._mcp_server_script = str(importlib.resources.files("aes_agent").joinpath("mcp/servers/local_search.py"))

    @property
    def is_running(self) -> bool:
        if self.turn >= self.max_turns:
            return False
        return True