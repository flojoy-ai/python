# contains value returned by a node's init function
class NodeInitContainer:
    def __init__(self, value=None):
        self.value = value
    
    def set(self, value):
        self.value = value

    def get(self):
        return self.value