from typing import Any, Callable
from flojoy.dao import Dao

from flojoy.types import NodeInitContainer

def node_init(node_id: str, func: Callable[[], Any]) -> None:
    daemon_container = NodeInitService().create_init_store(node_id)
    res = func()
    if res is not None:
        daemon_container.set(res)

class NodeInitService:
    """
    NodeInit - available during node initialization - intended to be used ONLY inside node_init functions
    """
    dao = Dao.get_instance()


    # this method will create the storage used for the node to hold whatever it initialized.
    def create_init_store(self, node_id):
        if self.has_init_store(node_id):
            raise ValueError(f"Storage for {node_id} init object already exists!")
        
        self.dao.set_init_container(node_id, NodeInitContainer())
        return self.get_init_store(node_id)

    # this method will get the storage used for the node to hold whatever it initialized.    
    def get_init_store(self, node_id) -> NodeInitContainer:
        store = self.dao.get_init_container(node_id)
        if store is None:
            raise ValueError(f"Storage for {node_id} init object has not been initialized!")
        return store 

    # this method will check if a node has an init store already created.
    def has_init_store(self, node_id) -> bool:
        return self.dao.has_init_container(node_id)