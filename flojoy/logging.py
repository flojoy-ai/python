from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
import logging
import io
import multiprocessing
import multiprocessing.connection
import threading
import os
from time import sleep
from typing import TextIO 



# Abstract Pipe class
class ReadWritePipe(ABC):

    @abstractmethod
    def read_is_empty(self) -> bool:
        pass

    @abstractmethod
    def close_reader(self):
        pass

    @abstractmethod
    def read_is_closed(self) -> bool:
        pass

    @abstractmethod
    def read(self) -> bytes:
        pass

    @abstractmethod
    def close_writer(self):
        pass

    @abstractmethod
    def write_is_closed(self) -> bool:
        pass

    @abstractmethod
    def writer_fileno(self) -> int:
        pass

    @abstractmethod
    def get_writer(self) -> TextIO:
        pass


class FileDescriptorReadWritePipe(ReadWritePipe):
    def __init__(self):
        super().__init__()
        self.fd_read, self.fd_write = os.pipe()
        self.reader = os.fdopen(self.fd_read, mode="rb")
        self._writer_is_closed = False
    
    def read_is_empty(self) -> bool:
        return not self.reader.peek()
    
    def read_is_closed(self) -> bool:
        return self.reader.closed

    def write_is_closed(self) -> bool:
        return self._writer_is_closed
    
    def close_writer(self):
        os.close(self.fd_write)
        self._writer_is_closed = True

    def close_reader(self):
        self.reader.close()
    
    def writer_fileno(self) -> int:
        return self.fd_write

    def get_writer(self) -> TextIO:
        return os.fdopen(self.fd_write, mode="wb")
    
    def read(self) -> bytes:
        return self.reader.readline()


    


class MPSpawnReadWritePipe(ReadWritePipe):

    @dataclass
    class _MPWriter:
        write_conn: multiprocessing.connection.Connection

        def write(self, data: bytes):
            self.write_conn.send_bytes(data.encode("utf-8"))

    def __init__(self) -> None:
        super().__init__()
        self.read_conn, self.write_conn = multiprocessing.get_context("spawn").Pipe(duplex=False)
        self._writer_is_closed = False
    
    def read_is_empty(self) -> bool:
        return not self.read_conn.poll()

    def read_is_closed(self) -> bool:
        return self.read_conn.closed
    
    def close_reader(self):
        return self.read_conn.close()
    
    def read(self) -> bytes:
        try:
            return self.read_conn.recv_bytes()
        except EOFError:
            return b''
    
    def write_is_closed(self) -> bool:
        return self._writer_is_closed
    
    def close_writer(self):
        self.write_conn.close()
        self._writer_is_closed = True

    def writer_fileno(self) -> int:
        return self.write_conn.fileno()

    def get_writer(self) -> TextIO:
        return MPSpawnReadWritePipe._MPWriter(write_conn=self.write_conn)
    
class LogPipeMode(Enum):
    MP_SPAWN = auto()
    SUBPROCESS = auto()

class LogPipe:
    def __init__(self, logger: logging.Logger, log_level: int, mode: LogPipeMode):
        if(mode == LogPipeMode.MP_SPAWN):
            self.pipe = MPSpawnReadWritePipe()
        elif(mode == LogPipeMode.SUBPROCESS):
            self.pipe = FileDescriptorReadWritePipe()
        else:
            raise ValueError(f"Invalid mode {mode}, expected one of {LogPipeMode.__members__}")
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.buffer = io.StringIO()
        self.logger = logger
        self.logger.addHandler(logging.StreamHandler(self.buffer))
        self.log_level = log_level
        self.pipe_read_lock = threading.Lock()

    def log_from_pipe(self, data: bytes):
        if(data != b'\n'):
            self.logger.log(self.log_level, data.decode("utf-8").rstrip("\n"))

    def run(self):
        """Log everything that comes from the pipe."""
        while True:
            # Check if the write end is closed
            if(self.pipe.write_is_closed()):
                # Read until empty, then release then lock
                while(data := self.pipe.read()):
                    self.log_from_pipe(data)
                return
            # Check if not empty and read
            if(not self.pipe.read_is_empty()):
                # Acquire the lock to ensure the write end isn't closed at this time
                with self.pipe_read_lock:
                    data = self.pipe.read()
                self.log_from_pipe(data)


    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # Close the write end of the pipe
        with self.pipe_read_lock:
            self.pipe.close_writer()
        # Wait for the thread to finish and flush any remaining logs
        # TODO(roulbac): Daemon thread, do we really need to join?
        self.thread.join()
        # Close the read end
        self.pipe.close_reader()
    
    def fileno(self) -> int:
        return self.pipe.writer_fileno()