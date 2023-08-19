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

    


# class LogPipeSubProcessing:
#     """A context manager that creates a pipe which can be written to by the subprocessing
#     module and read from by the logging module. This is intended to capture and redirect logs
#     from a subprocess to the logging module.
# 
#     Example usage:
#     ```
#     with logpipe.LogPipe(logging.INFO) as logpipe_stdout, logpipe.LogPipe(logging.ERROR) as logpipe_stderr:
#         with subprocess.Popen(
#             command=["python", "-m", "pip", "install", "flytekit"],
#             stdout=logpipe_stdout,
#             stderr=logpipe_stderr
#         ) as proc:
#             pass
#         # Here the logs are available in the logpipe_stdout.buffer and logpipe_stderr.buffer.
#         # This is useful for exception handling, so that we can accompany a raised exception with the logs that led to it.
#         captured_stdout = logpipe_stdout.buffer.getvalue()
#         captured_stderr = logpipe_stderr.buffer.getvalue()
#     ```
#     """
# 
#     def __init__(self, level: int):
#         """Setup the object with a logger and a log level.
# 
#         Args:
#             level: The log level to use for the captured logs.
#         """
#         self.level = level
#         self.fdRead, self.fdWrite = os.pipe()
#         self.pipeReader = os.fdopen(self.fdRead, mode="rb")
#         self.thread = threading.Thread(target=self.run, daemon=True)
#         self.buffer = io.BytesIO()
#         self.closed = False
# 
#     def __enter__(self):
#         """Start the thread when entering the context."""
#         self.thread.start()
#         return self
# 
#     def __exit__(self, exc_type, exc_value, traceback):
#         """Ensure everything is closed and terminated when exiting the context."""
#         self.close()
#         self.thread.join(2.0)
#     
#     def fileno(self):
#         """Return the write file descriptor of the pipe."""
#         return self.fdWrite
# 
#     def run(self):
#         """Log everything that comes from the pipe."""
#         while not self.closed or self.pipeReader.peek():
#             byte_data = self.pipeReader.readline()
#             if byte_data:
#                 logging.log(self.level, byte_data.decode("utf-8").rstrip("\n"))
#                 self.buffer.write(byte_data)
#         self.pipeReader.close()
# 
#     def close(self):
#         """Close the write end of the pipe."""
#         os.close(self.fdWrite)
#         self.closed = True
#         self.thread.join()
# # 
# class LogPipeWriterMultiprocessing:
#     def __init__(self, child_conn: multiprocessing.connection.Connection):
#         self.child_conn = child_conn
# 
#     def write(self, s):
#         self.child_conn.send_bytes(s.encode("utf-8"))
# 
# class LogPipeMultiProcessing:
#     def __init__(self, level: int):
#         """Setup the object with a logger and a log level.
# 
#         Args:
#             level: The log level to use for the captured logs.
#         """
#         self.level = level
#         self.parent_conn, self.child_conn = multiprocessing.Pipe(duplex=False)
#         self.parent_conn_lock = multiprocessing.Lock()
#         self.pipe_writer = LogPipeWriterMultiprocessing(self.child_conn)
#         self.thread = threading.Thread(target=self.run, daemon=True)
#         self.buffer = io.StringIO()
#         self.logger = logging.getLogger("TEST_PIPE")
#         self.logger.setLevel(self.level)
#         self.logger.addHandler(logging.StreamHandler(self.buffer))
# 
#     def run(self):
#         """Log everything that comes from the pipe."""
#         while True:
#             sleep(0.01)
#             with self.parent_conn_lock:
#                 if(self.parent_conn.closed):
#                     break
#                 if(not self.parent_conn.poll()):
#                     continue
#                 byte_data = self.parent_conn.recv_bytes()
#                 if(byte_data != b'\n'):
#                     self.logger.log(self.level, byte_data.decode("utf-8"))
#     
#     def __enter__(self):
#         """Start the thread when entering the context."""
#         self.thread.start()
#         return self
#     
#     def __exit__(self, exc_type, exc_value, traceback):
#         """Ensure everything is closed and terminated when exiting the context."""
#         self.close()
#     
#     def fileno(self):
#         """Return the write file descriptor of the pipe."""
#         return self.child_conn.fileno()
# 
#     def close(self):
#         while True:
#             # Grab the lock and check if there is data in the pipe
#             with self.parent_conn_lock:
#                 # If there is no data, close the pipe
#                 if not self.parent_conn.poll():
#                     self.parent_conn.close()
#                     break
#             # Else just free the lock and try again in a bit
#             sleep(0.01)
# 
# 
# 