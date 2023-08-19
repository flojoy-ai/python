import logging
import io
import multiprocessing
import multiprocessing.connection
import threading
import os
from time import sleep 


class LogPipeSubProcessing:
    """A context manager that creates a pipe which can be written to by the subprocessing
    module and read from by the logging module. This is intended to capture and redirect logs
    from a subprocess to the logging module.

    Example usage:
    ```
    with logpipe.LogPipe(logging.INFO) as logpipe_stdout, logpipe.LogPipe(logging.ERROR) as logpipe_stderr:
        with subprocess.Popen(
            command=["python", "-m", "pip", "install", "flytekit"],
            stdout=logpipe_stdout,
            stderr=logpipe_stderr
        ) as proc:
            pass
        # Here the logs are available in the logpipe_stdout.buffer and logpipe_stderr.buffer.
        # This is useful for exception handling, so that we can accompany a raised exception with the logs that led to it.
        captured_stdout = logpipe_stdout.buffer.getvalue()
        captured_stderr = logpipe_stderr.buffer.getvalue()
    ```
    """

    def __init__(self, level: int):
        """Setup the object with a logger and a log level.

        Args:
            level: The log level to use for the captured logs.
        """
        self.level = level
        self.fdRead, self.fdWrite = os.pipe()
        self.pipeReader = os.fdopen(self.fdRead, mode="rb")
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.buffer = io.BytesIO()
        self.closed = False

    def __enter__(self):
        """Start the thread when entering the context."""
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Ensure everything is closed and terminated when exiting the context."""
        self.close()
        self.thread.join(2.0)
    
    def fileno(self):
        """Return the write file descriptor of the pipe."""
        return self.fdWrite

    def run(self):
        """Log everything that comes from the pipe."""
        while not self.closed or self.pipeReader.peek():
            byte_data = self.pipeReader.readline()
            if byte_data:
                logging.log(self.level, byte_data.decode("utf-8").rstrip("\n"))
                self.buffer.write(byte_data)
        self.pipeReader.close()

    def close(self):
        """Close the write end of the pipe."""
        os.close(self.fdWrite)
        self.closed = True
        self.thread.join()

class LogPipeWriterMultiprocessing:
    def __init__(self, child_conn: multiprocessing.connection.Connection):
        self.child_conn = child_conn

    def write(self, s):
        self.child_conn.send_bytes(s.encode("utf-8"))

class LogPipeMultiProcessing:
    def __init__(self, level: int):
        """Setup the object with a logger and a log level.

        Args:
            level: The log level to use for the captured logs.
        """
        self.level = level
        self.parent_conn, self.child_conn = multiprocessing.Pipe(duplex=False)
        self.parent_conn_lock = multiprocessing.Lock()
        self.pipe_writer = LogPipeWriterMultiprocessing(self.child_conn)
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.buffer = io.StringIO()
        self.logger = logging.getLogger("TEST_PIPE")
        self.logger.setLevel(self.level)
        self.logger.addHandler(logging.StreamHandler(self.buffer))

    def run(self):
        """Log everything that comes from the pipe."""
        while True:
            sleep(0.01)
            with self.parent_conn_lock:
                if(self.parent_conn.closed):
                    break
                if(not self.parent_conn.poll()):
                    continue
                byte_data = self.parent_conn.recv_bytes()
                if(byte_data != b'\n'):
                    self.logger.log(self.level, byte_data.decode("utf-8"))
    
    def __enter__(self):
        """Start the thread when entering the context."""
        self.thread.start()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        """Ensure everything is closed and terminated when exiting the context."""
        self.close()
    
    def fileno(self):
        """Return the write file descriptor of the pipe."""
        return self.child_conn.fileno()

    def close(self):
        while True:
            # Grab the lock and check if there is data in the pipe
            with self.parent_conn_lock:
                # If there is no data, close the pipe
                if not self.parent_conn.poll():
                    self.parent_conn.close()
                    break
            # Else just free the lock and try again in a bit
            sleep(0.01)


