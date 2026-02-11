
class GracefulShutdownException(Exception):
    def __init__(self, signal_number: int, message: str = "Graceful shutdown requested"):
        super().__init__(message)
        
        self.signal_number = int(signal_number)