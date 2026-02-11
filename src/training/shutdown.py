import signal
import threading
from dataclasses import dataclass, field
import sys
from typing import Callable, Optional, Any
from mobilenetv2ssd.core.exceptions import GracefulShutdownException

@dataclass
class ShutdownHandler:
    _event: threading.Event = field(default_factory=threading.Event)
    _signal_number: Optional[int] = field(default= None, init= False)
    _previous_handlers: dict[int, Any] = field(default_factory=dict, init=False)
    
    def _handle_signal(self, signal_number: int, frame: Any):
        self._signal_number = int(signal_number)
        self._event.set()
        
    def register(self):
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError("ShutdownHandler.register() must be called from the main thread")
        
        # Catching Keyboard error
        self._previous_handlers[signal.SIGINT] = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._handle_signal)
        
        # Adding win32 option as well
        if sys.platform != "win32":
            self._previous_handlers[signal.SIGTERM] = signal.getsignal(signal.SIGTERM)
            signal.signal(signal.SIGTERM, self._handle_signal)
            
            
    def unregister(self):
        
        for sig_number, prev in self._previous_handlers.items():
            # Signalling all the handlers
            signal.signal(sig_number, prev)
        
        self._previous_handlers.clear()
        
        
    def is_requested(self):
        return self._event.is_set()
    
    @property
    def signal_number(self):
        return self._signal_number