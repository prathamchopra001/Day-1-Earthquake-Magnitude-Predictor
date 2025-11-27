import sys
import time
import threading
from datetime import datetime, timedelta
from typing import List, Optional, Callable


class ProgressTracker:
    
    def __init__(self, steps: List[str], title: str = "PROGRESS"):
        
        self.steps = steps
        self.title = title
        self.current_step = 0
        self.start_time = None
        self.step_times = []
        
    def start(self):
        self.start_time = time.time()
        self._print_header()
    
    def _print_header(self):
        print("\n" + "=" * 60)
        print(f"  {self.title}")
        print("=" * 60)
        print(f"\n  Total steps: {len(self.steps)}\n")
    
    def next_step(self, extra_info: str = None):
        
        # Record time for previous step
        if self.current_step > 0:
            self.step_times.append(time.time())
        
        self.current_step += 1
        
        if self.current_step > len(self.steps):
            return
        
        # Calculate progress
        progress = self.current_step / len(self.steps)
        bar_width = 30
        filled = int(bar_width * progress)
        bar = "█" * filled + "░" * (bar_width - filled)
        
        # Time info
        elapsed = time.time() - self.start_time
        if self.current_step > 1:
            avg_time = elapsed / (self.current_step - 1)
            remaining = avg_time * (len(self.steps) - self.current_step + 1)
            time_str = f"~{remaining:.0f}s remaining"
        else:
            time_str = ""
        
        # Print step info
        print(f"  [{bar}] {self.current_step}/{len(self.steps)} {time_str}")
        print(f"  → {self.steps[self.current_step - 1]}")
        
        if extra_info:
            print(f"    {extra_info}")
        
        print()
    
    def update_status(self, message: str):
        print(f"    {message}")
    
    def complete(self):
        self.step_times.append(time.time())
        
        total_time = time.time() - self.start_time
        
        print(f"  [{'█' * 30}] {len(self.steps)}/{len(self.steps)}")
        print(f"  ✓ All steps completed!")
        print(f"\n  Total time: {total_time:.2f} seconds")
        print("=" * 60 + "\n")


class SpinnerProgress:
    
    
    FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    
    def __init__(self, message: str = "Processing"):
        
        self.message = message
        self.running = False
        self.thread = None
        self.start_time = None
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False
    
    def start(self):
        """Start the spinner animation."""
        self.running = True
        self.start_time = time.time()
        self.thread = threading.Thread(target=self._animate)
        self.thread.start()
    
    def stop(self, success: bool = True):
        """Stop the spinner animation."""
        self.running = False
        if self.thread:
            self.thread.join()
        
        # Clear line and show result
        elapsed = time.time() - self.start_time
        sys.stdout.write("\r" + " " * 70 + "\r")
        
        if success:
            print(f"  ✓ {self.message} ({elapsed:.2f}s)")
        else:
            print(f"  ✗ {self.message} failed ({elapsed:.2f}s)")
    
    def _animate(self):
        """Animation loop."""
        frame_idx = 0
        while self.running:
            elapsed = time.time() - self.start_time
            frame = self.FRAMES[frame_idx % len(self.FRAMES)]
            sys.stdout.write(f"\r  {frame} {self.message}... ({elapsed:.1f}s)")
            sys.stdout.flush()
            time.sleep(0.1)
            frame_idx += 1


class ProgressBar:
    
    
    def __init__(self, total: int, desc: str = "Progress", width: int = 30):
        
        self.total = total
        self.desc = desc
        self.width = width
        self.current = 0
        self.start_time = time.time()
    
    def update(self, n: int = 1):
        """Update progress by n steps."""
        self.current += n
        self._display()
    
    def _display(self):
        """Display the progress bar."""
        progress = self.current / self.total if self.total > 0 else 0
        filled = int(self.width * progress)
        bar = "█" * filled + "░" * (self.width - filled)
        
        elapsed = time.time() - self.start_time
        
        if self.current > 0:
            rate = self.current / elapsed
            remaining = (self.total - self.current) / rate if rate > 0 else 0
            time_str = f"{elapsed:.0f}s<{remaining:.0f}s"
        else:
            time_str = "0s"
        
        percent = progress * 100
        
        sys.stdout.write(f"\r  {self.desc}: [{bar}] {percent:5.1f}% ({self.current}/{self.total}) {time_str}  ")
        sys.stdout.flush()
    
    def close(self):
        """Close the progress bar."""
        elapsed = time.time() - self.start_time
        print(f"\n  ✓ {self.desc} complete ({elapsed:.2f}s)\n")


def format_time(seconds: float) -> str:
    """Format seconds into human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}m {secs}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"


# Quick test
if __name__ == "__main__":
    print("Testing Progress Utilities\n")
    
    # Test ProgressTracker
    tracker = ProgressTracker([
        "Loading data",
        "Processing features",
        "Training model",
        "Saving results"
    ], title="TEST PIPELINE")
    
    tracker.start()
    
    for i in range(4):
        tracker.next_step()
        time.sleep(0.5)  # Simulate work
    
    tracker.complete()
    
    # Test SpinnerProgress
    print("\nTesting Spinner:")
    with SpinnerProgress("Doing something"):
        time.sleep(2)
    
    # Test ProgressBar
    print("\nTesting Progress Bar:")
    bar = ProgressBar(50, desc="Processing items")
    for i in range(50):
        time.sleep(0.05)
        bar.update()
    bar.close()
    
    print("All tests passed!")