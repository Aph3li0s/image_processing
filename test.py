import threading
import time

class MyThread(threading.Thread):
    def __init__(self, event, next_event, stop_event):
        super().__init__()
        self.event = event
        self.next_event = next_event
        self.stop_event = stop_event

    def run(self):
        while not self.stop_event.is_set():
            self.event.wait()  # Wait until the event is set
            if self.stop_event.is_set():
                break  # Exit loop if stop event is set
            print(f"{self.name} running")
            time.sleep(3)  # Simulating some work
            self.event.clear()  # Clear the event
            self.next_event.set()  # Set the event for the next thread

def main():
    event1 = threading.Event()
    event2 = threading.Event()
    stop_event = threading.Event()

    # Start both events initially
    event1.set()
    print("activate thread 1")
    thread1 = MyThread(event1, event2, stop_event)
    thread2 = MyThread(event2, event1, stop_event)

    thread1.start()
    thread2.start()

    # Simulating the occurrence of events
    event2.set()
    print("activate thread 2")
    time.sleep(2)
    event1.set()
    # Wait for threads to finish
    stop_event.set()  # Set the stop event to terminate threads
    thread1.join()
    thread2.join()

if __name__ == "__main__":
    main()
