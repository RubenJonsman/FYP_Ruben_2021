import threading
import winsound

class App:
    def __init__(self):
        self.thread = threading.Thread(target=self.infiniteloop)
        self.begin()

    def begin(self):
        self.thread.start()

    def run(self):
        frequency = 1000
        deli = 100
        winsound.Beep(frequency, deli)

    def infiniteloop(self):
        App.run(self)



