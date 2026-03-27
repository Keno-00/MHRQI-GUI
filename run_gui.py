import multiprocessing

from mhrqi.gui.main import launch_gui


if __name__ == "__main__":
    multiprocessing.freeze_support()
    launch_gui()
