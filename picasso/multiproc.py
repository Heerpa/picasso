"""
    picasso.multiproc
    ~~~~~~~~~~~~~~~~~

    Utilities for multiprocessing

    :authors: Heinrich Grabmayr, 2022
    :copyright: Copyright (c) 2022 Jungmann Lab, MPI of Biochemistry
"""
import logging as _logging
import logging.handlers as _logginghandlers
import multiprocessing as _multiprocessing
import time

def listener_configurer(filename):
    root = _logging.getLogger('mp')
    file_handler = _logginghandlers.RotatingFileHandler(
        filename, maxBytes=1e6, backupCount=5)
    # console_handler = _logging.StreamHandler()
    formatter = _logging.Formatter(
        '%(asctime)s %(processName)-10s %(name)s %(levelname)-8s %(message)s')
    file_handler.setFormatter(formatter)
    # console_handler.setFormatter(formatter)
    root.addHandler(file_handler)
    # root.addHandler(console_handler)
    root.setLevel(_logging.DEBUG)
    return file_handler


def listener_process(queue, logfilename):
    file_handler = listener_configurer(logfilename)
    nmem = 10
    pastmsgs = ['']*nmem
    while True:
        while not queue.empty():
            record = queue.get()
            if isinstance(record, str):
                # if record=='shutdown listener':
                root = _logging.getLogger('mp')
                root.removeHandler(file_handler)
                file_handler.close()
                return
            # logger = _logging.getLogger(record.name)
            # if str(record) not in pastmsgs:
            #     logger.handle(record)  # No level or filter logic applied - just do it!
            # else:
            #     pastmsgs = [str(record)] + pastmsgs[:-1]
            logger = _logging.getLogger('mp')
            if str(record) not in pastmsgs:
                logger.handle(record)  # No level or filter logic applied - just do it!
            else:
                pastmsgs = [str(record)] + pastmsgs[:-1]
        time.sleep(.5)


def start_multiprocesslogging(logfilename):
    logqueue = _multiprocessing.Queue(-1)
    listener = _multiprocessing.Process(
        target=listener_process, args=(logqueue, logfilename))
    listener.start()
    return logqueue, listener


def stop_multiprocesslogging(logqueue, listener):
    logqueue.put('shutdown listener')
    listener.join()


def worker_configurer(queue, index):
    h = _logginghandlers.QueueHandler(queue)  # Just the one handler needed
    root = _logging.getLogger('mp.worker{:d}'.format(index))
    root.addHandler(h)
    # send all messages, for demo; no other level or filter logic applied.
    root.setLevel(_logging.DEBUG)
    return root
