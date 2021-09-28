# Einstellungen fuer AI
# z.B. Batch size, Callbacks fuer fit,....

import platform
import string
import math
import os
import time

class DiUtil:
    # class or static variables

    def __init__(self):
        pass

    # Erstellt Verzeichnis
    @staticmethod
    def make_dir(dir):
        if not os.path.exists(dir):
            os.makedirs(dir)
        return dir

    # prueft ob es sich um eine endliche Zahl handelt
    @staticmethod
    def is_number(s):
        try:
            f = float(s)
            return math.isfinite(f)
        except ValueError:
            return False
        except TypeError:
            return False

    # Erstellt Logger -> rausschreiben in Terminal und Konsole
    @staticmethod
    def get_logger(_logger_name, _dir_list = [], _log_fn_name = 'Log.txt'):
        import logging
        # create logger
        logger = logging.getLogger(_logger_name)
        logger.setLevel(logging.INFO)
        # create file handler which logs even debug messages
        for dir in _dir_list:
            fh = logging.FileHandler(os.path.join(dir, _log_fn_name))
            formatter = logging.Formatter('%(asctime)s - %(message)s')
            fh.setFormatter(formatter)
            # add the handlers to the logger
            logger.addHandler(fh)
        logger.addHandler(logging.StreamHandler())
        return logger

    @staticmethod
    def get_random_string(self, stringLength=10):
        """Generate a random string of fixed length """
        import random
        letters = string.ascii_letters + string.digits
        return ''.join(random.choice(letters) for i in range(stringLength))

    # Python-Script starten (je nachdem python oder python3)
    @staticmethod
    def start_python_script(script_name_with_params, wait = True):
        import subprocess
        print('Start script ' + script_name_with_params)
        ret = os.system('python3 --version')
        if ret == 0:
            process_ps = 'python3 ' + script_name_with_params
        else:
            process_ps = 'python ' + script_name_with_params
        if wait:
            subprocess.run(process_ps, shell=True)
            return None
        else:
            pid = subprocess.Popen(process_ps, shell=True).pid
            return pid

    # Process vorhanden
    @staticmethod
    def process_is_running(pid):
        import subprocess
        if pid is None:
            return False
        if (platform.system() == 'Windows'):
            out = subprocess.check_output(
                ["tasklist", "/fi", "PID eq {}".format(pid)]).strip()
            if str(pid) in out.decode('utf-8', 'ignore'):
                return True
        if (platform.system() == 'Linux'):
            try:
                os.kill(pid, 0)
            except OSError:
                return False
            else:
                return True
        return False

    # Process suchen
    @staticmethod
    def get_pid_by_name(process_name):
        import subprocess
        if (platform.system() == 'Linux'):
            p = subprocess.Popen(['ps', '-A'], stdout=subprocess.PIPE)
            out, err = p.communicate()
            for line in out.splitlines():
                if process_name in line.decode('utf-8', 'ignore'):
                    pid = int(line.split(None, 1)[0])
                    return pid
        return None

    # Process beenden
    @staticmethod
    def process_kill(pid):
        import signal
        import subprocess
        if pid is None:
            return
        if (platform.system() == 'Windows'):
            subprocess.Popen("TASKKILL /F /PID {pid} /T".format(pid=pid))
        if (platform.system() == 'Linux'):
            os.kill(pid, signal.SIGKILL)

    # Internet verfuegbar (insbes. azure)
    @staticmethod
    def internet_connection_available(url = None, timeout = 5):
        import requests
        if url is None:
            url = 'http://azure.microsoft.com'
        try:
            request = requests.get(url, timeout = timeout)
            return True
        except (requests.ConnectionError, requests.Timeout) as exception:
            return False

    # Warte auf Verfuegbarkeit Internet - Achtung(!): kann ewig sein....
    @staticmethod
    def wait_for_internet_connection(url = None):
        while not DiUtil.internet_connection_available(url):
            time.sleep(5)
        return True

    @staticmethod
    def calc_md5(fn):
        import hashlib
        # BUF_SIZE is totally arbitrary, change for your app!
        BUF_SIZE = 65536  # lets read stuff in 64kb chunks!
        md5 = hashlib.md5()
        with open(fn, 'rb') as f:
            while True:
                data = f.read(BUF_SIZE)
                if not data:
                    break
                md5.update(data)
        return md5.hexdigest()



