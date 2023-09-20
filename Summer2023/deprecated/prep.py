
# In case openssl library is downloaded but not working as intended for PATH issue,
# this file updates PATH variable to make it work.
# !! Only works for Darwin !!

import platform, subprocess, logging, os


class Prep:
    def __init__(self, should_prep=False):
        logging.basicConfig(level=logging.INFO)
        self._os = platform.system()
        self._is_Darwin = True if self._os == 'Darwin' else False
        self._openssl_version = self.check_openssl()
        self._prev_path = self.check_path()
        self._new_path = self._prev_path

        if should_prep:
            self.update_path()
            self.check_path()
            self.check_openssl()

    def check_openssl(self):
        """
        Checks OpenSSL version, save, log & return the version info.
        :return: str in the form "(SSL Type) (SSL Ver.) (Release Date)"
        """
        if self._is_Darwin:
            try:
                self._openssl_version = subprocess.check_output(['openssl', 'version']).decode('utf-8')
                logging.info(self._openssl_version)
                return self._openssl_version

            except subprocess.CalledProcessError as e:
                print(f"An error occurred: {e}")

    def update_openssl(self):
        """
        Updates OpenSSL version.
        :return: None
        """
        if self._is_Darwin:
            try:
                if ' 3.' not in self._openssl_version:
                    logging.info("Updating OpenSSL...")
                    subprocess.run(['brew', 'upgrade', 'openssl'], check=True)
                    logging.info("Successfully updated.")
                else:
                    logging.info("OpenSSL is up to date.")
                    logging.info(self._openssl_version)

            except subprocess.CalledProcessError as e:
                print(f"An error occurred: {e}")

    def check_path(self):
        """
        Checks PATH variable, log it, and return it.
        :return: str containing "(PATH1):(PATH2): ..."
        """
        if self._is_Darwin:
            try:
                self._new_path = os.environ["PATH"]
                logging.info(self._new_path)
                return self._new_path

            except subprocess.CalledProcessError as e:
                print(f"An error occurred: {e}")

    def update_path(self):
        """
        Updates PATH variable for macOS environment. Pushes anaconda PATH's to last.
        :return: None
        """
        if self._is_Darwin and "anaconda" in self._new_path:
            try:
                logging.info("Updating PATH variable...")
                path_list = self._prev_path.split(":")
                new_path_list = [i for i in path_list if "conda" not in i] + [i for i in path_list if "conda" in i]
                self._new_path = ":".join(new_path_list)
                self._set_path(self._new_path)

            except subprocess.CalledProcessError as e:
                print(f"An error occurred: {e}")

    def revert_path(self):
        if self._is_Darwin:
            self._set_path(self._prev_path)
            self._new_path = self._prev_path

    def _set_path(self, path: str):
        if self._is_Darwin:
            os.environ["PATH"] = path
            logging.info("Successfully updated PATH variable. ")
