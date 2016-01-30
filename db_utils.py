import subprocess
import dotenv
import os

class SQL_Connect(object):
    """
    Class for common interactions with a MySQL database. If connecting
    to a db server over SSH, creates an SSH tunnel from host machine on
    port 9990 to the remote on port 3306. Reads the relevant SSH login
    info from a .env file in the same directory as this script.

    If connecting to a database locally, uses the defaults stored
    in the '/etc/mysql/my.cnf' file instead of .env file.
    """
    def __init__(self):
        import MySQLdb
        remote, user, paswd, ssh_user, ssh_key, dbname = self.load_env()
        self.dbname = dbname
        self.host   = '127.0.0.1'

        # first, try connecting locally
        try:
            config = '/etc/mysql/my.cnf'
            self.connection = MySQLdb.connect(host=self.host, db=dbname,
                                              read_default_file=config)

        # otherwise open an ssh_tunnel using details from .env
        except:
            self.ssh_tunnel(ssh_user, remote, ssh_key)
            self.connection = MySQLdb.connect(host=self.host, user=user,
                                              port=9990, passwd=paswd,
                                              db=dbname)
        self.cursor = self.connection.cursor()

    def load_env(self):
        """
        Find .env file in the directory holding this script
        """
        path = os.path.realpath(__file__)
        env  = os.path.join(os.path.dirname(path), '.env')
        dotenv.load_dotenv(env)

        dbname = os.environ.get("DB_NAME")
        remote = os.environ.get("DB_HOST")
        user   = os.environ.get("DB_USER")
        paswd  = os.environ.get("DB_PWD")
        ssh_user = os.environ.get("SSH_USER")
        ssh_key  = os.environ.get("SSH_KEY")
        return remote, user, paswd, ssh_user, ssh_key, dbname

    def ssh_tunnel(self, ssh_user, remote, ssh_key):
        """
        Assumes that the MySQL database is accessible on port 3306 on
        the remote machine.

        Don't forget to close this tunnel when you're finished
        grabbing stuff from the server!
        """
        cmd = ['ssh', '{}@{}'.format(ssh_user, remote), '-i',
               '{}'.format(ssh_key), '-f','-N', '-L', '9990:localhost:3306']
        subprocess.check_call(cmd)

    def kill_tunnel(self):
        call = subprocess.check_call(['killall', 'ssh'])

    def fetch_dict(self):
        """
        Helper function to get a dict of (column :: values)
        pairs from a MySQLdb.cursor query
        """
        data = self.cursor.fetchall()
        if data is None:
            return None
        desc = self.cursor.description
        arr = {}

        for idx, name in enumerate(desc):
            arr[name] = [dd[idx] for dd in data]
        return arr


class Postgres_Connect(object):
    """
    Class for common interactions with a Postgres database. If connecting
    to a db server over SSH, creates an SSH tunnel from host machine on
    port 9990 to the remote on port 5432. Reads the relevant SSH login
    info from a .env file in the same directory as this script.
    """
    def __init__(self):
        import psycopg2
        remote, user, paswd, ssh_user, ssh_key, dbname = self.load_env()
        self.dbname = dbname
        self.host   = remote

        # try connecting on localhost first
        try:
            conn_string = "host='127.0.0.1' dbname='{}' user='{}' "\
                          "password='{}'"\
                          .format(dbname, user, paswd)
            self.connection = psycopg2.connect(conn_string)

        # otherwise open an ssh tunnel using details from .env
        except:
            try:
                print('Unable to connect on localhost, trying an ssh tunnel')
                conn_string = "host='{}' dbname='{}' user='{}' "\
                              "password='{}' port='9990'"\
                              .format(self.host, dbname, user, paswd)
                self.ssh_tunnel(ssh_user, remote, ssh_key)
                self.connection = psycopg2.connect(conn_string)

            except:
                print('Unable to connect via ssh tunnel, killing tunnel')
                self.kill_tunnel()

        self.cursor = self.connection.cursor()


    def load_env(self):
        """
        Find .env file in the directory holding this script
        """
        path = os.path.realpath(__file__)
        env  = os.path.join(os.path.dirname(path), '.env')
        dotenv.load_dotenv(env)

        dbname = os.environ.get("DB_NAME")
        remote = os.environ.get("DB_HOST")
        user   = os.environ.get("DB_USER")
        paswd  = os.environ.get("DB_PWD")
        ssh_user = os.environ.get("SSH_USER")
        ssh_key  = os.environ.get("SSH_KEY")
        return remote, user, paswd, ssh_user, ssh_key, dbname


    def ssh_tunnel(self, ssh_user, remote, ssh_key):
        """
        Assumes that the Postgres database is accessible on port 5432 on
        the remote machine.

        Don't forget to close this tunnel when you're finished
        grabbing stuff from the server!
        """
        cmd = ['ssh', '{}@{}'.format(ssh_user, remote), '-i',
               '{}'.format(ssh_key), '-f','-N', '-L', '9990:localhost:5432']
        subprocess.check_call(cmd)

    def kill_tunnel(self):
        call = subprocess.check_call(['killall', 'ssh'])
