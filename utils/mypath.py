
import os


class MyPath(object):
    @staticmethod
    def db_root_dir(database=''):
        db_names = {'amr'}
        assert(database in db_names)

        if database == 'amr':
            preferred = '/datasets/amr'
            if os.path.exists(preferred):
                return preferred
            # Fallback to datasets folder inside the repository
            repo_path = os.getcwd()
            fallback = os.path.join(repo_path, 'datasets', 'amr')
            return fallback
        else:
            raise NotImplementedError

