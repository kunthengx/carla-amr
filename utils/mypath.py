
import os


class MyPath(object):
    @staticmethod
    def resolve_dataset_file(database, filename):
        """Resolve explicit paths before looking up a bare dataset filename."""
        path = os.path.expanduser(os.fspath(filename))
        if os.path.isabs(path) or os.path.dirname(path):
            return os.path.abspath(path)
        return os.path.join(MyPath.db_root_dir(database), path)

    @staticmethod
    def db_root_dir(database=''):
        db_names = {'amr'}
        assert(database in db_names)

        if database == 'amr':
            preferred = '/home/kunthengx/Documents/CARLA/anomaly-injection-v2-complete/sample_output/'
            if os.path.exists(preferred):
                return preferred
            # Fallback to datasets folder inside the repository
            repo_path = os.getcwd()
            fallback = os.path.join(repo_path, 'sample_output')
            return fallback
        else:
            raise NotImplementedError
