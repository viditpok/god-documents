import pickle
from pip._internal.operations.freeze import freeze


class PackageUtils:

    def __init__(self):
        pass

    def get_packages(self):
        """
        Do not change. Gets a list of importable packages in the environment.
        """
        package_list = []
        for package in freeze():
            if '@' in package:
                line = package.split('@')
            elif '==' in package:
                line = package.split('==')
            package_list.append(line[0])
            print(package)
        with open('env.pkl', 'wb') as f:
            pickle.dump(package_list, f)
