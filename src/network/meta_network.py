#### Network MetaClass ####

import os
import yaml

class MetaNetwork(type):

    def __new__(
        cls,
        name,
        bases,
        namespace
    ):
        try:
            spec_name = namespace["spec"]
        except:
            raise Exception(f"The class { name } must have `spec` property when you designate `metaclass=MetaNetwork`.")

        spec = MetaNetwork._load_spec(spec_name)
        print(spec_name)
        print(spec)
        return super().__new__(cls, name, bases, namespace)
    
    def _load_spec(
        name
    ):
        ext = ".yaml"
        root = os.path.abspath(os.path.dirname(__file__))
        path = os.path.abspath(os.path.join(root, "config", name)) + ext

        try:
            with open(path, "r") as f:
                spec = yaml.safe_load(f)
        except:
            raise Exception(f"The file { path } was not found: you have to match the name of configuration file with the value of `spec` property.")
            
        return spec


