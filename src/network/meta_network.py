#### Network MetaClass ####

import os
import yaml
import torch.nn as nn
class MetaNetwork(type):

    components = ["linear", "relu"]

    def __new__(
        cls,
        name,
        bases,
        namespace
    ):

        def _initialize(
            self,
            d_in,
            d_out
        ):

            self.d_in = d_in
            self.d_out = d_out

            spec_name = MetaNetwork._get_spec_name(
                self = self
            )
            spec = MetaNetwork._load_spec(
                spec_name = spec_name
            )
            self.components = MetaNetwork._get_components(
                self = self,
                spec = spec
            )

        def _forward(
            self,
            x
        ):
            return x

        namespace["__init__"] = _initialize
        namespace["forward"] = _forward
        return super().__new__(cls, name, bases, namespace)

    def _get_spec_name(
        self
    ):
        try:
            spec_name = self.spec
        except:
            raise Exception(f"The class { self.__name__ } must have `spec` property when you designate `metaclass=MetaNetwork`.")
        return spec_name
    
    def _load_spec(
        spec_name
    ):
        ext = ".yaml"
        root = os.path.abspath(os.path.dirname(__file__))
        path = os.path.abspath(os.path.join(root, "config", spec_name)) + ext

        try:
            with open(path, "r") as f:
                spec = yaml.safe_load(f)
        except:
            raise Exception(f"The file { path } was not found: you have to match the name of configuration file with the value of `spec` property.")

        return spec

    def _get_components(
        self,
        spec
    ):

        components = []
        dins = []
        douts = []
        dins.append(-1)
        douts.append(self.d_in)

        # `key` will be ignored
        for key, value in spec.items():

            if ("linear" in value):
                
                d_in = value["linear"]["d_in"]
                d_out = value["linear"]["d_out"]
                dins.append(d_in)
                douts.append(d_out)
                components.append("linear")

            elif ("relu" in value):

                d_in = -1
                d_out = -1
                dins.append(d_in)
                douts.append(d_out)
                components.append("relu")
        
        dins.append(self.d_out)
        douts.append(-1)

        for idx in range(len(components)):

            din = MetaNetwork._match(douts[idx], dins[idx+1])
            douts[idx] = dins[idx+1] = din

            if (components[idx] == "relu"):
                din = MetaNetwork._match(douts[idx+1], dins[idx+1])
                douts[idx+1] = dins[idx+1] = din

        for idx in reversed(range(len(components))):

            dout = MetaNetwork._match(douts[idx+1], dins[idx+2])
            douts[idx+1] = dins[idx+2] = dout

            if (components[idx] == "relu"):
                dout = MetaNetwork._match(douts[idx+1], dins[idx+1])
                douts[idx+1] = dins[idx+1] = dout
            
        for idx in range(1, len(components)):

            if (components[idx] == "linear"):
                component = nn.Linear(
                in_features = dins[idx],
                out_features = douts[idx]
            )
                components[idx] = component

            if (components[idx] == "relu"):
                component = nn.ReLU()
                components[idx] = component

        return components

    def _match(
        prev_d_out,
        next_d_in
    ):
        if (prev_d_out > 0 and next_d_in > 0):
            assert(prev_d_out == next_d_in)
            return prev_d_out
        elif (prev_d_out > 0 and next_d_in <= 0):
            return prev_d_out
        elif (prev_d_out <= 0 and next_d_in > 0):
            return next_d_in
        else:
            return -1
        

