#### Network MetaClass ####

import os
import yaml
import torch
import torch.nn as nn

class MetaNetwork(type):

    components = ["linear", "relu", "batchnorm"]

    def __new__(
        cls,
        name,
        bases,
        namespace
    ):

        def _init(
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

            for base in bases:
                base.__init__(self)

        def _forward(
            self,
            x
        ):
            x = torch.tensor(x)
            for component in self.components:
                x = component(x)
            return x

        namespace["__init__"] = _init
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
        components.append("")

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

            elif ("batchnorm" in value):

                d_in = -1
                d_out = -1
                dins.append(d_in)
                douts.append(d_out)
                components.append("batchnorm")
        
        dins.append(self.d_out)
        douts.append(-1)
        components.append("")

        for idx in range(1, len(components)-1):

            din = MetaNetwork._match(douts[idx-1], dins[idx])
            douts[idx-1] = dins[idx] = din

            if (components[idx] in ["relu", "batchnorm"]):
                din = MetaNetwork._match(douts[idx], dins[idx])
                douts[idx] = dins[idx] = din

        for idx in reversed(range(1, len(components)-1)):

            dout = MetaNetwork._match(douts[idx], dins[idx+1])
            douts[idx] = dins[idx+1] = dout

            if (components[idx] in ["relu", "batchnorm"]):
                dout = MetaNetwork._match(douts[idx], dins[idx])
                douts[idx] = dins[idx] = dout
            
        for idx in range(1, len(components)-1):

            if (components[idx] == "linear"):
                component = nn.Linear(
                    in_features = dins[idx],
                    out_features = douts[idx]
                )
                components[idx] = component

            elif (components[idx] == "relu"):
                component = nn.ReLU()
                components[idx] = component
            
            elif (components[idx] == "batchnorm"):
                component = nn.BatchNorm1d(
                    num_features = dins[idx]
                )
                components[idx] = component

        return components[1:-1]

    def _match(
        d1,
        d2
    ):
        if (d1 > 0 and d2 > 0):
            assert(d1 == d2)
            return d1
        elif (d1 > 0 and d2 <= 0):
            return d1
        elif (d1 <= 0 and d2 > 0):
            return d2
        else:
            return -1
