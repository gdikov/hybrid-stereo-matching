import yaml


def load_params(filename):
    with open(filename, 'r') as f:
        params = yaml.load(f)
    return params


def save_params(params, filename):
    with open(filename, 'w') as f:
        yaml.dump(params, f, default_flow_style=False)
