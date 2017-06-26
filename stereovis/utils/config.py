import yaml


def load_config(filename):
    with open(filename, 'r') as f:
        params = yaml.load(f)
    return params


def save_config(params, filename):
    with open(filename, 'w') as f:
        yaml.dump(params, f, default_flow_style=False)
