import yaml


def read_config():
    with open('config.yaml', 'r') as stream:
        return yaml.safe_load(stream)
