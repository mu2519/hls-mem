import tomllib
from jinja2 import Environment, FileSystemLoader


def custom_merge(dst: dict, src: dict) -> None:
    for k in src:
        if k in dst:
            if isinstance(src[k], str) and isinstance(dst[k], str):
                dst[k] = dst[k] + '\n' + src[k]
            elif isinstance(src[k], list) and isinstance(dst[k], list):
                dst[k] = dst[k] + src[k]
            else:
                raise TypeError
        else:
            dst[k] = src[k]


loader = FileSystemLoader('.')
env = Environment(loader=loader)
template = env.get_template('template.py.jinja')

with open('./cfg/meta.toml', 'rb') as f:
    meta = tomllib.load(f)

for k, v in meta.items():
    if not isinstance(v, list):
        raise TypeError
    cfg = {}
    for n in v:
        if not isinstance(n, str):
            raise TypeError
        with open('./cfg/' + n + '.toml', 'rb') as f:
            custom_merge(cfg, tomllib.load(f))
    src = template.render(cfg)
    with open('./src/' + k + '.py', 'w') as f:
        f.write(src)
