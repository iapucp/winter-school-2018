import requests, os
from PIL import Image
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from functools import partial

def download_one(x, path):
    ok = 0

    try:
        r = requests.get(x, timeout=30)
    except Exception:
        return ok

    if (r.status_code == requests.codes.ok):
        fn = x.split('/')[-1]
        img_file = path /fn
        with open(img_file, 'wb') as f:
            f.write(r.content)

        try:
            Image.open(img_file)
        except Exception:
            img_file.unlink()
        else:
            ok = 1
    
    r.close()
    return ok

def download_all(path, cls):
    print(f'Downloading {cls.stem}...')
    p = path / cls.stem
    p.mkdir(exist_ok=True)
    lines = cls.open().readlines()
    urls = {i.strip() for i in lines if len(i.split('.')[-1]) in [3,4]}
    print(f'Urls validos: {len(urls)}')
    urls = [i for i in urls if i.split('/')[-1] not in os.listdir(p)]
    print(f'Imagenes a bajar: {len(urls)}')
    
    with ThreadPoolExecutor(16) as ex:
        res = list(ex.map(partial(download_one, path=p), urls))
        
    print(f'Imagenes bajadas: {sum(res)}')
    print(f'Total imagenes: {len(list(p.iterdir()))}')
    print()

def download(path):
    p = Path(path)
    p.mkdir(exist_ok=True)
    urls_files = [f for f in p.iterdir() if f.suffix == '.txt']
    print(f'{len(urls_files)} clases encontradas: {", ".join([e.stem for e in urls_files])}.')
    for c in urls_files: download_all(p, c)
    