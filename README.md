
Pyramid app to render threads on lore.kernel.org as reddit-style posts

Install requirements (python 3.6+)

```bash
pip install -r ./requirements.txt
```

Install the app

```bash
cd src && pip install -e .
```

From src directory, run:

```bash
COOKIE_SECRET=supersecret gunicorn --paste server.ini
```
