# Pokemon legendary status prediction

A simple app which trains a logistic regression on a training dataset of pokemon
and predicts [legendary status](https://www.serebii.net/pokemon/legendary.shtml).

<img src="eagle.png" alt="eagle" width="35%">

Run with `uv` ([installation](https://docs.astral.sh/uv/getting-started/installation/)):

```bash
uv run --with-requirements requirements.txt  app.py
```

Then visit http://127.0.0.1:8000/ - you should see a swaggerdoc with a predict interface:

![screenshot.png](screenshot.png)