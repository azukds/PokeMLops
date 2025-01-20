from sklearn import model_selection
from sklearn.linear_model import LogisticRegression

import vetiver

import polars as pl
import polars.selectors as cs

# Load training data
pokemon = pl.read_csv(
    "https://gist.githubusercontent.com/armgilles/194bcff35001e7eb53a2a8b441e8b2c6/raw/92200bc0a673d5ce2110aaad4544ed6c4010f687/pokemon.csv"
).drop('#')

X_train, X_test, y_train, y_test = model_selection.train_test_split(
    pokemon.select(cs.all() - cs.matches('Legendary') - cs.string()),
    pokemon["Legendary"],
    test_size=0.2
)

# fit model
poke_fit = (
    LogisticRegression().
        fit(X_train,
            y_train))

# create vetiver model
pokemon = vetiver.VetiverModel(poke_fit,
                               ptype_data=X_train.to_pandas(),
                               model_name="pokemon")


myapp = vetiver.VetiverAPI(pokemon, check_ptype=True)

# next, run myapp.run() to start API and see visual documentation
# create app.py file that includes pinned VetiverAPI to be deployed

myapp.run()