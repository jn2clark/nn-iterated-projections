# Training neural networks with iterative projection algorithms

Setup some paramters and split the data.
```
nb_epoch = 10
batch_size = 256
k_folds = 5


# get some data
X_train, Y_train, _, _ = core.get_data(60000, 10000, 10)

# split training into folds
X_folds = np.array_split(X_train, k_folds)
Y_folds = np.array_split(Y_train, k_folds)

k = 1
X_train = list(X_folds)
X_test = X_train.pop(k)
X_train = np.concatenate(X_train)
Y_train = list(Y_folds)
Y_test = Y_train.pop(k)
Y_train = np.concatenate(Y_train)


```

Setup the model with some appropriate parameters.
```
# create the model
Model = ModelDM(lr=.001, img_rows=28, img_cols=28,img_channels=1, n_l1=16, n_l2=None, dout=0, n_filt=8, opt='Adam',regularizer=None, verbose=True, seed=1338)

```
Create and compile the model.
```
# create the model
Model.create_model()
```
Train the model using sgd.
```
# fit regular
Model.fit(X_train, Y_train, X_test, Y_test, batch_size, nb_epoch)
score_reg = Model.evaluate(X_test, Y_test)
```
Reset weights and train using difference map.
```
# fit dm
Model.initialize_weights()
_, weights_out, test_loss, test_accuracy, train_loss, train_accuracy, dm_errors = Model.fit_dm(X_train,
                                Y_train, X_test, Y_test, batch_size, nb_epoch, iterations=10,
                                val_lim=.99, n_set=3, early_term=False)
```
Save the trained model and weights.

```
# save a model (no suffix, added when saving)
Model.save_model("/a_model")
```
