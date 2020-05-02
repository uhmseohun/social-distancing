from dataset import load_data
from model import convnet
from tensorflow.keras.callbacks import ModelCheckpoint

model = convnet()
(x_data, y_data) = load_data()

checkpoint_path =\
  './checkpoint/' + 'Epoch{epoch:04d}_Loss{val_loss:.4f}.h5'

save_callback = ModelCheckpoint(
  filepath=checkpoint_path,
  monitor='val_loss',
  verbose=1,
  save_best_only=False,
  save_weights_only=True,
  mode='auto',
  period=5
)

model.fit(
  x_data, y_data,
  epochs=100,
  callbacks=[save_callback],
  validation_split=0.05,
  shuffle=True
)
