import tensorflow as tf
import numpy as np
from tensorflow import keras


def get_dataset_from_csv(csv_file_path,
        csv_header,
        target,
        #columns_defaults,
        batch_size=128, shuffle=False):
    dataset = tf.data.experimental.make_csv_dataset(
        csv_file_path,
        batch_size=batch_size,
        column_names=csv_header,
        #column_defaults=columns_defaults,
        label_name=target,
        num_epochs=1,
        header=True,
        shuffle=shuffle,
    )
    return dataset.cache()


def get_x_or_y_axis(x):
    xmin = np.min(x)
    xmax = np.max(x)
    xticks = []
    xticks_label = []
    for m in np.arange(int(xmin), int(xmax)):
        for k in np.arange(1, 10):
            xticks.append(np.power(10, float(m))*k)
            if k == 1:
                if (np.power(10, float(m))*k) >= 1:
                    xticks_label.append(int(np.power(10, float(m))*k))
                else:
                    xticks_label.append(np.power(10, float(m))*k)
            else:
                xticks_label.append('')
    xticks.append(np.power(10, float(xmax)))
    xticks_label.append(int(np.power(10, float(xmax))))
    return [xticks, xticks_label]


def calculate_target_class_weighs(target):

    counts = np.bincount(target.values.reshape(-1))
    print(
    "Number of positive samples in training data: {} ({:.2f}% of total)".format(
        counts[1], 100 * float(counts[1]) / len(target.values.reshape(-1))
    )
    )

    weight_for_0 = 1.0 / counts[0]
    weight_for_1 = 1.0 / counts[1]
    class_weight = {0: weight_for_0, 1: weight_for_1}
    return class_weight


def run_experiment(model,
                   train_dataset,
                   test_dataset,
                   num_epochs = 100):

    learning_rate = 0.01
    
    first_decay_steps = 10000
    lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
          learning_rate,
          first_decay_steps)

    model.compile(
        #optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        #loss=keras.losses.MeanSquaredError(),
        loss=tf.keras.losses.Huber(delta=10.0),
    )

    early_stopping_monitor = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=40,
        verbose=0,
        mode='auto',
        baseline=None,
        restore_best_weights=True
    )

    print("Start training the model...")
    history = model.fit(train_dataset,
                        epochs=num_epochs,
                        validation_data=test_dataset,
                        shuffle=True,
                        callbacks=[early_stopping_monitor],
                       )
    print("Model training finished")

    return history
