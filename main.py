# -*- coding: utf-8 -*-
from os import listdir
from argument import argument
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from util import get_data, create_dirs, save_to_dir
from util import generate_and_save
from keras.preprocessing.image import ImageDataGenerator

if __name__ == '__main__':
    args = argument()

    olivetti = fetch_olivetti_faces()

    X = olivetti.images
    y = olivetti.target

    data = train_test_split(X, y,
                            test_size=args.test_size,
                            random_state=42)

    x_train, y_train, x_test, y_test = get_data(data)

    train_dir, test_dir = create_dirs()

    save_to_dir(size=x_train.shape[0], label=y_train,
                directory=train_dir, x=x_train)

    save_to_dir(size=x_test.shape[0], label=y_test,
                directory=test_dir, x=x_test)

    train_size = x_train.shape[0]
    test_size = x_test.shape[0]

    gen_train = round(args.train_gen / train_size)
    gen_test = round(args.test_gen / test_size)

    data_gen = ImageDataGenerator(rescale=1./255, shear_range=0.2,
                                  zoom_range=0.2, horizontal_flip=True)

    generate_and_save(data_gen, set_path=train_dir,
                      set_dir=listdir(train_dir), gen_num=gen_train)
    generate_and_save(data_gen, set_path=test_dir,
                      set_dir=listdir(test_dir), gen_num=gen_test)
