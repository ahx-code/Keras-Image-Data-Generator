from os.path import join, exists
from shutil import rmtree
from os import mkdir, listdir
from cv2 import imwrite
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array


def create_dirs():
    """Create train and test dirs
    """
    data_dir = "data"
    train_dir = join(data_dir, "train")
    test_dir = join(data_dir, "test")
    if exists(data_dir):
        rmtree(path=data_dir)
    mkdir(path=data_dir)
    mkdir(path=train_dir)
    mkdir(path=test_dir)
    return train_dir, test_dir


def get_name(label, count):
    return str(str(label) + '_' +
               str(count) + '.png')


def get_data(data):
    x_train = data[0]
    y_train = data[2]
    x_test = data[1]
    y_test = data[3]
    return x_train, y_train, x_test, y_test


def save_to_dir(size, label, directory, x):
    """
    Args:
        size (int): set size
        label (str): class names
        directory (str): where images
            will be saved
        x (list): image data
    """
    current = label[0]
    count = 0

    for i in range(size):
        if current != label[i]:
            current = label[i]

        label_dir = join(directory, str(label[i]))

        if not exists(path=label_dir):
            mkdir(path=label_dir)

        image_name = get_name(label[i], count)
        file_name = join(label_dir, image_name)
        imwrite(filename=file_name, img=x[i] * 255)
        count += 1

    # Check if the calculation is correct
    check_set_size = 0
    entries = listdir(path=directory)

    for entry in entries:
        current_images = listdir(join(directory, entry))
        check_set_size += len(current_images)

    assert (check_set_size == size)


def generate_and_save(data_gen, set_path, set_dir, gen_num,
                      save_format=".png", save_prefix="gen_image"):
    for label in set_dir:
        if label != ".DS_Store":  # Mac-os specific problem
            path = join(set_path, label)
            images = listdir(path=path)

            for img in images:
                if img != '.DS_Store':  # Mac-os specific problem
                    count_img = 1
                    img_path = join(path, img)
                    loaded = load_img(path=img_path,
                                      color_mode="grayscale")
                    arr_img = img_to_array(img=loaded)
                    current = arr_img.reshape((1,) + arr_img.shape)

                    for _ in data_gen.flow(x=current,
                                           batch_size=1,
                                           save_to_dir=path,
                                           save_prefix=save_prefix,
                                           save_format=save_format):
                        count_img += 1
                        if count_img > gen_num:
                            break
