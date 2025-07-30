import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df_train = pd.read_csv('/kaggle/input/dpl-2025/Text_Image.csv')
df_test = pd.read_csv('/kaggle/input/dpl-2025/Test_Text_Image.csv')

base_image_dir_train = "/kaggle/input/dpl-2025/devset_images"
base_image_dir_test = "/kaggle/input/dpl-2025/testset_images"

df_train['image_path'] = df_train['image_path'].apply(lambda x: f"{base_image_dir_train}/{x}")
df_test['image_path'] = df_test['image_path'].apply(lambda x: f"{base_image_dir_test}/{x}")


def fill_missing(text):
    if isinstance(text, str) and text.strip() == "":
        return "missing"
    return text

df_train = df_train.fillna("missing")
df_train = df_train.applymap(fill_missing)

df_test = df_test.fillna("missing")
df_test = df_test.applymap(fill_missing)


texts = (
    df_train['title'].fillna('') + ' ' +
    df_train['description'].fillna('') + ' ' +
    df_train['user_tags'].fillna('')
).tolist()
image_paths_train = df_train['image_path'].tolist()
labels = df_train['label'].tolist()  # Assuming labels are in a column named 'label'

test_ids = df_test['image_id'].tolist()
test_texts = (
    df_test['title'].fillna('') + ' ' +
    df_test['description'].fillna('') + ' ' +
    df_test['user_tags'].fillna('')
).tolist()
image_paths_test = df_test['image_path'].tolist()

train_texts, val_texts, train_labels, val_labels, train_image_paths, val_image_paths = train_test_split(texts, labels, image_paths_train,
                                                                                      test_size=0.05, stratify=labels, random_state=42)