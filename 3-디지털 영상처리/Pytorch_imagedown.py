
import os
import shutil
from bing_image_downloader import downloader

for keyword, folder_name in [("wild boar", "wild_boar"),
                             ("domestic pig", "pig"),
                             ("deer animal", "deer"),
                             ("cat", "cat"),
                             ("dog", "dog")]:
    downloader.download(keyword, limit=150, output_dir='dataset_raw', force_replace=False)
    src = os.path.join('dataset_raw', keyword)
    dst = os.path.join('dataset', folder_name)

    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.move(src, dst)

