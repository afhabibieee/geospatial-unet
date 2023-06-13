from osgeo import gdal
import numpy as np
from sklearn.model_selection import train_test_split

def read_tif(file_path):
    """
    Membaca file TIFF dan mengembalikan array 3D, proyeksi, dan transformasi geografis.

    Parameters:
        file_path (str): Path file TIFF yang akan dibaca.

    Returns:
        tuple: Tuple berisi tiga elemen:
            - array gambar 3D (numpy.ndarray): Array 3D yang terdiri dari beberapa band gambar.
            - proyeksi (str): Proyeksi dataset TIFF.
            - transformasi geografis (tuple): Tuple berisi informasi transformasi geografis dataset TIFF.
    """
    dataset = gdal.Open(file_path, gdal.GA_ReadOnly)
    projection = dataset.GetProjection()
    geo_transform = dataset.GetGeoTransform()
    bands = []
    for b in range(1, dataset.RasterCount + 1):
        band = dataset.GetRasterBand(b)
        bands.append(band.ReadAsArray())
    return np.dstack(bands), projection, geo_transform

def cut_into_tiles(img, tile_size=224):
    """
    Memotong gambar menjadi potongan-potongan (tiles) dengan ukuran yang ditentukan.

    Parameters:
        img (numpy.ndarray): Array gambar yang akan dipotong.
        tile_size (int, optional): Ukuran tile yang diinginkan. Default: 224.

    Returns:
        dict: Dictionary yang berisi potongan-potongan gambar (tiles) beserta koordinatnya.
            Key: Koordinat (tuple) berisi (baris, kolom) dari tile.
            Value: Array gambar tile.
    """
    height, width, _ = img.shape
    tiles = {}
    for i in range(0, height, tile_size):
        for j in range(0, width, tile_size):
            tile = img[i:i+tile_size, j:j+tile_size]
            if tile.shape[0] == tile_size and tile.shape[1] == tile_size:
                tiles[(i, j)] = tile
    return tiles

def split_tiles(satellite_tiles, vegetation_tiles):
    """
    Memisahkan potongan-potongan gambar (tiles) menjadi data latih (train) dan data uji (test).

    Parameters:
        satellite_tiles (dict): Dictionary yang berisi potongan-potongan gambar (tiles) satelit.
            Key: Koordinat (tuple) berisi (baris, kolom) dari tile.
            Value: Array gambar tile satelit.
        vegetation_tiles (dict): Dictionary yang berisi potongan-potongan gambar (tiles) vegetasi.
            Key: Koordinat (tuple) berisi (baris, kolom) dari tile.
            Value: Array gambar tile vegetasi.

    Returns:
        tuple: Tuple berisi enam elemen:
            - train_indices (list): List indeks tile untuk data latih.
            - test_indices (list): List indeks tile untuk data uji.
            - sat_train (numpy.ndarray): Array gambar satelit data latih.
            - veg_train (numpy.ndarray): Array gambar vegetasi data latih.
            - sat_test (numpy.ndarray): Array gambar satelit data uji.
            - veg_test (numpy.ndarray): Array gambar vegetasi data uji.
    """
    # Membuat list dari tiles keys
    tile_indices = list(satellite_tiles.keys())

    # Split keys untuk train dan test
    train_indices, test_indices = train_test_split(tile_indices, test_size=0.2)

    sat_train = np.array([satellite_tiles[idx] for idx in train_indices])
    veg_train = np.array([vegetation_tiles[idx] for idx in train_indices])
    sat_test = np.array([satellite_tiles[idx] for idx in test_indices])
    veg_test = np.array([vegetation_tiles[idx] for idx in test_indices])

    return train_indices, test_indices, sat_train, veg_train, sat_test, veg_test

def jahit_tiles(tiles, output_shape):
    """
    Menggabungkan potongan-potongan gambar (tiles) menjadi gambar lengkap.

    Parameters:
        tiles (dict): Dictionary yang berisi potongan-potongan gambar (tiles).
            Key: Koordinat (tuple) berisi (baris, kolom) dari tile.
            Value: Array gambar tile.
        output_shape (tuple): Tuple berisi dimensi gambar lengkap yang diinginkan (tinggi, lebar, band).

    Returns:
        numpy.ndarray: Array gambar lengkap yang telah digabungkan.
    """
    full_img = np.zeros(output_shape[:2])  # Ambil tinggi dan lebarnya saja
    tile_size = next(iter(tiles.values())).shape[0]  # Dapatkan tile size dari tile pertama
    for (i, j), tile in tiles.items():
        full_img[i:i+tile_size, j:j+tile_size] = tile[..., 0]
    return full_img

def write_tif(output_path, data, geo_transform, projection):
    """
    Menulis hasil prediksi yang sudah dijahit ke file TIFF dengan parameter yang ditentukan.

    Parameters:
        output_path (str): Path file TIFF yang akan ditulis.
        data (numpy.ndarray): Array data yang sudah dijahit yang akan ditulis ke file TIFF.
        geo_transform (tuple): Tuple berisi informasi transformasi geografis.
        projection (str): Proyeksi dataset TIFF.

    Returns:
        None
    """
    driver = gdal.GetDriverByName('GTiff')
    dataset = driver.Create(
        output_path,
        data.shape[1],
        data.shape[0],
        1,
        gdal.GDT_Float32,
    )
    dataset.SetGeoTransform(geo_transform)
    dataset.SetProjection(projection)
    dataset.GetRasterBand(1).WriteArray(data)
    dataset.FlushCache()  # Write to disk.
