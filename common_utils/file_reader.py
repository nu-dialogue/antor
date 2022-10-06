import zipfile
import json

def read_zipped_json(filepath, filename):
    archive = zipfile.ZipFile(filepath, 'r')
    return json.load(archive.open(filename))

    