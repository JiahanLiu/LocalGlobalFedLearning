import requests
import os.path

URL_BASE = "http://167.71.247.3:3000/"
UPLOAD_URL = URL_BASE + "local_upload"
DOWNLOAD_URL = URL_BASE + "get_global_param"

FILE_DIR = "param_files/"
UPLOAD_NAME = "upload.txt"
DOWNLOAD_NAME = "download.txt"

pwd_path = os.path.abspath(os.path.dirname(__file__))

def upload_file_to_server(file_dir, file_name):
    with open(file_dir+file_name, 'rb') as file:
        files = {'file':('node1_param.txt', file)}
        r = requests.post(UPLOAD_URL, files=files)
        print(r)
        print("File Uploaded")

absolute_path = os.path.join(pwd_path, FILE_DIR)
upload_file_to_server(absolute_path, UPLOAD_NAME)

def get_file_from_server(file_dir, file_name):
    r = requests.get(DOWNLOAD_URL, allow_redirects=True)
    open(file_dir+file_name, 'wb').write(r.content)
    print("File Downloaded")

# get_file_from_server(absolute_path, DOWNLOAD_NAME)