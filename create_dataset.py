#%%
import requests
import csv
import urllib.request
import codecs
import os
from tqdm import tqdm

directories = ['train', 'val', 'test', 'train/images', 'val/images', 'test/images', 'train/labels', 'val/labels',
               'test/labels']

classNames = ["Door handle", "Person", "Light switch"]

MID_url = 'https://storage.googleapis.com/openimages/v5/class-descriptions-boxable.csv'
annotations = {}
annotations['train'] = 'https://storage.googleapis.com/openimages/2018_04/train/train-annotations-bbox.csv'
annotations['val'] = 'https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv'
annotations['test'] = 'https://storage.googleapis.com/openimages/v5/test-annotations-bbox.csv'

imageURLs = {}
imageURLs[
    'train'] = 'https://storage.googleapis.com/openimages/2018_04/train/train-images-boxable-with-rotation.csv'
imageURLs[
    'val'] = 'https://storage.googleapis.com/openimages/2018_04/validation/validation-images-with-rotation.csv'
imageURLs['test'] = 'https://storage.googleapis.com/openimages/2018_04/test/test-images-with-rotation.csv'


def create_directory_structure(directories):
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)


def sanity_check(imageIDs):
    for subset, info in imageIDs.items():
        for imageID in info.keys():
            filePath = os.path.join(subset, 'images', imageID + '.jpg')
            labelingPath = os.path.join(subset, 'labels', imageID + '.txt')
            if os.path.exists(filePath):
                if os.path.getsize(filePath) == 0:
                    print(filePath)
                    os.remove(filePath)
                    if os.path.exists(labelingPath):
                        print(labelingPath)
                        os.remove(labelingPath)


def downloadImages(imageInfo):
    for subset, details in imageInfo.items():
        total_downloads = 1
        n_downloads = [0, 0, 0]
        for imageID, info in details.items():
            if n_downloads[info['labels'][0][0]] < 600:
                n_downloads[info['labels'][0][0]] = n_downloads[info['labels'][0][0]] + 1

                filePath = os.path.join(subset, 'images', imageID + '.jpg')
                with open(filePath, 'wb') as handle:
                    response = requests.get(info['url'], stream=True)
                    if not response.ok:
                        print(response)
                        print("Removing " + filePath)
                        os.remove(filePath)
                        continue
                    for block in response.iter_content(1024):
                        if not block:
                            break

                        handle.write(block)

                print(total_downloads, "Downloaded " + imageID + " from " + info['url'])
                filePath = os.path.join(subset, 'labels', imageID + '.txt')
                with open(filePath, 'w') as file:
                    for [category, x_min, y_min, x_max, y_max] in info['labels']:
                        file.write(str(category) + ' ' + str(x_min) + ' ' + str(y_min) + ' ' + str(x_max) + ' ' + str(
                            y_max) + '\n')
                        idx = category

                print("Individual Downloads:", n_downloads)
                total_downloads += 1


def extractImageURLsFromIDs(URLs, imageIDs):
    print("Extracting Image URLs")
    for subset, url in URLs.items():
        print("Working on subset: ", subset)
        ftpstream = urllib.request.urlopen(url)
        csvfile = csv.reader(codecs.iterdecode(ftpstream, 'utf-8'))
        next(csvfile)
        for lineContents in tqdm(csvfile):
            imageID = lineContents[0]
            imageURL = lineContents[2]
            if imageID in imageIDs[subset]:
                imageIDs[subset][imageID]['url'] = imageURL

    print("Done.")
    return imageIDs


def extractImageIDs(URLs, categories):
    print("Extracting Image Ids...")
    imageIDs = {}

    for subset, url in URLs.items():
        print("Working on subset: ", subset)
        if subset not in imageIDs:
            imageIDs[subset] = {}

        ftpstream = urllib.request.urlopen(url)
        csvfile = csv.reader(codecs.iterdecode(ftpstream, 'utf-8'))
        next(csvfile)
        for lineContents in tqdm(csvfile):
            imageID = lineContents[0]
            category = lineContents[2]
            [x_min, x_max, y_min, y_max] = [float(lineContents[4]), float(lineContents[5]), float(lineContents[6]),
                                            float(lineContents[7])]
            if category in categories:
                if imageID not in imageIDs[subset]:
                    imageIDs[subset][imageID] = {}
                    imageIDs[subset][imageID]['labels'] = []
                imageIDs[subset][imageID]['labels'].append([categories.index(category), x_min, y_min, x_max, y_max])
    print("Done.")
    return imageIDs


def extractMIDs(url, objects):
    print("Extracting MIDs...")
    categories = []

    ftpstream = urllib.request.urlopen(url)
    csvfile = csv.reader(codecs.iterdecode(ftpstream, 'utf-8'))
    for lineContents in tqdm(csvfile):
        if lineContents[1] in objects:
            categories.append(lineContents[0])

    print("Done.")
    return categories


create_directory_structure(directories)

categories = extractMIDs(MID_url, classNames)

images_info = extractImageIDs(annotations, categories)

images_info = extractImageURLsFromIDs(imageURLs, images_info)

downloadImages(images_info)
sanity_check(images_info)

