import numpy as np 
import matplotlib.pyplot as plt 
import flickr_api
from flickr_api.api import flickr

flickr_api.set_keys(api_key = '7bf041bab823926fe2a02b0d530cc345', \
	api_secret = '1c62774cf84b482f')

def xml2url(xml):
	xml = xml.split()
	photo_id = xml[0][4:-1]
	secret = xml[2][8:-1]
	server_id = xml[3][8:-1]
	farm_id = xml[4][6:-1]
	url = 'https://farm%s.staticflickr.com/%s/%s_%s_b.jpg'\
		%(farm_id, server_id, photo_id, secret)
	return url

def getallurl(allxml):
	allxml = allxml.split('<photo ')
	urls = []
	for i in range(1, len(allxml)):
		urls.append(xml2url(allxml[i]))
	return urls

def query2urls(query, page=1, per_page=500):
	query_xml = flickr.photos.search(api_key = '7bf041bab823926fe2a02b0d530cc345', \
		text=query, per_page=per_page, page=page, tag_mode='all')
	return getallurl(query_xml)


query = 'table chair'
urls = query2urls(query)

for i in range(len(urls)):
	print urls[i]
	