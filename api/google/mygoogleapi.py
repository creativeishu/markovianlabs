# Google images

from bs4 import BeautifulSoup
import urllib2
import os
import json
from sys import argv, exit
from subprocess import call 

#==============================================================================

# you can change the query for the image  here
# query = raw_input("query image: ")
if len(argv)==2:
	query = argv[1]
else:
	print "Usage: python api.py <query>"
	exit()

savefile = True

#==============================================================================

DIR="Pictures"
if not os.path.exists(DIR):
            os.mkdir(DIR)
DIR = os.path.join(DIR, query.split()[0])

if not os.path.exists(DIR):
            os.mkdir(DIR)

myfilename = DIR + '/list.txt'
f = open(myfilename, 'w')

#==============================================================================

def get_soup(url,header):
    return BeautifulSoup(urllib2.urlopen(urllib2.Request(url,headers=header)), \
    	'html.parser')

#==============================================================================

# image_type="ActiOn"
query= query.split()
query='+'.join(query)
url="https://www.google.co.in/search?q="+query+"&source=lnms&tbm=isch"
print url
#add the directory for your image here
header={'User-Agent':"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 \
(KHTML, like Gecko) Chrome/43.0.2357.134 Safari/537.36"
}
soup = get_soup(url,header)

#==============================================================================

ActualImages=[]# contains the link for Large original images, type of  image
i = 0
for a in soup.find_all("div",{"class":"rg_meta"}):
    i += 1
    link , Type =json.loads(a.text)["ou"]  ,json.loads(a.text)["ity"]
    ActualImages.append((link,Type))
    f.write('%s \n'%link)
    if savefile:
        command = 'wget -O %s/%i.jpg %s'%(DIR, i, link)
        call(command, shell=True)
f.close()

print  "there are total" , len(ActualImages),"images"

#==============================================================================
