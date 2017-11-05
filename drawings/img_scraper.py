import urllib2
from BeautifulSoup import BeautifulSoup

import urllib2, sys
from BeautifulSoup import BeautifulSoup

site= "http://dota2.gamepedia.com/Category:Hero_minimap_icons"
hdr = {'User-Agent': 'Mozilla/5.0'}
req = urllib2.Request(site,headers=hdr)
page = urllib2.urlopen(req)
soup = BeautifulSoup(page)

for img in soup.findAll('img'):
  img_src_path = img['src']
  if "icon" in img_src_path and "media" in img_src_path:
    file_name = img_src_path.split("/")[-1].split("?")[0]
    print file_name
    with open("images/" + file_name,'wb') as f:
      f.write(urllib2.urlopen(img['src']).read())

