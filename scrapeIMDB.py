from urllib.request import urlopen
from urllib.error import HTTPError
from bs4 import BeautifulSoup
import re
import pandas as pd
import math

URL_PATTERN = ('http://www.imdb.com/search/title?at=100&countries=us&count=100'
              +'&title_type=feature&start={start_nr}&genres={genre}'
              +'&year={start_year}-{end_year}') 

def getBSObj(url):
    """Load url in bs object """
    print("Scraping: " + url + "\n")
    try:
        html = urlopen(url)
    except HTTPError as e:
        return None     
    try:
        bs_obj = BeautifulSoup(html.read(), 'lxml')            
    except AttributeError as e:
        return None    
    return bs_obj
    
    
def getNrMovies(bs_obj):    
    """Get nr of movies in selected period and genre"""
    nr_string = bs_obj.find('div', {'id':'left'})
    if nr_string == None:
        print("Number of movies in selection could not be found")
        return None
    else:
        nr_string = nr_string.getText()
        nr = re.findall("of (.*?)\n",nr_string)
        nr = nr[0]
        nr = nr.replace(",", "")
        nr = int(nr)
        return nr


def getMovDatTemp(bs_obj, genre, load_rem_nr):
    """Store film genres, titles, 
       IDs,and poster-urls in mov_dat_temp:
    """ 
    mov_dat_temp = pd.DataFrame(columns=['genre',
                                         'all_genres',
                                         'title',
                                         'IMDBID',
                                         'poster_URL'])
                                    
    titles = bs_obj.findAll('td', {'class':'image'})
    if titles == None:
        print("titles could not be found")
    else:
        i = 0
        for t in titles:
            if i >= load_rem_nr: break
            IMDBID = t.find('a').get('href')
            IMDBID = re.findall("e/(.*?)/",IMDBID)
            mov_dat_temp.set_value(i, 'IMDBID', IMDBID[0])
    
            title = t.find('a').get('title')
            mov_dat_temp.set_value(i, 'title', title)
            
            img_src = t.find('img').get('src')
            mov_dat_temp.set_value(i, 'poster_URL', img_src)
            i += 1
            
    #Get film genres:
    genres = bs_obj.findAll('span', {'class':'genre'})
    if genres == None:
        print("Genres could not be found")
    else:
        i = 0
        for g in genres:
            if i >= load_rem_nr: break
            gt = g.getText()
            mov_dat_temp.set_value(i, 'all_genres', gt)
            mov_dat_temp.set_value(i, 'genre', genre)
            i += 1
                
    return mov_dat_temp                

 
def getMovDat(genre_list = ['western'], years = ['2000','2014'], load_nr = 10):
    """Aggregate all movie info into dataframe 'mov_dat'
       
       Looping over genres, get list of movies in each genre and save:
       - Genres (String containing all genres, seperate by '|')
       - URL of poster
       - Title (Year)
       - IMDB ID (eg. tt1853728)
    """
    for genre in genre_list:
        
        # Construct address of IMDB list belonging to genre
        IMDB_url = URL_PATTERN.format(start_nr = 1, genre = genre,
                                      start_year = years[0], 
                                      end_year = years[1])
                                      
        bs_obj = getBSObj(IMDB_url)
        
        nr = getNrMovies(bs_obj)
        
        n_loads = math.ceil(min([nr,load_nr])/100)    
        
        for i_load in range(0,n_loads):
            if i_load > 0:   #don't reload the first 100 results  
                IMDB_url = URL_PATTERN.format(start_nr = str((i_load)*100+1), 
                                              genre = genre,
                                              start_year = years[0],
                                              end_year = years[1])
                
                print('Scraping: ' + IMDB_url + '\n')
                bs_obj = getBSObj(IMDB_url)
            
            #Remaining titles to load:
            load_rem_nr = min([load_nr - i_load * 100, 100])
            mov_dat_temp = getMovDatTemp(bs_obj, genre, load_rem_nr)
            
            #Append mov_dat_temp to mov_dat            
            if i_load == 0 and genre == genre_list[0]:
                mov_dat = mov_dat_temp
            else:
                mov_dat = mov_dat.append(mov_dat_temp, ignore_index=True)    
    
    return mov_dat

if __name__=="__main__":
    mov_dat = getMovDat()