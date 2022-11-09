import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
    

class app():
    def __init__(self):
        self.menu_selection = ''
        self.selections = None
        self.ogdf = pd.read_csv('movies.csv')
        self.df = self.ogdf.copy(deep=True)
        self.currrecs = self.df.sample(10)
    
    def EuclideanDistance(self, row1, row2):
        return abs(row1 - row2)
    
    def cosine_similarity(self, base: str, comparison: str):
        tfid = TfidfVectorizer()
        tfidf_matrix = tfid.fit_transform((base, comparison))
        results = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])
        return results[0][0]
    
    def weighted_jaccard(self, weighted_dictionary: dict, comparator_genres: str):

        numerator = 0
        denominator = weighted_dictionary['total']
        for genre in comparator_genres.split(';'):
            if genre in weighted_dictionary:
                numerator += weighted_dictionary[genre]

        return numerator / denominator
    
    def cosine_weighted_jaccard_and_ED(self, genres_weighted_dictionary_dict: dict, titles: str, comparator_movie: pd.core.series.Series):
        cs_result = self.cosine_similarity(titles, comparator_movie['title'])
        wjs_result = self.weighted_jaccard(genres_weighted_dictionary_dict, comparator_movie['genres'])
        ed_result = self.EuclideanDistance(self.selections['year'].mean(), comparator_movie['year'])
        cs_result = (cs_result + 1) / 2.0
        # Weights:
        #Weight of .1 (10%) for the year
        ed_result *= 0.1
        # Use a weight of 0.3 (20%) for the title:
        cs_result *= 0.3
        # Use a weight of 0.6 (70%) for the genre
        wjs_result *= 0.6
        
        
        return wjs_result + cs_result + ed_result
        
        
        
    def getNewRecommendations(self, idx):
        try:
            if self.selections is None:
                self.selections = self.ogdf[self.ogdf['IMDB_id'] == int(idx)].copy(deep=True)
            else:
                self.selections.loc[len(self.selections)] = self.ogdf[self.ogdf['IMDB_id'] == int(idx)].values[0]
            titles = self.selections['title'].values
            titles = ' '.join(titles)
            genres_weighted_dictionary = {'total': 0}
            for el in self.selections['genres']:
                for genre in el.split(';'):
                    if genre in genres_weighted_dictionary:
                        genres_weighted_dictionary[genre] += 1
                    else:
                        genres_weighted_dictionary[genre] = 1
                    genres_weighted_dictionary['total'] += 1
            self.df = self.ogdf.copy(deep=True)
            self.df = self.df[self.df['rating'].isin(['PG','PG-13', 'R', 'G'])]
            self.df = self.df[self.df['year'].astype(int) >= 2005]
            self.df = self.df[self.df['stars'].astype(float) >= 5.5]
            self.df['multiple_metrics'] = self.df.apply(lambda x: self.cosine_weighted_jaccard_and_ED(genres_weighted_dictionary, titles, x), axis='columns')               
            self.df = self.df[~self.df['title'].isin(self.selections['title'])]
            self.df = self.df.sort_values(by=['multiple_metrics'], ascending=False)
            self.currrecs = self.df.head(10)
            
        except Exception as e:
            return 
               
    def op1(self): 
        while True:
            print('\nYour Selections:\n')
            if self.selections is None:
                print('No selections yet')
            else:
                for item in self.selections[['IMDB_id', 'title', 'stars']].values:
                    print(f"{item[0]}:   {item[1]}  \n   IMDB Rating: {item[2]}/10 stars\n")
            print('\nCurrent Recommendations: \n')
            for item in self.currrecs[['IMDB_id', 'title', 'stars']].values:
                print(f"{item[0]}:   {item[1]}  \n   IMDB Rating: {item[2]}/10 stars\n")
            input('\nPress Enter to go back')
            break
    

    
    def enterRecommendation(self, idx):
        try:
            tmp = self.df[self.df['IMDB_id'] == int(idx)]
            if tmp is None:
                raise
            else:
                self.getNewRecommendations(idx)
                return True
        except:
            return False
            
    
    def search(self, searchword):
        while True:
            print('\nSearch Results: ')
            tmp = self.ogdf[self.ogdf['title'].str.contains(searchword, case=False)][['IMDB_id', 'title', 'stars']]
            for item in tmp.values:
                print(f"{item[0]}: {item[1]}\n   IMDB Rating: {item[2]}/10 stars\n")
            print()
            print('\nEnter the ID of the movie you want to add to your list')
            print('press Enter to go back and search again')
            print('-1 to exit')
            selection = input('Enter selection: ')
            if selection == '':
                break
            elif selection == '-1':
                exit()
            else:
                print('\n Thinking....\n', end='')
                r = self.enterRecommendation(selection)
                if r is True:     
                    print('\nMovie added to list')
                else:
                    print('\nInvalid ID')
                break
                
    def op2(self):
        selection = ''
        while True:
            selection = input('\nEnter a search term (press Enter to go back, -1 to exit):\n ')
            if selection == '':
                break
            elif selection == '-1':
                exit()
            else:
                self.search(selection)
                break
        
    def start(self):
        while True:
            print('\nOptions: ')
            print('1. View Current recommendations')
            print('2. Search Movies')
            print('(-1 to Exit)')
            self.menu_selection = input('Enter selection: ')
            
            if self.menu_selection == '1':
                self.op1()
            elif self.menu_selection == '2':
                self.op2()
            elif self.menu_selection == '-1':
                break
            else:
                continue
        
        
if __name__ == '__main__':
    app = app()
    app.start()
    
        