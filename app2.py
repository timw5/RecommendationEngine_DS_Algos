import warnings
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import Levenshtein
warnings.simplefilter(action='ignore', category=FutureWarning)
# this is because of a recent pandas update issue ^


class App:
    def __init__(self):
        self.menu_selection = ''
        self.filters = []
        self.displayfilters = []
        self.features = []
        self.weights = []
        self.num_clusters = 0
        self.selections = None
        self.currrecs = None
        self.ogdf = pd.read_csv('movies.csv')
        self.df = self.ogdf.copy(deep=True)

    def start(self):
        while(True):
            print('\n\nSearch for a movie to begin generating recommendations')
            search_term = input('Enter search term (-1 to quit):   ')
            if search_term == '-1':
                exit()
            if not self.search(search_term):
                continue
            if not self.pick_important_factors():
                continue
            if not self.choose_cluster():
                continue
            if self.choose_filters():
                break
            continue  
        self.main()                  


            

    def search(self, search_word):
        while True:
            print('\n\nSearch Results: ')
            tmp = self.df[self.df['title'].str.contains(search_word, case=False)][['IMDB_id', 'title', 'stars']]
            for item in tmp.values:
                print(f"{item[0]}: {item[1]}\n   IMDB Rating: {item[2]}/10 stars\n")
            print('\nEnter the ID of the movie you want to add to your list')
            print("press 0 or Enter to go back and search again")
            print('-1 to exit\n')
            selection = input('Enter selection:   ')
            if selection == '' or selection == '0':
                return False
            elif selection == '-1':
                exit()
            else:
                try:
                    self.selections = self.df[self.df['IMDB_id'] == int(selection)]
                    if self.selections.empty:
                        raise ValueError
                except ValueError:
                    print('Invalid Selection\n')
                    continue
                else:
                    break
        return True

    def pick_important_factors(self):
        print('\n\nPick at least one of these factors that are important to you')
        print('And we will tailor your recommendations towards those factors')
        print('Press -1 to go back to select a different movie')
        print("Press '0', or Enter to continue")
        print('press -2 to exit')
        self.features.clear()
        count = 0
        selection = ''
        feature_map = {'1': 'Genre', '2': 'Title'}
        while True:
            if count == 2:
                break
            for item in feature_map.items():
                if item[1] not in self.features:
                    print(item[0], ' ', item[1])
            selection = input('Enter the number corresponding to your selection:   ')
            try:
                if selection == '0' or selection == '':
                    if len(self.features) == 0:
                        print('You must select at least one feature')
                        continue
                    break
                
                int_selection = int(selection)
                if int_selection > 0 and int_selection < 3:
                    count += 1
                    self.features.append(feature_map[selection])
                    continue
                elif int_selection == -1:
                    return False
                elif int_selection == -2:
                    exit()
            except ValueError:
                print('Invalid selection')
                continue
        return True

    def choose_cluster(self):
        while True:
            print('\n\nSelect How many Clusters you would like to use (must be greater than 2)')
            print('-1 to exit, \n0 to go back to change the factors influencing your recommendations')
            selection = input('Enter number of clusters:   ')
            try:
                if selection == '0':
                    return self.pick_important_factors()
                elif selection == '-1':
                    exit()
                else:
                    self.num_clusters = int(selection)
                    break
            except ValueError:
                print('Invalid selection')
                continue
        self.cluster()
        return True
        # self.choose_filters()

    def choose_filters(self):
        filtermap = {'1': self.cosine_with_description, '2': self.levenstein_with_title, '3': self.filterYears}
        filtermap2 = {'1': 'Cosine Similarity with Description', '2': 'Levenstein Distance with Title',
                      '3': 'Euclidean Distance with Year'}
        options = [{'1': 'Use cosine similarity on the description of the movies'},
                   {'2': 'Use Levenstein Distance on the title of the movies'},
                   {'3': 'Use Euclidean Distance on the year of the movies'}]
        counts = 0
        selection = ''
        while counts < 3 or selection != '0':
            print('\n\nMovies are now clustered, We are ready to make recommendations')
            print('Select your filters, and we will generate recommendations for you')
            if len(options) == 0:
                break
             
            for item in options:
                for key, value in item.items():
                    print(f'{key}. {value}')
            print('0 to continue')
            print('-1 to go back to change your selection')
            print('-2 to exit')
            selection = input('Enter your selection:   ')
            if selection == '-1':
                if self.choose_cluster():
                    continue
                break
            if selection == '-2':
                exit()
            elif not selection.isdigit() or int(selection) > 3 or int(selection) < -1:
                print('invalid selection')
                continue
            elif selection == '0':
                if len(self.filters) == 0:
                    print('You must select at least one filter')
                    continue
                else:
                    break       
            else:
                weight = self.choose_weight()
                if weight == -1:
                    continue
                else:
                    cp = options.copy()
                    for idx, item in enumerate(cp):
                        for key, value in item.items():
                            if key == selection:
                                options.pop(idx)
                    self.weights.append({filtermap2[selection]: weight})
                    self.filters.append({filtermap[selection]: weight})
                    counts += 1
                    if counts == 3:
                        break
                    elif sum([sum(y) for y in [x.values() for x in self.weights]]) == 100:
                        break
                    continue
        return True
    
    def choose_weight(self):
        print('\n\nEnter a weight to assign to this filter')
        print('Keep in mind, the sum of all weights must be less than or equal to 100')
        print('0 to exit')
        print('-1 to go back to change your selection')
        currweight = 0
        weightleft = 100
        if len(self.weights) > 0:
            print('current weights:   ')
            for item in self.weights:
                for key, value in item.items():
                    print(f'Filter: {key}\nWith a Weight of: {value}')
                    currweight += value
            print(f'Total Weight of current filters: {currweight}')
            weightleft = 100 - currweight
            print(f'you have {weightleft} left to assign')
        else:
            print('No weights have been assigned yet, you have up to 100 to assign')
        while True:
            selection = input(f'Enter a value between 0 and {weightleft} (or -1 to go back): ')
            if selection == '-1':
                return -1
            try:
                int_selection = int(selection)
                if int_selection > weightleft:
                    print('That value is too high')
                    continue
                elif int_selection <= 0:
                    print('value must be greater than 0')
                    continue
                else:
                    return int_selection
            except ValueError:
                print('Invalid selection')
                continue
        
    def cluster(self):
        print(f'\nClustering with {self.num_clusters} clusters')
        print(f'\nBased on {set(self.features)}')
        self.ogdf = self.df.copy(deep=True)
        if set(self.features) == {'Genre', 'Title'}:
            genres_and_titles = self.ogdf['genres'] + self.ogdf['title']
            vectorizer = TfidfVectorizer(stop_words='english')
            X = vectorizer.fit_transform(genres_and_titles)
            kmeans = KMeans(n_clusters=self.num_clusters, random_state=0)
            kmeans.fit(X)
            self.ogdf['genre_title_cluster'] = kmeans.labels_
            selected_movies = self.selections['genres'] + self.selections['title']
            Y = vectorizer.transform(selected_movies)
            prediction = kmeans.predict(Y)
            clustered_movies = self.ogdf[self.ogdf['genre_title_cluster'] == int(prediction)]
            self.ogdf = clustered_movies
        elif set(self.features) == {'Genre'}:
            genres = self.ogdf['genres']
            vectorizer = TfidfVectorizer(stop_words='english')
            X = vectorizer.fit_transform(genres)
            kmeans = KMeans(n_clusters=self.num_clusters, random_state=0)
            kmeans.fit(X)
            self.ogdf['genre_cluster'] = kmeans.labels_
            selected_genres = self.selections['genres']
            Y = vectorizer.transform(selected_genres)
            prediction = kmeans.predict(Y)
            clustered_movies = self.ogdf[self.ogdf['genre_cluster'] == prediction]
            self.ogdf = clustered_movies
        else:
            titles = self.ogdf['title']
            vectorizer = TfidfVectorizer(stop_words='english')
            X = vectorizer.fit_transform(titles)
            kmeans = KMeans(n_clusters=self.num_clusters, random_state=0)
            kmeans.fit(X)
            self.ogdf['title_cluster'] = kmeans.labels_
            selected_titles = self.selections['title']
            Y = vectorizer.transform(selected_titles)
            prediction = kmeans.predict(Y)
            clustered_movies = self.ogdf[self.ogdf['title_cluster'] == prediction]
            self.ogdf = clustered_movies

    @staticmethod
    def cosine_with_description(base, comparison):
        
        tfid = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfid.fit_transform((base, comparison['plot']))
        results = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])
        return results[0][0]

    @staticmethod
    def levenstein_with_title(base, comparison):
        b_title = " ".join(base)
        if not isinstance(comparison, str):
            comparison = " ".join(comparison)
        return Levenshtein.distance(b_title, comparison)

        
    def processFilters(self):
        genres_weighted_dictionary = {'total': 0}
        desc = self.selections['plot'].values
        desc = ' '.join(desc)
        for el in self.selections['genres']:
            for genre in el.split(';'):
                if genre in genres_weighted_dictionary:
                    genres_weighted_dictionary[genre] += 1
                else:
                    genres_weighted_dictionary[genre] = 1
                genres_weighted_dictionary['total'] += 1
        cs, lv = 0, 0
        for item in self.weights:
            for key, value in item.items():
                if key == 'Cosine Similarity with Description':
                    cs = value/100
                elif key == 'Levenstein Distance with Title':
                    lv = value/100
                    
        self.filterYears()
        self.ogdf = self.ogdf[self.ogdf['stars'] >= 6]
        self.ogdf = self.ogdf[self.ogdf['rating'].isin(['R', 'PG-13', 'PG', 'G', 'NC-17'])]
        self.ogdf['cos_sim'] = self.ogdf.apply(lambda x: self.cosine_with_description(desc, x), axis=1)
        self.ogdf['lev_dist'] = self.ogdf.apply(lambda x: self.levenstein_with_title(self.selections['title'], x['title']), axis=1)
        self.ogdf['score'] = self.ogdf['cos_sim'] * cs + self.ogdf['lev_dist'] * lv
        self.ogdf = self.ogdf.sort_values(by='score', ascending=False)

    def displayRecs(self):
        while True:
            print('\nYour Selections:\n')
            if self.selections is None:
                print('No selections yet')
            else:
                for item in self.selections[['IMDB_id', 'title', 'stars']].values:
                    print(f"{item[0]}:   {item[1]}  \n   IMDB Rating: {item[2]}/10 stars\n")
            print('\nCurrent Recommendations: \n')
            for item in self.recommendations[['IMDB_id', 'title', 'stars']].values:
                print(f"{item[0]}:   {item[1]}  \n   IMDB Rating: {item[2]}/10 stars\n")
            input('\nPress Enter to go back')
            break
        
    def main(self):
        self.processFilters()
        self.recommendations = self.ogdf.head(10)
        while True:
            print('\nOptions: ')
            print('1. View Current recommendations')
            print('2. Generate new recommendations')
            print('(-1 to Exit)')
            self.menu_selection = input('Enter selection: ')
            if self.menu_selection == '1':
                self.displayRecs()
            elif self.menu_selection == '2':
                self.start()
            elif self.menu_selection == '-1':
                exit()
            else:
                continue
        
    def filterYears(self):
        #75-100% = 0.5 std devs
        #50-75% = 1 std devs
        #25-50% = 1.5 std dev
        #1-25% = 2 std dev
        if 'Euclidean Distance with Year' in self.weights:
            year_weight = self.weights['Euclidean Distance with Year']
        else:
            return
        if year_weight < 25 and year_weight > 0:
            std_year = self.ogdf['year'].std()
            lowerLimit = int(self.selections['year'].mean() - 2 * std_year)
            UpperLimit = int(self.selections['year'].mean() + 2 * std_year)
        elif year_weight < 50 and year_weight > 25:
            lowerLimit = int(self.selections['year'].mean() - 1.5 * std_year)
            UpperLimit = int(self.selections['year'].mean() + 1.5 * std_year)
        elif year_weight < 75 and year_weight > 50:
            lowerLimit = int(self.selections['year'].mean() -  std_year)
            UpperLimit = int(self.selections['year'].mean() +  std_year)
        elif year_weight < 100 and year_weight > 75:
            lowerLimit = int(self.selections['year'].mean() - 0.5 * std_year)
            UpperLimit = int(self.selections['year'].mean() + 0.5 * std_year)
        else:
            return
        self.ogdf = self.ogdf[(self.ogdf['year'] > lowerLimit) & (self.ogdf['year'] < UpperLimit)]

if __name__ == '__main__':
    app = App()
    app.start()
