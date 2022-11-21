import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans  



class App():
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
        search_Term = ''
        print('\n\nSearch for a movie to begin generating recommendations')
        search_Term = input('Enter search term (-1 to quit):   ')
        if search_Term == '-1':
            exit()
        else:
            self.search(search_Term)
    
    def search(self, searchword):
        while True:
            print('\n\nSearch Results: ')
            tmp = self.ogdf[self.ogdf['title'].str.contains(searchword, case=False)][['IMDB_id', 'title', 'stars']]
            for item in tmp.values:
                print(f"{item[0]}: {item[1]}\n   IMDB Rating: {item[2]}/10 stars\n")
            print('\nEnter the ID of the movie you want to add to your list')
            print("press '0' to go back and search again")
            print('-1 to exit')
            selection = input('Enter selection:   ')
            if selection == '' or selection == '0':
                self.start()
            elif selection == '-1':
                exit()
            else:
                try:
                    self.selections = self.ogdf[self.ogdf['IMDB_id'] == selection]
                    if self.selections is None:
                        raise
                except Exception as e:
                    print('Invalid selection')
                    continue
                else:
                    break
        self.pick_Important_Factors()
        
    def pick_Important_Factors(self):
        print('\n\nPick at least one of these factors that are important to you')
        print('And we will tailor your recommendations towards those factors')
        print('Press -1 to go back to select a different movie')
        print("Press '0', or Enter to continue")
        print('press -2 to exit')
        count = 0
        feature_Map = {'1': 'Genre', '2': 'Title'}
        while count < 2 or selection != '0':
            selection = ''
            print('1. Genre')
            print('2. Title')
            selection = input('Enter the number corresponding to your selection:   ')
            if int(selection) > 0:
                count+=1
                self.features.append(feature_Map[selection])
                if count == 2:
                    break
                continue
            elif count == '0':
                break
            elif(int(selection) == -1):
                self.search()
            elif(int(selection) == -2):
                exit()
            else:
                print('Invalid selection')
                continue
        self.choose_Cluster()
        
    def choose_Cluster(self):
        while True:
            print('\n\nSelect How many Clusters you would like to use')
            print('-1 to exit, 0 to go back to change your factor selections') 
            selection = ''
            selection = input('Enter number of clusters:   ')
            if not selection.isdigit():
                print('Invalid selection')
                continue
            elif selection == '0':
                self.pick_Important_Factors()
            elif selection == '-1':
                exit()
            else:
                self.num_clusters = int(selection)
                break
        self.choose_Filters()
    
    def choose_Filters(self):
        filtermap = {'1': self.cosine_with_description, '2': self.levenstein_with_title, '3': self.euclidean_with_year}
        filtermap2 = {'1': 'Cosine Similarity with Description', '2': 'Levenstein Distance with Title', '3': 'Euclidean Distance with Year'}
        options = [{'1': 'Use cosine similarity on the description of the movies'}, {'2': 'Use Levenstein Distance on the title of the movies'}, {'3': 'Use Euclidean Distance on the year of the movies'}]
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
                self.choose_Cluster()
            if selection == '-2':
                exit()
            elif not selection.isdigit() or int(selection) > 3 or int(selection) < -1:
                print('invalid selection')
            else:
                weight = self.choose_Weight()
                if weight == -1:
                    continue
                else:
                    cp = options.copy()
                    for idx, item in enumerate(cp):
                        for key, value in item.items():
                            if key == selection:
                                options.pop(idx)
                    self.weights.append({filtermap2[selection] : weight})
                    self.filters.append({filtermap[selection] : weight})
                    counts += 1
                    continue
        self.main()
        
    def choose_Weight(self):
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
            elif not selection.isdigit():
                print('invalid selection')
            if int(selection) > weightleft:
                print('That value is too high')
                continue
            elif int(selection) <= 0:
                print('value must be greater than 0')
                continue
            else:
                return int(selection)
        
    def cosine_with_description(self, weight):
        print(f'\ncosine with description({weight}%)')
        
    def levenstein_with_title(self, weight):
        print(f'\nlevenstein with title({weight}%)')
        
    def euclidean_with_year(self, weight):
        print(f'\neuclidean with year({weight}%)')
        
    def Cluster(self):
        print(f'\nClustering with {self.num_clusters} clusters')
        
    def processFilters(self):
        self.Cluster()
        for item in self.filters:
            for func, param in item.items():
                func(param)  
        
    def main(self):
        self.processFilters()          
    
        

if __name__ == '__main__':
    app =App()
    app.start()
    
    
    