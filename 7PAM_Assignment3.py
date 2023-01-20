#Importing libraries
import pandas as pnd
import csv
import matplotlib.pyplot as mtpltlib
from matplotlib.lines import Line2D
from sklearn.cluster import KMeans
from scipy.optimize import curve_fit
import warnings 
warnings.filterwarnings("ignore")


#Loading data
def load_data(file):
    """This function takes in a csv file in the Worldbank format and returns a transposed version as well as the original
    version in the dataframe format of pandas.
    
    Parameters
    ----------
    csv_file : str
        The filename of the csv file.

    Returns
    -------
    final : pandas dataframe
        A dataframe containing the original csv data.
    transposed_csv : pandas dataframe
        A dataframe containing the transposed csv data.
    """
    #empty list created to store the data
    final = []
    #opening the file in read mode
    with open (file, 'r') as file_handle:
        #reading it using csv.reader of python
        reading = csv.reader(file_handle, delimiter=",")
        #appending the data values in the list
        for _ in reading:
            final.append(_)
    #converting list into a dataframe
    final = pnd.DataFrame(final)
    #transposing the dataframe
    transposed = final.T
    #returns original dataframe and transposed dataframe
    return final, transposed


#calling the function and storing the dataframes in 2 variables
original_version, trasnposed_version = load_data("API_19_DS2_en_csv_v2_4756035.csv")


original_version


#using slicing to remove the first 4 rows of the dataframe
original_version = original_version[4:]


#Taking mean for various years and storing them in new columns to represent decade-wise data
climate = pnd.read_csv("API_19_DS2_en_csv_v2_4756035.csv", skiprows =3)

column1 = ['1960','1961','1962','1963','1964','1965','1966','1967','1968','1969']
column2 = ['1970','1971','1972','1973','1974','1975','1976','1977','1978','1979']
column3 = ['1980','1981','1982','1983','1984','1985','1986','1987','1988','1989']
column4 = ['1990','1991','1992','1993','1994','1995','1996','1997','1998','1999']
column5 = ['2000','2001','2002','2003','2004','2005','2006','2007','2008','2009']
column6 = ['2010','2011','2012','2013','2014','2015','2016','2017','2018','2019']
column7 = ['2020','2021']
climate['1960s'] = climate[column1].mean(axis=1,skipna = True)
climate['1970s'] = climate[column2].mean(axis=1,skipna = True)
climate['1980s'] = climate[column3].mean(axis=1,skipna = True)
climate['1990s'] = climate[column4].mean(axis=1,skipna = True)
climate['2000s'] = climate[column5].mean(axis=1,skipna = True)
climate['2010s'] = climate[column6].mean(axis=1,skipna = True)
climate['2020s'] = climate[column7].mean(axis=1,skipna = True)
climate.head()

#Dropping unnecessary columns
columns = ['1960','1961','1962','1963','1964','1965','1965','1966','1967','1968',
           '1969','1970','1971','1972','1973','1974','1975','1976','1977','1978','1979','1980',
           '1981','1982','1983','1984','1985','1986','1987','1988','1989','1990','1991','1992','1993',
           '1994','1995','1996','1997','1998','1999','2000','2001','2002','2003','2004','2005','2006',
           '2007','2008','2009','2010','2011','2012','2013','2014','2015','2016','2017','2018','2019',
           '2020','Unnamed: 66','2021','Indicator Code','Country Code']
#Setting the column indicator name as index for data 
climate.set_index("Indicator Name",inplace = True)
#Filling null values with 0
climate.fillna(value = 0, axis=1, inplace = True)
climate.fillna(value = 0, axis=0, inplace = True)
climate.head(80)

#Taking out the data of 10 countries and sorting it
new_climate = climate[climate["Country Name"].isin(["United States","India","Japan","Australia","United Kingdom","China","France","Canada","Russia","Brazil","Germany"])]
new_climate.sort_values(by=['Country Name'], inplace = True)
new_climate.head()

#Taking out the data related to urban population (% of total population) for the chosen countries
climate_urban = new_climate.loc['Urban population (% of total population)']
#Plotting the data
climate_urban.plot(x='Country Name', y=['1980s','1990s','2000s','2010s'], kind='bar', stacked =False, figsize=(10,6))
#Title for the chart
mtpltlib.title('Country-wise urban population (% of total population)', fontsize = 25)
# generating legend
mtpltlib.legend(title='Decades', fontsize='15', bbox_to_anchor=(1.05, 1.0), loc='upper left')
# generating labels
mtpltlib.xlabel('Country Name',fontsize='20')
mtpltlib.ylabel('Urban population (% of total population)',fontsize=17)
mtpltlib.show()


#setting the first row as the column names for the dataframe
original_version.columns = original_version.iloc[0]


#dropping the row and unnecessary columns
original_version.drop(4, inplace=True)
original_version.drop(columns=["Country Code", "Indicator Code", ""], inplace = True)



#filling the missing cells with the value zero
original_version.fillna(value = 0, axis=1, inplace = True)
original_version.fillna(value = 0, axis=0, inplace = True)
# #replacing any blank strings with zero
original_version.replace("", 0, inplace=True)



original_version.info()


#converting the data types of the columns to float
original_version[original_version.columns[2:64]] = original_version[original_version.columns[2:64]].astype(float)



original_version.info()


# presenting transposed version from load_data function
trasnposed_version


#dropping unnecessary columns
trasnposed_version.drop(columns= [0,1,2,3], inplace=True)


trasnposed_version


#setting the column names
trasnposed_version.columns = trasnposed_version.iloc[0]


# trasnposed_version.drop(columns=["Country Code", "Indicator Code"], inplace = True)
trasnposed_version.drop([0,1,3], inplace=True)


trasnposed_version


#replacing null values with zero
trasnposed_version.fillna(value = 0, axis=1, inplace = True)
trasnposed_version.fillna(value = 0, axis=0, inplace = True)
#replacing "" with zero
trasnposed_version.replace("", 0, inplace=True)


trasnposed_version


#checking the names of the indicators
original_version["Indicator Name"].unique()


#Selecting the column for grouping the data according to it
attribute = original_version.groupby(["Indicator Name"])
#Getting the data related to only one value, "Urban population growth (annual %)" of the column,"Indicator Name"
original_version1 = attribute.get_group("Urban population growth (annual %)")


#final dataset 
original_version1


#getting only numerical data from the dataset to build the model
original_version2 = original_version1[original_version1.columns[2:65]]
original_version2


#K means model with 5 clusters
cluster_model = KMeans(n_clusters = 5)


#training the model
clusters = cluster_model.fit_predict(original_version2)


#creating a blank dataframe
data_=pnd.DataFrame()
#storing the value of clusters generated from the k-means model into the column "Clusters" of this dataframe
data_["Clusters"] = clusters
#checking unique cluster values
values_ = data_["Clusters"].unique()
values_


index = list(original_version1.index.values)
#setting the index of data_ dataframe as that of the original_version1 so that they can be merged later on
data_["index"]=index
data_.set_index(["index"], drop=True)



original_version1["index"]=index
original_version1.set_index(["index"], drop=True)



#merging both the datasets
original_version1 = pnd.merge(original_version1, data_)



#dropping the unnecessary column index
original_version1.drop(columns = ["index"], inplace = True)



original_version1



#Number of values in each cluster
original_version1.Clusters.value_counts()


#computing the centroids for each cluster
centroid_vals = cluster_model.cluster_centers_
data1=[]
data2=[]
#getting the centroid value of only 2017 and 2021
for i in centroid_vals:
    for j in range(len(i)):
        x = i[57]
        y = i[61]
    data1.append(i[57])
    data2.append(i[61])



#defining the colors in a list
colors = ['#fc0303', '#fa05f2', '#9b05ff', '#0533ed', "#029925"]
#mapping the colors according to unique values in the column Clusters
original_version1['c'] = original_version1.Clusters.map({0:colors[0], 1:colors[1], 2:colors[2], 3:colors[3], 4:colors[4]})
#initiating a figure
fig, ax = mtpltlib.subplots(1, figsize=(15,8))
#plotting a scatter plot of data
mtpltlib.scatter(original_version1["2017"], original_version1["2021"], c=original_version1.c, alpha = 0.7, s=40)
#plotting a scatter plot of centroids
mtpltlib.scatter(data1, data2, marker='^',facecolor=colors,edgecolor="black", s=100)
#getting the legend for data
legend_elements = [Line2D([0], [0], marker='o', color='w', label='Cluster or C{}'.format(i+1), 
                   markerfacecolor=mcolor, markersize=5) for i, mcolor in enumerate(colors)]
#getting the legend for centroids
centroid_legend = [Line2D([0], [0], marker='^', color='w', label='Centroid of C{}'.format(i+1), 
                   markerfacecolor=mcolor,markeredgecolor="black", markersize=10) for i, mcolor in enumerate(colors)]
#final legend elements
legend_elements.extend(centroid_legend)
#setting the legend
mtpltlib.legend(handles=legend_elements, loc='upper right', title="Clusters", fontsize=10, bbox_to_anchor=(1.15,1))
#setting xlabel, ylabel and title
mtpltlib.xlabel("2017 data in %", fontsize='18')
mtpltlib.ylabel("2021 data in %", fontsize='18')
mtpltlib.title("Basic K-means clustering on the basis of urban population growth (annual %)", fontsize='20')



#Building a function to fit the model using the curve_fit of scipy
def data_fit(frame, col_1, col_2):
    """This function takes in a dataframe and columns to fit the model using the curve_fit methodology of the scipy library.
    
    Parameters
    ----------
    csv_fileframe : pandas dataframe
        The name of the pandas dataframe.
    col_1 : str
        The name of the column of dataframe to be taken as x.  
    col_2 : str
        The name of the column of dataframe to be taken as y.  

    Returns
    -------
    errors : numpy array
        An array containing the errors or ppot.
    covariance : numpy array
        An array containing the covariances.
    """
    col1_data = frame[col_1]
    col2_data = frame[col_2]
    
    def mod_func(x, m, b):
        return m*x+b
    
    #calling curve_fit
    errors,covariance = curve_fit(mod_func, col1_data, col2_data)
    mtpltlib.figure(figsize=(15,8))
    #plotting the data 
    mtpltlib.plot(col1_data, col2_data, "bo", label="data", color ="g")
    #plotting the best fitted line
    mtpltlib.plot(col1_data, mod_func(col1_data, *errors), "b-", label="Best Fit")
    mtpltlib.xlabel(col_1)
    mtpltlib.ylabel(col_2)
    mtpltlib.legend(bbox_to_anchor=(1,1))
    mtpltlib.title("Plotting best fit for urban population growth (annual %)")
    return errors, covariance
        

#calling the function
data_fit(original_version1, "2017", "2021")


