# baltimore-housing-crime
An analysis of data from Open Baltimore for WGU D502 Capstone project; evaluating the relationship between homicides and building vacancies per CSA (Community Statistical Area), controlling for economic and population variables.
## Datasets and Source URLs
* NIBRS Homicide Data | https://data.baltimorecity.gov/maps/204beefe92a645d79fdf0969957bbdf8
* Vacant Building Notices | https://data.baltimorecity.gov/maps/691d65a5f85640e6aaa46930bd9dc102
* Median Household Income | https://data.baltimorecity.gov/maps/8613366cfbc7447a9efd9123604c65c1
* Population | https://data.baltimorecity.gov/maps/56d5b4e5480049e98315c2732aa48437
* CSAs (Community Statistical Areas) Reference | https://data.baltimorecity.gov/maps/9c96ae20e6cc41258015c2fd288716c4
## Directories
* raw_data - contains raw imported datasets and exploratory analysis notebook
* clean_data - contains cleaned and transformed datasets
## Files
* merge.py - used to merge cleaned datasets into single dataset for regression analysis
* regression.py - used to execute the regression analysis
* raw_datasets.zip - contains all raw datasets
* README.md - contains project overview
* requirements.txt - contains project dependencies documentation