
import pandas as pd


class Clean_Data:
    def __init__(self, data: pd.DataFrame):
        """Initializes for cleaning of the data"""
        self.data = data

    def cleaner(self):
        """Cleans the dataset by handling missing values and removing outliers."""
        # Handle missing numerical values by filling with the median
        for col in self.data.select_dtypes(include=["float64", "int64"]).columns:
            self.data[col].fillna(self.data[col].median(), inplace=True)

        # Handle missing categorical values by filling with the mode
        for col in self.data.select_dtypes(include=["object"]).columns:
            self.data[col].fillna(self.data[col].mode()[0], inplace=True)

        # Drop columns that are not needed
        columns_to_drop = ["Unnamed: 0", "municipality_code", "locality", "postal_code"]
        self.data.drop(columns=columns_to_drop, errors="ignore", inplace=True)

        # Remove outliers based on the 'price' column (using 1st and 99th percentiles)
        self.data = self.data[
            (self.data["price"] > self.data["price"].quantile(0.01))
            & (self.data["price"] < self.data["price"].quantile(0.99))
        ]

        # Return the cleaned dataframe
        return self.data



    
