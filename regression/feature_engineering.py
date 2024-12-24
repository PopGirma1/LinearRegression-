import pandas as pd
import numpy as np

class FeatureEngineering:
    def __init__(self, data):
        """Initializes the feature engineering class with the provided DataFrame."""
        self.df = data

    def handle_missing_values(self):
        """Cleans the dataset by handling missing values and removing outliers."""
        # Handle missing numerical values by filling with the median
        for col in self.df.select_dtypes(include=["float64", "int64"]).columns:
            self.df[col].fillna(self.df[col].median(), inplace=True)

        # Handle missing categorical values by filling with the mode
        for col in self.df.select_dtypes(include=["object"]).columns:
            self.df[col].fillna(self.df[col].mode()[0], inplace=True)

    def total_area(self):
        """Adds a feature for the total area (livingarea + surfaceoftheplot)."""
        if 'livingarea' in self.df.columns and 'surfaceoftheplot' in self.df.columns:
            self.df['Total_Area'] = self.df['livingarea'] + self.df['surfaceoftheplot']

    def add_property_features_count(self):
        """Adds a feature that counts the total number of property features."""
        property_feature_columns = ['kitchen', 'loft', 'terrace', 'garden', 'pool', 'gardensurface']
        if all(col in self.df.columns for col in property_feature_columns):
            self.df['total_property_features'] = self.df[property_feature_columns].sum(axis=1)

    def add_average_room_size(self):
        """Adds a feature for the average room size (livingarea / bedrooms)."""
        if 'livingarea' in self.df.columns and 'bedrooms' in self.df.columns:
            self.df['average_room_Size'] = np.where(
                self.df['bedrooms'] > 0,
                self.df['livingarea'] / self.df['bedrooms'],
                0
            )

    def price_per_sqm(self):
        """Adds a feature for price per square meter (price / Total_Area)."""
        if 'price' in self.df.columns and 'Total_Area' in self.df.columns:
            self.df['price_per_sqm'] = np.where(
                self.df['Total_Area'] > 0,
                self.df['price'] / self.df['Total_Area'],
                0
            )


    def feature_engineering(self):
        """
        Runs all feature engineering methods to enhance the dataset.
        Returns the transformed DataFrame.
        """
        self.handle_missing_values()  # Handle missing data first
        self.total_area()  # Add total area
        self.add_property_features_count()  # Count total amenities
        self.add_average_room_size()  # Calculate average room size
        self.price_per_sqm()
        return self.df  # Return the transformed DataFrame
