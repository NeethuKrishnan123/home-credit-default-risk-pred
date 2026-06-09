from preprocessing import preprocess
from feature_engineering import features
from feature_selection import feature_selection
from train_model import train_model

def run_pipeline():

    print("Loading Data")
    df = preprocess()

    print("Feature Engineering")
    df = features(df)

    print("Feature Selection")
    df = feature_selection(df)
    
    print("Training Model")
    train_model(df)

    print("Pipeline Completed")

if __name__ == "__main__":
    run_pipeline()


