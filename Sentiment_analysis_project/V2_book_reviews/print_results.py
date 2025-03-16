import torch
from V2_book_reviews import logger
from V2_book_reviews.Train import train_tools
from V2_book_reviews.Model.model import SentimentAnalysisModel, SentimentAnalysisModel_V0
from V2_book_reviews.Model import model_tools

directory = "C:/Users/julie/Desktop/All/Important/Polytech/Inge_3/AI/Sentitment_analysis_project/Results/"

# Model 1
#logger_filename = directory + "Imbalanced_dataset_Logger_2024-11-16_17-44_ns200000_sr0.8_epochs5_lr0.0001.pkl"
#model_filename = directory + "Imbalanced_dataset_Model_2024-11-16_17-44_ns200000_sr0.8_epochs5_lr0.0001_state_dict.pth"

# Model 2
logger_filename = directory + "Balanced_Dataset_Logger_2024-11-17_14-04_ns100000_sr0.85_epochs50_lr2e-05.pkl"
model_filename = directory + "Balanced_Dataset_Model_2024-11-17_14-04_ns100000_sr0.85_epochs50_lr2e-05_state_dict.pth"

# Model 3
#logger_filename = directory + "Balanced_Dataset_Logger_2024-11-17_15-10_ns600000_sr0.85_epochs1_lr2e-05.pkl"
#model_filename = directory + "Balanced_Dataset_Model_2024-11-17_15-10_ns600000_sr0.85_epochs1_lr2e-05_state_dict.pth"

def main():
    print("Loading model...")
    # Load the model
    model_loaded = SentimentAnalysisModel()
    #model_loaded = SentimentAnalysisModel_V0()
    model_loaded.load_state_dict(torch.load(model_filename, weights_only=True, map_location=torch.device('cpu')))

    print("Loading logger...")
    # Load the logger and print training results
    training_logger_loaded = logger.ModelTrainingLogger()
    training_logger_loaded.load_from_file(logger_filename)
    train_tools.plot_training_results(training_logger_loaded)

    model_tools.use_model(model_loaded)


if __name__ == '__main__':
    main()