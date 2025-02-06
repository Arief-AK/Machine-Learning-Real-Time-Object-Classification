from include.Logger import Logger
from include.TensorModel import TensorModel

if __name__ == "__main__":
    # Create a logger
    logger = Logger(__name__)

    # Initalise model handler
    model_handler = TensorModel()
    model_handler.load_data()