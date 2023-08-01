from datetime import datetime
from pytz import timezone
import torch


class config:
    seed = 42
    epochs = 10
    batch_size = 16
    image_height = 512
    image_width = 512
    # is_continue = False
    use_amp = True
    freeze = False
    model_name = "tf_efficientnet_b4_ns"
    training_date = str(datetime.now(timezone('Asia/Tokyo')).strftime('%Y-%m-%d'))
    data_path = "dataset"
    weight_path = "weights"
    checkpoint_path = "checkpoints"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    gradient_accumulation_steps = 1
    """
    "0":string"Cassava Bacterial Blight (CBB)"
    "1":string"Cassava Brown Streak Disease (CBSD)"
    "2":string"Cassava Green Mottle (CGM)"
    "3":string"Cassava Mosaic Disease (CMD)"
    "4":string"Healthy"
    """
    num_classes = 5

    lr = 3e-4
    num_workers = 4
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    log_file_name = "logs/" + str(datetime.now(timezone('Asia/Tokyo')).strftime('%Y-%m-%d')) + ".log"
