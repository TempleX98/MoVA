CONTROLLER_HEART_BEAT_EXPIRATION = 30
WORKER_HEART_BEAT_INTERVAL = 15

LOGDIR = "."

# Model Constants
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
IMAGE_PLACEHOLDER = "<image-placeholder>"


CONTROLLER_HEART_BEAT_EXPIRATION = 30
WORKER_HEART_BEAT_INTERVAL = 15

LOGDIR = "."

ROUTING_SYSTEM_PROMPT = '''
You are a helpful language and vision assistant router. Based on the visual content, questions, and model pool the user provides, you need to consider the expertise of these models to select the most suitable models to help you answer the questions. Answer with the model's letter from the given choices directly.'''

prompt = "You are a helpful assistant router. Based on the visual content, questions, and model pool the user provides, you need to consider the expertise of these models to select the most 3 suitable models to help you answer the questions. Answer with the model's letter from the given choices directly. If no models are selected, just answer 'none'.\nModel pool:"

models = [
    "This model can effectively extract the accurate spatial and semantic information from natural images.",
    "This model can handle images with text accurately and efficiently but fails to process natural images.",
    "This model is a state-of-the-art object detector that can identify objects in images.",
    "This model shows remarkable skill in text recognition, attaining top-tier text analysis outcomes across diverse areas",
    "This model is a specialized model designed to achieve state-of-the-art plot and chart understanding performance.",
    "This model is a foundation model designed for biomedical vision-language processing.",
    "This model is a leading image segmentation framework and achieves strong zero-shot segmentation performance.",
]
options = ["A", "B", "C", "D", "E", "F", "G"]

for i in range(len(models)):
    prompt = prompt + "\n" + options[i] + ". " + models[i]
ROUTING_PROMPT = prompt + "\n\nQuestion:\n"
