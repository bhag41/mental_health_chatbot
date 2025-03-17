# Import the libraries here
import os as os
from datasets import load_dataset
import json as json
from enum import Enum
import random as random
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
#  Add OpenAI api_key
api_key = os.getenv('OPENAI_API_KEY')
if not api_key: raise ValueError("API key not found. Please set the OPENAI_API_KEY environment variable.")

#  Define the RoleType Enum here
class RoleType(Enum):
    USER = 'user'
    SYSTEM = 'system'
    ASSISTANT = 'assistant'
    
# Define the Role class here
class Role(object):
    def __init__(self, role_type:RoleType, content):
        self.role = role_type.value
        self.content = content
        self.value = {'role': self.role, 'content':self.content } 

#  Define the messsage class here
class Message(object):
    def __init__(self, user_content, system_content, assistant_content):
        self.user_role = Role(role_type=RoleType.USER , content=user_content)
        self.system_role = Role(role_type=RoleType.SYSTEM , content=system_content)
        self.assistant_role = Role(role_type=RoleType.ASSISTANT, content=assistant_content)
        self.message = {'messages':[self.system_role.value, self.user_role.value, self.assistant_role.value, ]}

# Load the dataset 
dataset = load_dataset("Amod/mental_health_counseling_conversations", split = 'train')

# Create a sample Message object
context = dataset[152]['Context']
response = dataset[152]['Response']
system_content = "You serve as a supportive and honest psychology and psychotherapy assistant. Your main duty is to offer compassionate, understanding, and non-judgmental responses to users seeking emotional and psychological assistance. Respond with empathy and exhibit active listening skills. Your replies should convey that you comprehend the user’s emotions and worries. In cases where a user mentions thoughts of self-harm, suicide, or harm to others, prioritize their safety. Encourage them to seek immediate professional help and provide emergency contact details as needed. It’s important to note that you are not a licensed medical professional. Refrain from diagnosing or prescribing treatments. Instead, guide users to consult with a licensed therapist or medical expert for tailored advice. Never store or disclose any personal information shared by users. Uphold their privacy at all times. Avoid taking sides or expressing personal viewpoints. Your responsibility is to create a secure space for users to express themselves and reflect. Always aim to foster a supportive and understanding environment for users to share their emotions and concerns. Above all, prioritize their well-being and safety."
message_obj = Message(user_content=context, system_content=system_content, assistant_content = response )

print(message_obj.message)

#  Create the train_dataset variable
# Sample 100 items from the 'train' split
sampled_dataset = random.choices(dataset, k=100)
train_dataset = []

# Print the sampled data to verify
print(sampled_dataset[1])

for row in sampled_dataset:
    message_obj = Message(user_content=row['Context'], system_content=system_content, assistant_content=row['Response'])
    train_dataset.append(message_obj.message)

print(train_dataset[1])

# Save data in JSONl format 
def save_to_jsonl(data, file_path):
    with open(file_path, 'w') as file:
        for row in data:
            line = json.dumps(row)
            file.write(line + '\n')


# Store the data in JSONL format
training_data_path = '/data/train.jsonl'
save_to_jsonl(train_dataset[:-5], training_data_path)

validation_data_path = '/data/validation.jsonl'
save_to_jsonl(train_dataset[-5:], validation_data_path)

# Load the training and validation files
training_data = open(training_data_path, "rb")
validation_data = open(validation_data_path, "rb")

# Initialize the OpenAI client
client = OpenAI(api_key=api_key)

# Upload the training and validation files
training_response = client.files.create(file=training_data, purpose="fine-tune")
training_file_id = training_response.id

validation_response = client.files.create(file=validation_data, purpose="fine-tune")
validation_file_id = validation_response.id

print("Training file id:", training_file_id)
print("Validation file id:", validation_file_id)

# Create a fine-tuning job
response = client.fine_tuning.jobs.create(
    training_file=training_file_id,
    model="gpt-3.5-turbo",
    suffix="my-test-model",
    validation_file=validation_file_id
)

job_id = response.id

print(response)

# Retrieve the job status
job_id = response.id

job_status = client.fine_tuning.jobs.retrieve(job_id)
print(job_status)

# Create and store message dictionaries
system_message = """You serve as a supportive and honest psychology and psychotherapy assistant. Your main duty is to offer compassionate, understanding, and non-judgmental responses to users seeking emotional and psychological assistance. Respond with empathy and exhibit active listening skills. Your replies should convey that you comprehend the user's emotions and worries. In cases where a user mentions thoughts of self-harm, suicide, or harm to others, prioritize their safety. Encourage them to seek immediate professional help and provide emergency contact details as needed. It's important to note that you are not a licensed medical professional. Refrain from diagnosing or prescribing treatments. Instead, guide users to consult with a licensed therapist or medical expert for tailored advice. Never store or disclose any personal information shared by users. Uphold their privacy at all times. Avoid taking sides or expressing personal viewpoints. Your responsibility is to create a secure space for users to express themselves and reflect. Always aim to foster a supportive and understanding environment for users to share their emotions and concerns. Above all, prioritize their well-being and safety."""

messages = []
messages.append({"role": "system", "content": system_message})
user_message = "Every winter I find myself getting sad because of the weather. How can I fight this?"
messages.append({"role": "user", "content": user_message})

# Test the fine-tuned chat completion model
completion = client.chat.completions.create(
    model= response.model,
    messages=messages
)
print(completion.choices[0].message)

# Get and compare the output of the gpt-3.5-turbo chat completion model
completion = client.chat.completions.create(
    model= "gpt-3.5-turbo",
    messages=messages
)
print(completion.choices[0].message)