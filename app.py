# Import the libraries here
import os as os
from datasets import load_dataset
import json as json
from enum import Enum
import random as random
from openai import OpenAI


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