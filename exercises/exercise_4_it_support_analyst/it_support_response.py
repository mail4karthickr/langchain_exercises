from langchain_core.pydantic_v1 import BaseModel, Field

class ITSupportResponse(BaseModel):
    orig_message: str = Field("The original customer message")
    orig_lang: str = Field("Detected language of the customer message e.g. Spanish")
    category: str = Field("1-2 word describing the category of the problem")
    trans_message: str = Field("Translated customer message in English")
    response: str = Field("Response to the customer in orig_lang")
    trans_response: str = Field("Response to the customer in English")